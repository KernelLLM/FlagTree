module {
  func.func @run_non_tail_fused_e6986ee2ee0cde4e7e6be3d2a930cbef(%arg0: !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg1: !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: i64, %arg5: i64, %arg6: !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>) attributes {cannbotdsl.contract_digest = "e6986ee2ee0cde4e7e6be3d2a930cbef0c8325cb8ce80e6b02194f6df5da4950", cannbotdsl.public_slot_count = 7 : i64, cannbotdsl.slot_dtype_bits = [32, 32, 8, 8, 64, 64, 64], cannbotdsl.slot_dtype_codes = [2, 2, 1, 1, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 2, 2, 1], cannbotdsl.slot_type_digests = ["299b8584bfb7b4ecbed3e029654203d20256497197396c46e927fd078ee9206a", "299b8584bfb7b4ecbed3e029654203d20256497197396c46e927fd078ee9206a", "0466425dd1673c3d734b77386a8d2c26ab0afa3f2945f4af539e71f1f55f41c9", "1d2672dce8d9ff4fcf6d37f57043432984990a93dd5ece4b4944846a06c54afa", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "4126671a83041647cd98afed881383228865da287ea44573939f32b1d956765b"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel_run_non_tail_fused_0(%arg5) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg6] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 5>, name = "_DynamicMxQuantSrKernel_run_non_tail_fused", op_type = "_DynamicMxQuantSrKernel_run_non_tail_fused", roles = array<i64: 0, 0, 1, 1, 0>}} : !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel_run_non_tail_fused_0(%arg0: !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg1: !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: i64, %arg5: !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel_run_non_tail_fused_0", cannir.resource.ids_prepared} {
    %c1664_i32 = arith.constant 1664 : i32
    %c26624_i32 = arith.constant 26624 : i32
    %c832_i32 = arith.constant 832 : i32
    %c13312_i32 = arith.constant 13312 : i32
    %c1 = arith.constant 1 : index
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
    %alloc_3 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<2xi64>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<13312xf32, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 53248 : i64} : memref<13312xf32, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 106496 : i64} : memref<13312xui8, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 119808 : i64} : memref<13312xui8, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 133120 : i64} : memref<832xui8, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 133952 : i64} : memref<832xui8, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 136576 : i64} : memref<3328xi32, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 163200 : i64} : memref<13312xui8, 1>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 176512 : i64} : memref<832xui8, 1>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 177344 : i64} : memref<26624xui8, 1>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 203968 : i64} : memref<832xui16, 1>
    %0 = ascscalar.scalar_ptr %alloc_3[%c0] : memref<2xi64>, index -> <gm, i64>
    %1 = ascscalar.load_dev %0 : !ascscalar.scalar_ptr<gm, i64> -> i64
    %2 = ascscalar.scalar_ptr %alloc_3[%c1] : memref<2xi64>, index -> <gm, i64>
    %3 = ascscalar.load_dev %2 : !ascscalar.scalar_ptr<gm, i64> -> i64
    %4 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
    %5 = cannir.extract_shape %4 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, -1)>
    %6 = cannir.extract_dim %5[1] : !cannir.shape<(-1, -1)>
    %7 = arith.remsi %6, %c64_i64 : i64
    %8 = arith.cmpi ne, %7, %c0_i64 : i64
    %9 = arith.cmpi slt, %7, %c0_i64 : i64
    %10 = arith.andi %8, %9 : i1
    %11 = arith.divsi %6, %c64_i64 : i64
    %12 = arith.extui %10 : i1 to i64
    %13 = arith.subi %11, %12 : i64
    %14 = cannir.extract_dim %5[0] : !cannir.shape<(-1, -1)>
    %15 = arith.muli %14, %13 : i64
    %16 = ascscalar.get_block_num
    %17 = arith.addi %15, %16 : i64
    %18 = arith.subi %17, %c1_i64 : i64
    %19 = arith.remsi %18, %16 : i64
    %20 = arith.cmpi ne, %19, %c0_i64 : i64
    %21 = arith.cmpi slt, %19, %c0_i64 : i64
    %22 = arith.cmpi slt, %16, %c0_i64 : i64
    %23 = arith.xori %21, %22 : i1
    %24 = arith.andi %20, %23 : i1
    %25 = arith.divsi %18, %16 : i64
    %26 = arith.extui %24 : i1 to i64
    %27 = arith.subi %25, %26 : i64
    %28 = arith.addi %27, %c207_i64 : i64
    %29 = arith.remsi %28, %c208_i64 : i64
    %30 = arith.cmpi ne, %29, %c0_i64 : i64
    %31 = arith.cmpi slt, %29, %c0_i64 : i64
    %32 = arith.andi %30, %31 : i1
    %33 = arith.divsi %28, %c208_i64 : i64
    %34 = arith.extui %32 : i1 to i64
    %35 = arith.subi %33, %34 : i64
    %36 = arith.addi %27, %35 : i64
    %37 = arith.subi %36, %c1_i64 : i64
    %38 = arith.remsi %37, %35 : i64
    %39 = arith.cmpi ne, %38, %c0_i64 : i64
    %40 = arith.cmpi slt, %38, %c0_i64 : i64
    %41 = arith.cmpi slt, %35, %c0_i64 : i64
    %42 = arith.xori %40, %41 : i1
    %43 = arith.andi %39, %42 : i1
    %44 = arith.divsi %37, %35 : i64
    %45 = arith.extui %43 : i1 to i64
    %46 = arith.subi %44, %45 : i64
    %47 = arith.addi %46, %c3_i64 : i64
    %48 = arith.remsi %47, %c4_i64 : i64
    %49 = arith.cmpi ne, %48, %c0_i64 : i64
    %50 = arith.cmpi slt, %48, %c0_i64 : i64
    %51 = arith.andi %49, %50 : i1
    %52 = arith.divsi %47, %c4_i64 : i64
    %53 = arith.extui %51 : i1 to i64
    %54 = arith.subi %52, %53 : i64
    %55 = arith.muli %54, %c4_i64 : i64
    %56 = arith.muli %54, %c256_i64 : i64
    %57 = arith.addi %13, %55 : i64
    %58 = arith.subi %57, %c1_i64 : i64
    %59 = arith.remsi %58, %55 : i64
    %60 = arith.cmpi ne, %59, %c0_i64 : i64
    %61 = arith.cmpi slt, %59, %c0_i64 : i64
    %62 = arith.cmpi slt, %55, %c0_i64 : i64
    %63 = arith.xori %61, %62 : i1
    %64 = arith.andi %60, %63 : i1
    %65 = arith.divsi %58, %55 : i64
    %66 = arith.extui %64 : i1 to i64
    %67 = arith.subi %65, %66 : i64
    %68 = arith.muli %14, %67 : i64
    %69 = ascscalar.get_block_idx
    scf.for %arg6 = %69 to %68 step %16  : i64 {
      %70 = arith.remsi %arg6, %67 : i64
      %71 = arith.cmpi ne, %70, %c0_i64 : i64
      %72 = arith.cmpi slt, %70, %c0_i64 : i64
      %73 = arith.cmpi slt, %67, %c0_i64 : i64
      %74 = arith.xori %72, %73 : i1
      %75 = arith.andi %71, %74 : i1
      %76 = arith.divsi %arg6, %67 : i64
      %77 = arith.extui %75 : i1 to i64
      %78 = arith.subi %76, %77 : i64
      %79 = arith.select %75, %67, %c0_i64 : i64
      %80 = arith.addi %70, %79 : i64
      %81 = arith.muli %80, %56 : i64
      %82 = arith.subi %6, %81 : i64
      %83 = arith.cmpi sgt, %82, %56 : i64
      %84 = arith.select %83, %56, %82 : i64
      %85 = arith.minsi %78, %14 : i64
      %86 = arith.subi %14, %85 : i64
      %87 = arith.minsi %86, %c1_i64 : i64
      %88 = arith.cmpi sle, %78, %14 : i64
      %89 = arith.select %88, %87, %c0_i64 : i64
      %90 = arith.maxsi %56, %c1_i64 : i64
      %91 = arith.divsi %6, %90 : i64
      %92 = arith.minsi %80, %91 : i64
      %93 = arith.muli %92, %56 : i64
      %94 = arith.subi %6, %93 : i64
      %95 = arith.minsi %56, %94 : i64
      %96 = arith.cmpi sle, %80, %91 : i64
      %97 = arith.select %96, %95, %c0_i64 : i64
      %98 = cannir.extract_stride %4 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
      %99 = cannir.extract_dim %98[0] : !cannir.stride<(-1, 2)>
      %100 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
      %101 = cannir.extract_stride %100 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
      %102 = cannir.extract_dim %101[0] : !cannir.stride<(-1, 2)>
      %103 = arith.muli %78, %102 : i64
      %104 = arith.muli %81, %c2_i64 : i64
      %105 = arith.addi %103, %104 : i64
      %106 = arith.muli %105, %c32_i64 : i64
      %107 = ascscalar.memref_address %alloc : memref<1xf32>
      %108 = arith.divui %106, %c8_i64 : i64
      %109 = arith.addi %107, %108 : i64
      %110 = ascscalar.bind_gm_ptr %109 : memref<1xf32>
      %111 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<13312xf32, 1> -> memref<?xf32, 1>
      %112 = arith.cmpi ugt, %89, %c0_i64 : i64
      %113 = arith.cmpi ugt, %97, %c0_i64 : i64
      %114 = arith.andi %112, %113 : i1
      scf.if %114 {
        ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
        %321 = arith.trunci %97 : i64 to i32
        %322 = arith.trunci %89 : i64 to i32
        ascvec.copy_gm2ub_nddma %111[%c0], %110[%c0], %321, %322, %c1_i32, %c1_i32, %c1_i32, %c2_i64, %99, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c13312_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 : memref<?xf32, 1>, memref<1xf32>
        ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
      }
      %115 = arith.muli %78, %c2_i64 : i64
      %116 = arith.muli %115, %6 : i64
      %117 = arith.addi %116, %81 : i64
      %118 = arith.remsi %84, %c256_i64 : i64
      %119 = arith.cmpi ne, %118, %c0_i64 : i64
      %120 = arith.cmpi slt, %118, %c0_i64 : i64
      %121 = arith.andi %119, %120 : i1
      %122 = arith.divsi %84, %c256_i64 : i64
      %123 = arith.extui %121 : i1 to i64
      %124 = arith.subi %122, %123 : i64
      %125 = arith.addi %84, %c1023_i64 : i64
      %126 = arith.remsi %125, %c1024_i64 : i64
      %127 = arith.cmpi ne, %126, %c0_i64 : i64
      %128 = arith.cmpi slt, %126, %c0_i64 : i64
      %129 = arith.andi %127, %128 : i1
      %130 = arith.divsi %125, %c1024_i64 : i64
      %131 = arith.extui %129 : i1 to i64
      %132 = arith.subi %130, %131 : i64
      %133 = arith.trunci %1 : i64 to i32
      %134 = arith.extui %133 : i32 to i64
      %135 = arith.subi %1, %134 : i64
      %136 = arith.remsi %135, %c4294967296_i64 : i64
      %137 = arith.cmpi ne, %136, %c0_i64 : i64
      %138 = arith.cmpi slt, %136, %c0_i64 : i64
      %139 = arith.andi %137, %138 : i1
      %140 = arith.divsi %135, %c4294967296_i64 : i64
      %141 = arith.extui %139 : i1 to i64
      %142 = arith.subi %140, %141 : i64
      %143 = arith.trunci %142 : i64 to i32
      %144 = arith.divui %117, %c16_i64 : i64
      %145 = arith.trunci %144 : i64 to i32
      %146 = arith.extui %145 : i32 to i64
      %147 = arith.subi %144, %146 : i64
      %148 = arith.remsi %147, %c4294967296_i64 : i64
      %149 = arith.cmpi ne, %148, %c0_i64 : i64
      %150 = arith.cmpi slt, %148, %c0_i64 : i64
      %151 = arith.andi %149, %150 : i1
      %152 = arith.divsi %147, %c4294967296_i64 : i64
      %153 = arith.extui %151 : i1 to i64
      %154 = arith.subi %152, %153 : i64
      %155 = arith.trunci %154 : i64 to i32
      %156 = arith.trunci %3 : i64 to i32
      %157 = arith.extui %156 : i32 to i64
      %158 = arith.subi %3, %157 : i64
      %159 = arith.remsi %158, %c4294967296_i64 : i64
      %160 = arith.cmpi ne, %159, %c0_i64 : i64
      %161 = arith.cmpi slt, %159, %c0_i64 : i64
      %162 = arith.andi %160, %161 : i1
      %163 = arith.divsi %158, %c4294967296_i64 : i64
      %164 = arith.extui %162 : i1 to i64
      %165 = arith.subi %163, %164 : i64
      %166 = arith.trunci %165 : i64 to i32
      %167 = arith.divui %156, %c4_i32 : i32
      %168 = arith.remui %166, %c4_i32 : i32
      %169 = arith.muli %168, %c1073741824_i32 : i32
      %170 = arith.addi %167, %169 : i32
      %171 = arith.divui %166, %c4_i32 : i32
      %172 = arith.muli %132, %c64_i64 : i64
      %173 = arith.divui %144, %c16384_i64 : i64
      %174 = arith.addi %144, %172 : i64
      %175 = arith.subi %174, %c1_i64 : i64
      %176 = arith.divui %175, %c16384_i64 : i64
      %177 = arith.cmpi eq, %173, %176 : i64
      scf.if %177 {
        %321 = arith.trunci %173 : i64 to i32
        %322 = arith.extui %321 : i32 to i64
        %323 = arith.subi %173, %322 : i64
        %324 = arith.remsi %323, %c4294967296_i64 : i64
        %325 = arith.cmpi ne, %324, %c0_i64 : i64
        %326 = arith.cmpi slt, %324, %c0_i64 : i64
        %327 = arith.andi %325, %326 : i1
        %328 = arith.divsi %323, %c4294967296_i64 : i64
        %329 = arith.extui %327 : i1 to i64
        %330 = arith.subi %328, %329 : i64
        %331 = arith.trunci %330 : i64 to i32
        %332 = arith.addi %170, %321 : i32
        %333 = arith.addi %171, %331 : i32
        %334 = arith.cmpi ult, %332, %170 : i32
        %335 = arith.extui %334 : i1 to i32
        %336 = arith.addi %333, %335 : i32
        %337 = arith.extui %332 : i32 to i64
        %338 = arith.muli %337, %c3449720151_i64 : i64
        %339 = arith.trunci %338 : i64 to i32
        %340 = arith.extui %339 : i32 to i64
        %341 = arith.subi %338, %340 : i64
        %342 = arith.remsi %341, %c4294967296_i64 : i64
        %343 = arith.cmpi ne, %342, %c0_i64 : i64
        %344 = arith.cmpi slt, %342, %c0_i64 : i64
        %345 = arith.andi %343, %344 : i1
        %346 = arith.divsi %341, %c4294967296_i64 : i64
        %347 = arith.extui %345 : i1 to i64
        %348 = arith.subi %346, %347 : i64
        %349 = arith.trunci %348 : i64 to i32
        %350 = arith.xori %336, %143 : i32
        %351 = arith.extui %350 : i32 to i64
        %352 = arith.muli %351, %c3449720151_i64 : i64
        %353 = arith.trunci %352 : i64 to i32
        %354 = arith.extui %353 : i32 to i64
        %355 = arith.subi %352, %354 : i64
        %356 = arith.remsi %355, %c4294967296_i64 : i64
        %357 = arith.cmpi ne, %356, %c0_i64 : i64
        %358 = arith.cmpi slt, %356, %c0_i64 : i64
        %359 = arith.andi %357, %358 : i1
        %360 = arith.divsi %355, %c4294967296_i64 : i64
        %361 = arith.extui %359 : i1 to i64
        %362 = arith.subi %360, %361 : i64
        %363 = arith.trunci %362 : i64 to i32
        %364 = arith.addi %133, %c-1640531527_i32 : i32
        %365 = arith.addi %143, %c-1150833019_i32 : i32
        %366 = arith.xori %363, %339 : i32
        %367 = arith.xori %366, %364 : i32
        %368 = arith.extui %367 : i32 to i64
        %369 = arith.muli %368, %c3528531795_i64 : i64
        %370 = arith.trunci %369 : i64 to i32
        %371 = arith.extui %370 : i32 to i64
        %372 = arith.subi %369, %371 : i64
        %373 = arith.remsi %372, %c4294967296_i64 : i64
        %374 = arith.cmpi ne, %373, %c0_i64 : i64
        %375 = arith.cmpi slt, %373, %c0_i64 : i64
        %376 = arith.andi %374, %375 : i1
        %377 = arith.divsi %372, %c4294967296_i64 : i64
        %378 = arith.extui %376 : i1 to i64
        %379 = arith.subi %377, %378 : i64
        %380 = arith.trunci %379 : i64 to i32
        %381 = arith.xori %349, %133 : i32
        %382 = arith.addi %133, %c1013904242_i32 : i32
        %383 = arith.xori %353, %382 : i32
        %384 = arith.addi %143, %c1993301258_i32 : i32
        %385 = arith.xori %380, %384 : i32
        ascsimt.vf(%145, %381, %365, %383, %385, %133, %143, %370, %alloc_10, %172 : i32, i32, i32, i32, i32, i32, i32, i32, memref<3328xi32, 1>, i64) {
        ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<3328xi32, 1>, %arg16: i64):
          %c256_i64_15 = arith.constant 256 : i64
          %c3_i64_16 = arith.constant 3 : i64
          %c2_i64_17 = arith.constant 2 : i64
          %c1_i64_18 = arith.constant 1 : i64
          %c4_i64_19 = arith.constant 4 : i64
          %c1_i32_20 = arith.constant 1 : i32
          %c10_i32 = arith.constant 10 : i32
          %c3_i32 = arith.constant 3 : i32
          %c-1150833019_i32_21 = arith.constant -1150833019 : i32
          %c-1640531527_i32_22 = arith.constant -1640531527 : i32
          %c-845247145_i32 = arith.constant -845247145 : i32
          %c-766435501_i32 = arith.constant -766435501 : i32
          %c16384_i32 = arith.constant 16384 : i32
          %386 = ascsimt.thread_idx {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
          %387 = arith.extsi %386 {cannir.vf_threads = 256 : i64} : i32 to i64
          scf.for %arg17 = %387 to %arg16 step %c256_i64_15  : i64 {
            %388 = arith.trunci %arg17 : i64 to i32
            %389 = arith.addi %arg7, %388 : i32
            %390 = arith.remui %389, %c16384_i32 : i32
            %391 = arith.xori %390, %arg8 : i32
            %392 = arith.muli %391, %c-766435501_i32 : i32
            %393 = "ascsimt.bitcast"(%391) : (i32) -> ui32
            %394 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
            %395 = ascsimt.math_binary %393, %394, "mulhi" : ui32, ui32 -> ui32
            %396 = "ascsimt.bitcast"(%395) : (ui32) -> i32
            %397 = arith.xori %396, %arg9 : i32
            %398 = arith.muli %397, %c-845247145_i32 : i32
            %399 = "ascsimt.bitcast"(%397) : (i32) -> ui32
            %400 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
            %401 = ascsimt.math_binary %399, %400, "mulhi" : ui32, ui32 -> ui32
            %402 = "ascsimt.bitcast"(%401) : (ui32) -> i32
            %403 = arith.xori %402, %arg10 : i32
            %404 = arith.xori %392, %arg11 : i32
            %405:4 = scf.for %arg18 = %c3_i32 to %c10_i32 step %c1_i32_20 iter_args(%arg19 = %403, %arg20 = %398, %arg21 = %404, %arg22 = %arg14) -> (i32, i32, i32, i32)  : i32 {
              %414 = arith.muli %arg19, %c-766435501_i32 : i32
              %415 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
              %416 = ascsimt.math_binary %415, %394, "mulhi" : ui32, ui32 -> ui32
              %417 = "ascsimt.bitcast"(%416) : (ui32) -> i32
              %418 = arith.muli %arg21, %c-845247145_i32 : i32
              %419 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
              %420 = ascsimt.math_binary %419, %400, "mulhi" : ui32, ui32 -> ui32
              %421 = "ascsimt.bitcast"(%420) : (ui32) -> i32
              %422 = arith.muli %arg18, %c-1640531527_i32_22 : i32
              %423 = arith.addi %arg12, %422 : i32
              %424 = arith.muli %arg18, %c-1150833019_i32_21 : i32
              %425 = arith.addi %arg13, %424 : i32
              %426 = arith.xori %421, %arg20 : i32
              %427 = arith.xori %426, %423 : i32
              %428 = arith.xori %417, %arg22 : i32
              %429 = arith.xori %428, %425 : i32
              scf.yield %427, %418, %429, %414 : i32, i32, i32, i32
            }
            %406 = arith.muli %arg17, %c4_i64_19 : i64
            %407 = arith.index_cast %406 : i64 to index
            ascsimt.store %arg15[%407], %405#0 : memref<3328xi32, 1>, i32
            %408 = arith.addi %406, %c1_i64_18 : i64
            %409 = arith.index_cast %408 : i64 to index
            ascsimt.store %arg15[%409], %405#1 : memref<3328xi32, 1>, i32
            %410 = arith.addi %406, %c2_i64_17 : i64
            %411 = arith.index_cast %410 : i64 to index
            ascsimt.store %arg15[%411], %405#2 : memref<3328xi32, 1>, i32
            %412 = arith.addi %406, %c3_i64_16 : i64
            %413 = arith.index_cast %412 : i64 to index
            ascsimt.store %arg15[%413], %405#3 : memref<3328xi32, 1>, i32
          } {cannir.vf_threads = 256 : i64}
        } {threads = 256 : i64}
      } else {
        ascsimt.vf(%145, %155, %170, %171, %133, %143, %alloc_10, %172 : i32, i32, i32, i32, i32, i32, memref<3328xi32, 1>, i64) {
        ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: memref<3328xi32, 1>, %arg14: i64):
          %c256_i64_15 = arith.constant 256 : i64
          %c3_i64_16 = arith.constant 3 : i64
          %c2_i64_17 = arith.constant 2 : i64
          %c1_i64_18 = arith.constant 1 : i64
          %c4_i64_19 = arith.constant 4 : i64
          %c1_i32_20 = arith.constant 1 : i32
          %c10_i32 = arith.constant 10 : i32
          %c0_i32_21 = arith.constant 0 : i32
          %c-1150833019_i32_22 = arith.constant -1150833019 : i32
          %c-1640531527_i32_23 = arith.constant -1640531527 : i32
          %c-845247145_i32 = arith.constant -845247145 : i32
          %c-766435501_i32 = arith.constant -766435501 : i32
          %c262144_i32 = arith.constant 262144 : i32
          %c16384_i32 = arith.constant 16384 : i32
          %321 = ascsimt.thread_idx {ascsimt.scope = 2 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
          %322 = arith.extsi %321 {cannir.vf_threads = 256 : i64} : i32 to i64
          scf.for %arg15 = %322 to %arg14 step %c256_i64_15  : i64 {
            %323 = arith.trunci %arg15 : i64 to i32
            %324 = arith.addi %arg7, %323 : i32
            %325 = arith.cmpi ult, %324, %arg7 : i32
            %326 = arith.extui %325 : i1 to i32
            %327 = arith.addi %arg8, %326 : i32
            %328 = arith.divui %324, %c16384_i32 : i32
            %329 = arith.remui %327, %c16384_i32 : i32
            %330 = arith.muli %329, %c262144_i32 : i32
            %331 = arith.addi %328, %330 : i32
            %332 = arith.divui %327, %c16384_i32 : i32
            %333 = arith.addi %arg9, %331 : i32
            %334 = arith.addi %arg10, %332 : i32
            %335 = arith.cmpi ult, %333, %arg9 : i32
            %336 = arith.extui %335 : i1 to i32
            %337 = arith.addi %334, %336 : i32
            %338 = arith.remui %324, %c16384_i32 : i32
            %339:4 = scf.for %arg16 = %c0_i32_21 to %c10_i32 step %c1_i32_20 iter_args(%arg17 = %c0_i32_21, %arg18 = %338, %arg19 = %333, %arg20 = %337) -> (i32, i32, i32, i32)  : i32 {
              %348 = arith.muli %arg17, %c-766435501_i32 : i32
              %349 = "ascsimt.bitcast"(%arg17) : (i32) -> ui32
              %350 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %351 = ascsimt.math_binary %349, %350, "mulhi" : ui32, ui32 -> ui32
              %352 = "ascsimt.bitcast"(%351) : (ui32) -> i32
              %353 = arith.muli %arg19, %c-845247145_i32 : i32
              %354 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
              %355 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %356 = ascsimt.math_binary %354, %355, "mulhi" : ui32, ui32 -> ui32
              %357 = "ascsimt.bitcast"(%356) : (ui32) -> i32
              %358 = arith.muli %arg16, %c-1640531527_i32_23 : i32
              %359 = arith.addi %arg11, %358 : i32
              %360 = arith.muli %arg16, %c-1150833019_i32_22 : i32
              %361 = arith.addi %arg12, %360 : i32
              %362 = arith.xori %357, %arg18 : i32
              %363 = arith.xori %362, %359 : i32
              %364 = arith.xori %352, %arg20 : i32
              %365 = arith.xori %364, %361 : i32
              scf.yield %363, %353, %365, %348 : i32, i32, i32, i32
            }
            %340 = arith.muli %arg15, %c4_i64_19 : i64
            %341 = arith.index_cast %340 : i64 to index
            ascsimt.store %arg13[%341], %339#0 : memref<3328xi32, 1>, i32
            %342 = arith.addi %340, %c1_i64_18 : i64
            %343 = arith.index_cast %342 : i64 to index
            ascsimt.store %arg13[%343], %339#1 : memref<3328xi32, 1>, i32
            %344 = arith.addi %340, %c2_i64_17 : i64
            %345 = arith.index_cast %344 : i64 to index
            ascsimt.store %arg13[%345], %339#2 : memref<3328xi32, 1>, i32
            %346 = arith.addi %340, %c3_i64_16 : i64
            %347 = arith.index_cast %346 : i64 to index
            ascsimt.store %arg13[%347], %339#3 : memref<3328xi32, 1>, i32
          } {cannir.vf_threads = 256 : i64}
        } {threads = 256 : i64}
      }
      %178 = arith.trunci %124 : i64 to i32
      ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
      %179 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<13312xf32, 1> -> memref<?xui16, 1>
      ascvec.vf(%alloc_8, %alloc_4, %179, %alloc_10, %alloc_6, %178 : memref<832xui8, 1>, memref<13312xf32, 1>, memref<?xui16, 1>, memref<3328xi32, 1>, memref<13312xui8, 1>, i32) {
      ^bb0(%arg7: memref<832xui8, 1>, %arg8: memref<13312xf32, 1>, %arg9: memref<?xui16, 1>, %arg10: memref<3328xi32, 1>, %arg11: memref<13312xui8, 1>, %arg12: i32):
        %c512_i32 = arith.constant 512 : i32
        %c1_i32_15 = arith.constant 1 : i32
        %c2139062143_i32 = arith.constant 2139062143 : i32
        %c-8388608_i32 = arith.constant -8388608 : i32
        %c126_i8 = arith.constant 126 : i8
        %c-1_i8 = arith.constant -1 : i8
        %c56_i8 = arith.constant 56 : i8
        %cst = arith.constant 0.0156249991 : f32
        %c64_i32_16 = arith.constant 64 : i32
        %c254_i32 = arith.constant 254 : i32
        %c-8_i32 = arith.constant -8 : i32
        %c8_i32 = arith.constant 8 : i32
        %c255_i32 = arith.constant 255 : i32
        %c128_i32 = arith.constant 128 : i32
        %c256_i32 = arith.constant 256 : i32
        %c0_17 = arith.constant 0 : index
        %c1_i8 = arith.constant 1 : i8
        %c32_i8 = arith.constant 32 : i8
        %c65535_i32 = arith.constant 65535 : i32
        %c3_i8 = arith.constant 3 : i8
        %c0_i8 = arith.constant 0 : i8
        %c-128_i8 = arith.constant -128 : i8
        %c0_i32_18 = arith.constant 0 : i32
        %321 = ascvec.create_mask : !ascvec.mask
        %322 = ascvec.create_mask elem_bits 8 : !ascvec.mask
        %323 = ascvec.arange %c0_i32_18 : i32 -> <ui32>
        %324 = ascvec.shr %323, 3, %321 : <ui32> -> <ui32>
        %325 = ascvec.duplicate_scalar %c-128_i8, %322 : i8 -> <ui8>
        %326 = ascvec.arange %c0_i8 : i8 -> <ui8>
        %327 = ascvec.duplicate_scalar %c3_i8, %322 : i8 -> <ui8>
        %328 = ascvec.and %326, %327, %322 : <ui8>, <ui8> -> <ui8>
        %329 = ascvec.shl %328, 6, %322 : <ui8> -> <ui8>
        %330 = ascvec.shr %326, 2, %322 : <ui8> -> <ui8>
        %331 = ascvec.or %329, %330, %322 : <ui8>, <ui8> -> <ui8>
        %332 = ascvec.duplicate_scalar %c65535_i32, %321 : i32 -> <ui32>
        %333 = ascvec.cmp_lt_scalar %326, %c32_i8, %322 : <ui8>, i8 -> !ascvec.mask
        %334 = ascvec.cmp_lt_scalar %328, %c1_i8, %322 : <ui8>, i8 -> !ascvec.mask
        %335 = ascvec.mask_and %333, %334, %322
        ascvec.store_unalign_begin %arg7 : memref<832xui8, 1>
        scf.for %arg13 = %c0_i32_18 to %arg12 step %c1_i32_15  : i32 {
          %336 = arith.muli %arg13, %c256_i32 : i32
          %337 = arith.index_cast %336 : i32 to index
          %result0, %result1 = ascvec.loadalign_deintlv %arg8[%337] {dintlv_width = "b32"} : memref<13312xf32, 1>, index -> <f32>, <f32>
          %338 = arith.addi %336, %c128_i32 : i32
          %339 = arith.index_cast %338 : i32 to index
          %result0_21, %result1_22 = ascvec.loadalign_deintlv %arg8[%339] {dintlv_width = "b32"} : memref<13312xf32, 1>, index -> <f32>, <f32>
          %result0_23, %result1_24 = ascvec.deintlv %result0, %result0_21 : <f32>, <f32> -> <f32>, <f32>
          %result0_25, %result1_26 = ascvec.deintlv %result1, %result1_22 : <f32>, <f32> -> <f32>, <f32>
          %340 = ascvec.abs %result0_23, %321 : <f32>
          %341 = ascvec.reinterpret %340 : <f32> -> <ui32>
          %342 = ascvec.abs %result0_25, %321 : <f32>
          %343 = ascvec.reinterpret %342 : <f32> -> <ui32>
          %344 = ascvec.abs %result1_24, %321 : <f32>
          %345 = ascvec.reinterpret %344 : <f32> -> <ui32>
          %346 = ascvec.abs %result1_26, %321 : <f32>
          %347 = ascvec.reinterpret %346 : <f32> -> <ui32>
          %348 = ascvec.max %341, %343, %321 : <ui32>
          %349 = ascvec.max %345, %347, %321 : <ui32>
          %350 = ascvec.max %348, %349, %321 : <ui32>
          %351 = ascvec.reduce_max_datablock %350, %321 : <ui32>
          %352 = arith.muli %arg13, %c512_i32 : i32
          %353 = arith.index_cast %352 : i32 to index
          %result0_27, %result1_28 = ascvec.loadalign_deintlv %arg9[%353] {dintlv_width = "b16"} : memref<?xui16, 1>, index -> <ui16>, <ui16>
          %354 = arith.addi %352, %c256_i32 : i32
          %355 = arith.index_cast %354 : i32 to index
          %result0_29, %result1_30 = ascvec.loadalign_deintlv %arg9[%355] {dintlv_width = "b16"} : memref<?xui16, 1>, index -> <ui16>, <ui16>
          %356 = ascvec.reinterpret %result1_28 : <ui16> -> <ui8>
          %357 = ascvec.reinterpret %result1_30 : <ui16> -> <ui8>
          %result0_31, %result1_32 = ascvec.deintlv %356, %357 : <ui8>, <ui8> -> <ui8>, <ui8>
          %358 = ascvec.and %result1_32, %325, %322 : <ui8>, <ui8> -> <ui8>
          %359 = ascvec.reinterpret %358 : <ui8> -> <ui32>
          %360 = ascvec.shr %351, 23, %321 : <ui32> -> <ui32>
          %361 = ascvec.cmp_ge_scalar %360, %c255_i32, %321 : <ui32>, i32 -> !ascvec.mask
          %362 = ascvec.max_scalar %360, %c8_i32, %321 : <ui32>, i32
          %363 = ascvec.add_scalar %362, %c-8_i32, %321 : <ui32>, i32
          %364 = ascvec.duplicate_scalar %c255_i32, %361 merge %363 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %365 = ascvec.duplicate_scalar %c254_i32, %321 : i32 -> <ui32>
          %366 = ascvec.sub %365, %364, %321 : <ui32>
          %367 = ascvec.cmp_ne_scalar %364, %c0_i32_18, %321 : <ui32>, i32 -> !ascvec.mask
          %368 = ascvec.shl %366, 23, %367 : <ui32> -> <ui32>
          %369 = ascvec.reinterpret %368 : <ui32> -> <f32>
          %370 = ascvec.gather_reg %369, %324 : <f32>, <ui32> -> <f32>
          %371 = ascvec.reinterpret %341 : <ui32> -> <f32>
          %372 = ascvec.mul %371, %370, %321 : <f32>
          %373 = ascvec.reinterpret %343 : <ui32> -> <f32>
          %374 = ascvec.mul %373, %370, %321 : <f32>
          %375 = ascvec.reinterpret %345 : <ui32> -> <f32>
          %376 = ascvec.mul %375, %370, %321 : <f32>
          %377 = ascvec.reinterpret %347 : <ui32> -> <f32>
          %378 = ascvec.mul %377, %370, %321 : <f32>
          %379 = arith.muli %arg13, %c64_i32_16 : i32
          %380 = arith.index_cast %379 : i32 to index
          %381 = ascvec.loadalign %arg10[%380] : memref<3328xi32, 1>, index -> <i32>
          %382 = ascvec.reinterpret %381 : <i32> -> <ui32>
          %383 = ascvec.shl %382, 8, %321 : <ui32> -> <ui32>
          %384 = ascvec.shr %382, 24, %321 : <ui32> -> <ui32>
          %385 = ascvec.or %383, %384, %321 : <ui32>, <ui32> -> <ui32>
          %386 = ascvec.shr %382, 16, %321 : <ui32> -> <ui32>
          %387 = ascvec.and %385, %332, %321 : <ui32>, <ui32> -> <ui32>
          %388 = ascvec.and %382, %332, %321 : <ui32>, <ui32> -> <ui32>
          %389 = ascvec.shr %385, 16, %321 : <ui32> -> <ui32>
          %390 = ascvec.reinterpret %372 : <f32> -> <ui32>
          %391 = ascvec.reinterpret %390 : <ui32> -> <f32>
          %392 = ascvec.max_scalar %391, %cst, %321 : <f32>, f32
          %393 = ascvec.add %391, %392, %321 : <f32>
          %394 = ascvec.reinterpret %393 : <f32> -> <ui32>
          %395 = ascvec.shr %394, 4, %321 : <ui32> -> <ui32>
          %396 = ascvec.add %395, %386, %321 : <ui32>
          %397 = ascvec.reinterpret %374 : <f32> -> <ui32>
          %398 = ascvec.reinterpret %397 : <ui32> -> <f32>
          %399 = ascvec.max_scalar %398, %cst, %321 : <f32>, f32
          %400 = ascvec.add %398, %399, %321 : <f32>
          %401 = ascvec.reinterpret %400 : <f32> -> <ui32>
          %402 = ascvec.shr %401, 4, %321 : <ui32> -> <ui32>
          %403 = ascvec.add %402, %387, %321 : <ui32>
          %404 = ascvec.reinterpret %376 : <f32> -> <ui32>
          %405 = ascvec.reinterpret %404 : <ui32> -> <f32>
          %406 = ascvec.max_scalar %405, %cst, %321 : <f32>, f32
          %407 = ascvec.add %405, %406, %321 : <f32>
          %408 = ascvec.reinterpret %407 : <f32> -> <ui32>
          %409 = ascvec.shr %408, 4, %321 : <ui32> -> <ui32>
          %410 = ascvec.add %409, %388, %321 : <ui32>
          %411 = ascvec.reinterpret %378 : <f32> -> <ui32>
          %412 = ascvec.reinterpret %411 : <ui32> -> <f32>
          %413 = ascvec.max_scalar %412, %cst, %321 : <f32>, f32
          %414 = ascvec.add %412, %413, %321 : <f32>
          %415 = ascvec.reinterpret %414 : <f32> -> <ui32>
          %416 = ascvec.shr %415, 4, %321 : <ui32> -> <ui32>
          %417 = ascvec.add %416, %389, %321 : <ui32>
          %418 = ascvec.reinterpret %396 : <ui32> -> <ui16>
          %419 = ascvec.reinterpret %403 : <ui32> -> <ui16>
          %result0_33, %result1_34 = ascvec.deintlv %418, %419 : <ui16>, <ui16> -> <ui16>, <ui16>
          %420 = ascvec.reinterpret %410 : <ui32> -> <ui16>
          %421 = ascvec.reinterpret %417 : <ui32> -> <ui16>
          %result0_35, %result1_36 = ascvec.deintlv %420, %421 : <ui16>, <ui16> -> <ui16>, <ui16>
          %422 = ascvec.reinterpret %result1_34 : <ui16> -> <ui8>
          %423 = ascvec.reinterpret %result1_36 : <ui16> -> <ui8>
          %result0_37, %result1_38 = ascvec.deintlv %422, %423 : <ui8>, <ui8> -> <ui8>, <ui8>
          %424 = ascvec.gather_reg %result0_37, %331 : <ui8>, <ui8> -> <ui8>
          %425 = ascvec.reinterpret %424 : <ui8> -> <ui32>
          %426 = ascvec.reinterpret %425 : <ui32> -> <ui8>
          %427 = ascvec.add_scalar %426, %c56_i8, %322 : <ui8>, i8
          %428 = ascvec.cmp_ne_scalar %427, %c-1_i8, %322 : <ui8>, i8 -> !ascvec.mask
          %429 = ascvec.min_scalar %427, %c126_i8, %428 : <ui8>, i8
          %430 = ascvec.reinterpret %429 : <ui8> -> <ui32>
          %431 = ascvec.or %430, %359, %321 : <ui32>, <ui32> -> <ui32>
          %432 = ascvec.reinterpret %370 : <f32> -> <ui32>
          %433 = ascvec.cmp_eq_scalar %432, %c-8388608_i32, %321 : <ui32>, i32 -> !ascvec.mask
          %434 = ascvec.duplicate_scalar %c2139062143_i32, %433 merge %431 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %435 = ascvec.reinterpret %434 : <ui32> -> <ui8>
          ascvec.storealign %arg11[%337], %435, %322 : memref<13312xui8, 1>, index, <ui8>
          %436 = ascvec.reinterpret %364 : <ui32> -> <ui8>
          %437 = ascvec.squeeze_store_reg %436, %335 : <ui8>, !ascvec.mask -> <ui8>
          ascvec.store_unalign %arg7[%c0_17], %437 : memref<832xui8, 1>, <ui8>
        }
        %c0_19 = arith.constant 0 : index
        %c1_20 = arith.constant 1 : index
        scf.for %arg13 = %c0_19 to %c1_20 step %c1_20 {
          ascvec.store_unalign_post %arg7[%c0_17] : memref<832xui8, 1>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
      ascvec.copy_ub2ub %alloc_11[%c0], %alloc_6[%c0], %c13312_i32 : memref<13312xui8, 1>, memref<13312xui8, 1>
      ascvec.vf(%alloc_11, %alloc_6 : memref<13312xui8, 1>, memref<13312xui8, 1>) {
      ^bb0(%arg7: memref<13312xui8, 1>, %arg8: memref<13312xui8, 1>):
      } {cannir.vf_mode = "simd"}
      ascvec.copy_ub2ub %alloc_12[%c0], %alloc_8[%c0], %c832_i32 : memref<832xui8, 1>, memref<832xui8, 1>
      ascvec.vf(%alloc_12, %alloc_8 : memref<832xui8, 1>, memref<832xui8, 1>) {
      ^bb0(%arg7: memref<832xui8, 1>, %arg8: memref<832xui8, 1>):
      } {cannir.vf_mode = "simd"}
      %180 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
      %181 = cannir.extract_shape %180 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, -1)>
      %182 = cannir.extract_dim %181[0] : !cannir.shape<(-1, -1)>
      %183 = arith.minsi %78, %182 : i64
      %184 = arith.subi %182, %183 : i64
      %185 = arith.minsi %184, %c1_i64 : i64
      %186 = arith.cmpi sle, %78, %182 : i64
      %187 = arith.select %186, %185, %c0_i64 : i64
      %188 = cannir.extract_dim %181[1] : !cannir.shape<(-1, -1)>
      %189 = arith.divsi %188, %90 : i64
      %190 = arith.minsi %80, %189 : i64
      %191 = arith.muli %190, %56 : i64
      %192 = arith.subi %188, %191 : i64
      %193 = arith.minsi %56, %192 : i64
      %194 = arith.cmpi sle, %80, %189 : i64
      %195 = arith.select %194, %193, %c0_i64 : i64
      %196 = cannir.extract_stride %180 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
      %197 = cannir.extract_dim %196[0] : !cannir.stride<(-1, 2)>
      %198 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, f32>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
      %199 = cannir.extract_stride %198 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
      %200 = cannir.extract_dim %199[0] : !cannir.stride<(-1, 2)>
      %201 = arith.muli %78, %200 : i64
      %202 = arith.addi %201, %104 : i64
      %203 = arith.muli %202, %c32_i64 : i64
      %204 = ascscalar.memref_address %alloc_0 : memref<1xf32>
      %205 = arith.divui %203, %c8_i64 : i64
      %206 = arith.addi %204, %205 : i64
      %207 = ascscalar.bind_gm_ptr %206 : memref<1xf32>
      %208 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<13312xf32, 1> -> memref<?xf32, 1>
      %209 = arith.cmpi ugt, %187, %c0_i64 : i64
      %210 = arith.cmpi ugt, %195, %c0_i64 : i64
      %211 = arith.andi %209, %210 : i1
      scf.if %211 {
        ascvec.get_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
        %321 = arith.trunci %195 : i64 to i32
        %322 = arith.trunci %187 : i64 to i32
        ascvec.copy_gm2ub_nddma %208[%c0], %207[%c0], %321, %322, %c1_i32, %c1_i32, %c1_i32, %c2_i64, %197, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c13312_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 : memref<?xf32, 1>, memref<1xf32>
        ascvec.release_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
      }
      %212 = arith.addi %115, %c1_i64 : i64
      %213 = arith.muli %212, %6 : i64
      %214 = arith.addi %213, %81 : i64
      %215 = arith.divui %214, %c16_i64 : i64
      %216 = arith.trunci %215 : i64 to i32
      %217 = arith.extui %216 : i32 to i64
      %218 = arith.subi %215, %217 : i64
      %219 = arith.remsi %218, %c4294967296_i64 : i64
      %220 = arith.cmpi ne, %219, %c0_i64 : i64
      %221 = arith.cmpi slt, %219, %c0_i64 : i64
      %222 = arith.andi %220, %221 : i1
      %223 = arith.divsi %218, %c4294967296_i64 : i64
      %224 = arith.extui %222 : i1 to i64
      %225 = arith.subi %223, %224 : i64
      %226 = arith.trunci %225 : i64 to i32
      %227 = arith.divui %215, %c16384_i64 : i64
      %228 = arith.addi %215, %172 : i64
      %229 = arith.subi %228, %c1_i64 : i64
      %230 = arith.divui %229, %c16384_i64 : i64
      %231 = arith.cmpi eq, %227, %230 : i64
      scf.if %231 {
        %321 = arith.trunci %227 : i64 to i32
        %322 = arith.extui %321 : i32 to i64
        %323 = arith.subi %227, %322 : i64
        %324 = arith.remsi %323, %c4294967296_i64 : i64
        %325 = arith.cmpi ne, %324, %c0_i64 : i64
        %326 = arith.cmpi slt, %324, %c0_i64 : i64
        %327 = arith.andi %325, %326 : i1
        %328 = arith.divsi %323, %c4294967296_i64 : i64
        %329 = arith.extui %327 : i1 to i64
        %330 = arith.subi %328, %329 : i64
        %331 = arith.trunci %330 : i64 to i32
        %332 = arith.addi %170, %321 : i32
        %333 = arith.addi %171, %331 : i32
        %334 = arith.cmpi ult, %332, %170 : i32
        %335 = arith.extui %334 : i1 to i32
        %336 = arith.addi %333, %335 : i32
        %337 = arith.extui %332 : i32 to i64
        %338 = arith.muli %337, %c3449720151_i64 : i64
        %339 = arith.trunci %338 : i64 to i32
        %340 = arith.extui %339 : i32 to i64
        %341 = arith.subi %338, %340 : i64
        %342 = arith.remsi %341, %c4294967296_i64 : i64
        %343 = arith.cmpi ne, %342, %c0_i64 : i64
        %344 = arith.cmpi slt, %342, %c0_i64 : i64
        %345 = arith.andi %343, %344 : i1
        %346 = arith.divsi %341, %c4294967296_i64 : i64
        %347 = arith.extui %345 : i1 to i64
        %348 = arith.subi %346, %347 : i64
        %349 = arith.trunci %348 : i64 to i32
        %350 = arith.xori %336, %143 : i32
        %351 = arith.extui %350 : i32 to i64
        %352 = arith.muli %351, %c3449720151_i64 : i64
        %353 = arith.trunci %352 : i64 to i32
        %354 = arith.extui %353 : i32 to i64
        %355 = arith.subi %352, %354 : i64
        %356 = arith.remsi %355, %c4294967296_i64 : i64
        %357 = arith.cmpi ne, %356, %c0_i64 : i64
        %358 = arith.cmpi slt, %356, %c0_i64 : i64
        %359 = arith.andi %357, %358 : i1
        %360 = arith.divsi %355, %c4294967296_i64 : i64
        %361 = arith.extui %359 : i1 to i64
        %362 = arith.subi %360, %361 : i64
        %363 = arith.trunci %362 : i64 to i32
        %364 = arith.addi %133, %c-1640531527_i32 : i32
        %365 = arith.addi %143, %c-1150833019_i32 : i32
        %366 = arith.xori %363, %339 : i32
        %367 = arith.xori %366, %364 : i32
        %368 = arith.extui %367 : i32 to i64
        %369 = arith.muli %368, %c3528531795_i64 : i64
        %370 = arith.trunci %369 : i64 to i32
        %371 = arith.extui %370 : i32 to i64
        %372 = arith.subi %369, %371 : i64
        %373 = arith.remsi %372, %c4294967296_i64 : i64
        %374 = arith.cmpi ne, %373, %c0_i64 : i64
        %375 = arith.cmpi slt, %373, %c0_i64 : i64
        %376 = arith.andi %374, %375 : i1
        %377 = arith.divsi %372, %c4294967296_i64 : i64
        %378 = arith.extui %376 : i1 to i64
        %379 = arith.subi %377, %378 : i64
        %380 = arith.trunci %379 : i64 to i32
        %381 = arith.xori %349, %133 : i32
        %382 = arith.addi %133, %c1013904242_i32 : i32
        %383 = arith.xori %353, %382 : i32
        %384 = arith.addi %143, %c1993301258_i32 : i32
        %385 = arith.xori %380, %384 : i32
        ascsimt.vf(%216, %381, %365, %383, %385, %133, %143, %370, %alloc_10, %172 : i32, i32, i32, i32, i32, i32, i32, i32, memref<3328xi32, 1>, i64) {
        ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<3328xi32, 1>, %arg16: i64):
          %c256_i64_15 = arith.constant 256 : i64
          %c3_i64_16 = arith.constant 3 : i64
          %c2_i64_17 = arith.constant 2 : i64
          %c1_i64_18 = arith.constant 1 : i64
          %c4_i64_19 = arith.constant 4 : i64
          %c1_i32_20 = arith.constant 1 : i32
          %c10_i32 = arith.constant 10 : i32
          %c3_i32 = arith.constant 3 : i32
          %c-1150833019_i32_21 = arith.constant -1150833019 : i32
          %c-1640531527_i32_22 = arith.constant -1640531527 : i32
          %c-845247145_i32 = arith.constant -845247145 : i32
          %c-766435501_i32 = arith.constant -766435501 : i32
          %c16384_i32 = arith.constant 16384 : i32
          %386 = ascsimt.thread_idx {ascsimt.scope = 4 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
          %387 = arith.extsi %386 {cannir.vf_threads = 256 : i64} : i32 to i64
          scf.for %arg17 = %387 to %arg16 step %c256_i64_15  : i64 {
            %388 = arith.trunci %arg17 : i64 to i32
            %389 = arith.addi %arg7, %388 : i32
            %390 = arith.remui %389, %c16384_i32 : i32
            %391 = arith.xori %390, %arg8 : i32
            %392 = arith.muli %391, %c-766435501_i32 : i32
            %393 = "ascsimt.bitcast"(%391) : (i32) -> ui32
            %394 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
            %395 = ascsimt.math_binary %393, %394, "mulhi" : ui32, ui32 -> ui32
            %396 = "ascsimt.bitcast"(%395) : (ui32) -> i32
            %397 = arith.xori %396, %arg9 : i32
            %398 = arith.muli %397, %c-845247145_i32 : i32
            %399 = "ascsimt.bitcast"(%397) : (i32) -> ui32
            %400 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
            %401 = ascsimt.math_binary %399, %400, "mulhi" : ui32, ui32 -> ui32
            %402 = "ascsimt.bitcast"(%401) : (ui32) -> i32
            %403 = arith.xori %402, %arg10 : i32
            %404 = arith.xori %392, %arg11 : i32
            %405:4 = scf.for %arg18 = %c3_i32 to %c10_i32 step %c1_i32_20 iter_args(%arg19 = %403, %arg20 = %398, %arg21 = %404, %arg22 = %arg14) -> (i32, i32, i32, i32)  : i32 {
              %414 = arith.muli %arg19, %c-766435501_i32 : i32
              %415 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
              %416 = ascsimt.math_binary %415, %394, "mulhi" : ui32, ui32 -> ui32
              %417 = "ascsimt.bitcast"(%416) : (ui32) -> i32
              %418 = arith.muli %arg21, %c-845247145_i32 : i32
              %419 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
              %420 = ascsimt.math_binary %419, %400, "mulhi" : ui32, ui32 -> ui32
              %421 = "ascsimt.bitcast"(%420) : (ui32) -> i32
              %422 = arith.muli %arg18, %c-1640531527_i32_22 : i32
              %423 = arith.addi %arg12, %422 : i32
              %424 = arith.muli %arg18, %c-1150833019_i32_21 : i32
              %425 = arith.addi %arg13, %424 : i32
              %426 = arith.xori %421, %arg20 : i32
              %427 = arith.xori %426, %423 : i32
              %428 = arith.xori %417, %arg22 : i32
              %429 = arith.xori %428, %425 : i32
              scf.yield %427, %418, %429, %414 : i32, i32, i32, i32
            }
            %406 = arith.muli %arg17, %c4_i64_19 : i64
            %407 = arith.index_cast %406 : i64 to index
            ascsimt.store %arg15[%407], %405#0 : memref<3328xi32, 1>, i32
            %408 = arith.addi %406, %c1_i64_18 : i64
            %409 = arith.index_cast %408 : i64 to index
            ascsimt.store %arg15[%409], %405#1 : memref<3328xi32, 1>, i32
            %410 = arith.addi %406, %c2_i64_17 : i64
            %411 = arith.index_cast %410 : i64 to index
            ascsimt.store %arg15[%411], %405#2 : memref<3328xi32, 1>, i32
            %412 = arith.addi %406, %c3_i64_16 : i64
            %413 = arith.index_cast %412 : i64 to index
            ascsimt.store %arg15[%413], %405#3 : memref<3328xi32, 1>, i32
          } {cannir.vf_threads = 256 : i64}
        } {threads = 256 : i64}
      } else {
        ascsimt.vf(%216, %226, %170, %171, %133, %143, %alloc_10, %172 : i32, i32, i32, i32, i32, i32, memref<3328xi32, 1>, i64) {
        ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: memref<3328xi32, 1>, %arg14: i64):
          %c256_i64_15 = arith.constant 256 : i64
          %c3_i64_16 = arith.constant 3 : i64
          %c2_i64_17 = arith.constant 2 : i64
          %c1_i64_18 = arith.constant 1 : i64
          %c4_i64_19 = arith.constant 4 : i64
          %c1_i32_20 = arith.constant 1 : i32
          %c10_i32 = arith.constant 10 : i32
          %c0_i32_21 = arith.constant 0 : i32
          %c-1150833019_i32_22 = arith.constant -1150833019 : i32
          %c-1640531527_i32_23 = arith.constant -1640531527 : i32
          %c-845247145_i32 = arith.constant -845247145 : i32
          %c-766435501_i32 = arith.constant -766435501 : i32
          %c262144_i32 = arith.constant 262144 : i32
          %c16384_i32 = arith.constant 16384 : i32
          %321 = ascsimt.thread_idx {ascsimt.scope = 5 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
          %322 = arith.extsi %321 {cannir.vf_threads = 256 : i64} : i32 to i64
          scf.for %arg15 = %322 to %arg14 step %c256_i64_15  : i64 {
            %323 = arith.trunci %arg15 : i64 to i32
            %324 = arith.addi %arg7, %323 : i32
            %325 = arith.cmpi ult, %324, %arg7 : i32
            %326 = arith.extui %325 : i1 to i32
            %327 = arith.addi %arg8, %326 : i32
            %328 = arith.divui %324, %c16384_i32 : i32
            %329 = arith.remui %327, %c16384_i32 : i32
            %330 = arith.muli %329, %c262144_i32 : i32
            %331 = arith.addi %328, %330 : i32
            %332 = arith.divui %327, %c16384_i32 : i32
            %333 = arith.addi %arg9, %331 : i32
            %334 = arith.addi %arg10, %332 : i32
            %335 = arith.cmpi ult, %333, %arg9 : i32
            %336 = arith.extui %335 : i1 to i32
            %337 = arith.addi %334, %336 : i32
            %338 = arith.remui %324, %c16384_i32 : i32
            %339:4 = scf.for %arg16 = %c0_i32_21 to %c10_i32 step %c1_i32_20 iter_args(%arg17 = %c0_i32_21, %arg18 = %338, %arg19 = %333, %arg20 = %337) -> (i32, i32, i32, i32)  : i32 {
              %348 = arith.muli %arg17, %c-766435501_i32 : i32
              %349 = "ascsimt.bitcast"(%arg17) : (i32) -> ui32
              %350 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %351 = ascsimt.math_binary %349, %350, "mulhi" : ui32, ui32 -> ui32
              %352 = "ascsimt.bitcast"(%351) : (ui32) -> i32
              %353 = arith.muli %arg19, %c-845247145_i32 : i32
              %354 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
              %355 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %356 = ascsimt.math_binary %354, %355, "mulhi" : ui32, ui32 -> ui32
              %357 = "ascsimt.bitcast"(%356) : (ui32) -> i32
              %358 = arith.muli %arg16, %c-1640531527_i32_23 : i32
              %359 = arith.addi %arg11, %358 : i32
              %360 = arith.muli %arg16, %c-1150833019_i32_22 : i32
              %361 = arith.addi %arg12, %360 : i32
              %362 = arith.xori %357, %arg18 : i32
              %363 = arith.xori %362, %359 : i32
              %364 = arith.xori %352, %arg20 : i32
              %365 = arith.xori %364, %361 : i32
              scf.yield %363, %353, %365, %348 : i32, i32, i32, i32
            }
            %340 = arith.muli %arg15, %c4_i64_19 : i64
            %341 = arith.index_cast %340 : i64 to index
            ascsimt.store %arg13[%341], %339#0 : memref<3328xi32, 1>, i32
            %342 = arith.addi %340, %c1_i64_18 : i64
            %343 = arith.index_cast %342 : i64 to index
            ascsimt.store %arg13[%343], %339#1 : memref<3328xi32, 1>, i32
            %344 = arith.addi %340, %c2_i64_17 : i64
            %345 = arith.index_cast %344 : i64 to index
            ascsimt.store %arg13[%345], %339#2 : memref<3328xi32, 1>, i32
            %346 = arith.addi %340, %c3_i64_16 : i64
            %347 = arith.index_cast %346 : i64 to index
            ascsimt.store %arg13[%347], %339#3 : memref<3328xi32, 1>, i32
          } {cannir.vf_threads = 256 : i64}
        } {threads = 256 : i64}
      }
      ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
      %232 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<13312xf32, 1> -> memref<?xui16, 1>
      ascvec.vf(%alloc_9, %alloc_5, %232, %alloc_10, %alloc_7, %178 : memref<832xui8, 1>, memref<13312xf32, 1>, memref<?xui16, 1>, memref<3328xi32, 1>, memref<13312xui8, 1>, i32) {
      ^bb0(%arg7: memref<832xui8, 1>, %arg8: memref<13312xf32, 1>, %arg9: memref<?xui16, 1>, %arg10: memref<3328xi32, 1>, %arg11: memref<13312xui8, 1>, %arg12: i32):
        %c512_i32 = arith.constant 512 : i32
        %c1_i32_15 = arith.constant 1 : i32
        %c2139062143_i32 = arith.constant 2139062143 : i32
        %c-8388608_i32 = arith.constant -8388608 : i32
        %c126_i8 = arith.constant 126 : i8
        %c-1_i8 = arith.constant -1 : i8
        %c56_i8 = arith.constant 56 : i8
        %cst = arith.constant 0.0156249991 : f32
        %c64_i32_16 = arith.constant 64 : i32
        %c254_i32 = arith.constant 254 : i32
        %c-8_i32 = arith.constant -8 : i32
        %c8_i32 = arith.constant 8 : i32
        %c255_i32 = arith.constant 255 : i32
        %c128_i32 = arith.constant 128 : i32
        %c256_i32 = arith.constant 256 : i32
        %c0_17 = arith.constant 0 : index
        %c1_i8 = arith.constant 1 : i8
        %c32_i8 = arith.constant 32 : i8
        %c65535_i32 = arith.constant 65535 : i32
        %c3_i8 = arith.constant 3 : i8
        %c0_i8 = arith.constant 0 : i8
        %c-128_i8 = arith.constant -128 : i8
        %c0_i32_18 = arith.constant 0 : i32
        %321 = ascvec.create_mask : !ascvec.mask
        %322 = ascvec.create_mask elem_bits 8 : !ascvec.mask
        %323 = ascvec.arange %c0_i32_18 : i32 -> <ui32>
        %324 = ascvec.shr %323, 3, %321 : <ui32> -> <ui32>
        %325 = ascvec.duplicate_scalar %c-128_i8, %322 : i8 -> <ui8>
        %326 = ascvec.arange %c0_i8 : i8 -> <ui8>
        %327 = ascvec.duplicate_scalar %c3_i8, %322 : i8 -> <ui8>
        %328 = ascvec.and %326, %327, %322 : <ui8>, <ui8> -> <ui8>
        %329 = ascvec.shl %328, 6, %322 : <ui8> -> <ui8>
        %330 = ascvec.shr %326, 2, %322 : <ui8> -> <ui8>
        %331 = ascvec.or %329, %330, %322 : <ui8>, <ui8> -> <ui8>
        %332 = ascvec.duplicate_scalar %c65535_i32, %321 : i32 -> <ui32>
        %333 = ascvec.cmp_lt_scalar %326, %c32_i8, %322 : <ui8>, i8 -> !ascvec.mask
        %334 = ascvec.cmp_lt_scalar %328, %c1_i8, %322 : <ui8>, i8 -> !ascvec.mask
        %335 = ascvec.mask_and %333, %334, %322
        ascvec.store_unalign_begin %arg7 : memref<832xui8, 1>
        scf.for %arg13 = %c0_i32_18 to %arg12 step %c1_i32_15  : i32 {
          %336 = arith.muli %arg13, %c256_i32 : i32
          %337 = arith.index_cast %336 : i32 to index
          %result0, %result1 = ascvec.loadalign_deintlv %arg8[%337] {dintlv_width = "b32"} : memref<13312xf32, 1>, index -> <f32>, <f32>
          %338 = arith.addi %336, %c128_i32 : i32
          %339 = arith.index_cast %338 : i32 to index
          %result0_21, %result1_22 = ascvec.loadalign_deintlv %arg8[%339] {dintlv_width = "b32"} : memref<13312xf32, 1>, index -> <f32>, <f32>
          %result0_23, %result1_24 = ascvec.deintlv %result0, %result0_21 : <f32>, <f32> -> <f32>, <f32>
          %result0_25, %result1_26 = ascvec.deintlv %result1, %result1_22 : <f32>, <f32> -> <f32>, <f32>
          %340 = ascvec.abs %result0_23, %321 : <f32>
          %341 = ascvec.reinterpret %340 : <f32> -> <ui32>
          %342 = ascvec.abs %result0_25, %321 : <f32>
          %343 = ascvec.reinterpret %342 : <f32> -> <ui32>
          %344 = ascvec.abs %result1_24, %321 : <f32>
          %345 = ascvec.reinterpret %344 : <f32> -> <ui32>
          %346 = ascvec.abs %result1_26, %321 : <f32>
          %347 = ascvec.reinterpret %346 : <f32> -> <ui32>
          %348 = ascvec.max %341, %343, %321 : <ui32>
          %349 = ascvec.max %345, %347, %321 : <ui32>
          %350 = ascvec.max %348, %349, %321 : <ui32>
          %351 = ascvec.reduce_max_datablock %350, %321 : <ui32>
          %352 = arith.muli %arg13, %c512_i32 : i32
          %353 = arith.index_cast %352 : i32 to index
          %result0_27, %result1_28 = ascvec.loadalign_deintlv %arg9[%353] {dintlv_width = "b16"} : memref<?xui16, 1>, index -> <ui16>, <ui16>
          %354 = arith.addi %352, %c256_i32 : i32
          %355 = arith.index_cast %354 : i32 to index
          %result0_29, %result1_30 = ascvec.loadalign_deintlv %arg9[%355] {dintlv_width = "b16"} : memref<?xui16, 1>, index -> <ui16>, <ui16>
          %356 = ascvec.reinterpret %result1_28 : <ui16> -> <ui8>
          %357 = ascvec.reinterpret %result1_30 : <ui16> -> <ui8>
          %result0_31, %result1_32 = ascvec.deintlv %356, %357 : <ui8>, <ui8> -> <ui8>, <ui8>
          %358 = ascvec.and %result1_32, %325, %322 : <ui8>, <ui8> -> <ui8>
          %359 = ascvec.reinterpret %358 : <ui8> -> <ui32>
          %360 = ascvec.shr %351, 23, %321 : <ui32> -> <ui32>
          %361 = ascvec.cmp_ge_scalar %360, %c255_i32, %321 : <ui32>, i32 -> !ascvec.mask
          %362 = ascvec.max_scalar %360, %c8_i32, %321 : <ui32>, i32
          %363 = ascvec.add_scalar %362, %c-8_i32, %321 : <ui32>, i32
          %364 = ascvec.duplicate_scalar %c255_i32, %361 merge %363 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %365 = ascvec.duplicate_scalar %c254_i32, %321 : i32 -> <ui32>
          %366 = ascvec.sub %365, %364, %321 : <ui32>
          %367 = ascvec.cmp_ne_scalar %364, %c0_i32_18, %321 : <ui32>, i32 -> !ascvec.mask
          %368 = ascvec.shl %366, 23, %367 : <ui32> -> <ui32>
          %369 = ascvec.reinterpret %368 : <ui32> -> <f32>
          %370 = ascvec.gather_reg %369, %324 : <f32>, <ui32> -> <f32>
          %371 = ascvec.reinterpret %341 : <ui32> -> <f32>
          %372 = ascvec.mul %371, %370, %321 : <f32>
          %373 = ascvec.reinterpret %343 : <ui32> -> <f32>
          %374 = ascvec.mul %373, %370, %321 : <f32>
          %375 = ascvec.reinterpret %345 : <ui32> -> <f32>
          %376 = ascvec.mul %375, %370, %321 : <f32>
          %377 = ascvec.reinterpret %347 : <ui32> -> <f32>
          %378 = ascvec.mul %377, %370, %321 : <f32>
          %379 = arith.muli %arg13, %c64_i32_16 : i32
          %380 = arith.index_cast %379 : i32 to index
          %381 = ascvec.loadalign %arg10[%380] : memref<3328xi32, 1>, index -> <i32>
          %382 = ascvec.reinterpret %381 : <i32> -> <ui32>
          %383 = ascvec.shl %382, 8, %321 : <ui32> -> <ui32>
          %384 = ascvec.shr %382, 24, %321 : <ui32> -> <ui32>
          %385 = ascvec.or %383, %384, %321 : <ui32>, <ui32> -> <ui32>
          %386 = ascvec.shr %382, 16, %321 : <ui32> -> <ui32>
          %387 = ascvec.and %385, %332, %321 : <ui32>, <ui32> -> <ui32>
          %388 = ascvec.and %382, %332, %321 : <ui32>, <ui32> -> <ui32>
          %389 = ascvec.shr %385, 16, %321 : <ui32> -> <ui32>
          %390 = ascvec.reinterpret %372 : <f32> -> <ui32>
          %391 = ascvec.reinterpret %390 : <ui32> -> <f32>
          %392 = ascvec.max_scalar %391, %cst, %321 : <f32>, f32
          %393 = ascvec.add %391, %392, %321 : <f32>
          %394 = ascvec.reinterpret %393 : <f32> -> <ui32>
          %395 = ascvec.shr %394, 4, %321 : <ui32> -> <ui32>
          %396 = ascvec.add %395, %386, %321 : <ui32>
          %397 = ascvec.reinterpret %374 : <f32> -> <ui32>
          %398 = ascvec.reinterpret %397 : <ui32> -> <f32>
          %399 = ascvec.max_scalar %398, %cst, %321 : <f32>, f32
          %400 = ascvec.add %398, %399, %321 : <f32>
          %401 = ascvec.reinterpret %400 : <f32> -> <ui32>
          %402 = ascvec.shr %401, 4, %321 : <ui32> -> <ui32>
          %403 = ascvec.add %402, %387, %321 : <ui32>
          %404 = ascvec.reinterpret %376 : <f32> -> <ui32>
          %405 = ascvec.reinterpret %404 : <ui32> -> <f32>
          %406 = ascvec.max_scalar %405, %cst, %321 : <f32>, f32
          %407 = ascvec.add %405, %406, %321 : <f32>
          %408 = ascvec.reinterpret %407 : <f32> -> <ui32>
          %409 = ascvec.shr %408, 4, %321 : <ui32> -> <ui32>
          %410 = ascvec.add %409, %388, %321 : <ui32>
          %411 = ascvec.reinterpret %378 : <f32> -> <ui32>
          %412 = ascvec.reinterpret %411 : <ui32> -> <f32>
          %413 = ascvec.max_scalar %412, %cst, %321 : <f32>, f32
          %414 = ascvec.add %412, %413, %321 : <f32>
          %415 = ascvec.reinterpret %414 : <f32> -> <ui32>
          %416 = ascvec.shr %415, 4, %321 : <ui32> -> <ui32>
          %417 = ascvec.add %416, %389, %321 : <ui32>
          %418 = ascvec.reinterpret %396 : <ui32> -> <ui16>
          %419 = ascvec.reinterpret %403 : <ui32> -> <ui16>
          %result0_33, %result1_34 = ascvec.deintlv %418, %419 : <ui16>, <ui16> -> <ui16>, <ui16>
          %420 = ascvec.reinterpret %410 : <ui32> -> <ui16>
          %421 = ascvec.reinterpret %417 : <ui32> -> <ui16>
          %result0_35, %result1_36 = ascvec.deintlv %420, %421 : <ui16>, <ui16> -> <ui16>, <ui16>
          %422 = ascvec.reinterpret %result1_34 : <ui16> -> <ui8>
          %423 = ascvec.reinterpret %result1_36 : <ui16> -> <ui8>
          %result0_37, %result1_38 = ascvec.deintlv %422, %423 : <ui8>, <ui8> -> <ui8>, <ui8>
          %424 = ascvec.gather_reg %result0_37, %331 : <ui8>, <ui8> -> <ui8>
          %425 = ascvec.reinterpret %424 : <ui8> -> <ui32>
          %426 = ascvec.reinterpret %425 : <ui32> -> <ui8>
          %427 = ascvec.add_scalar %426, %c56_i8, %322 : <ui8>, i8
          %428 = ascvec.cmp_ne_scalar %427, %c-1_i8, %322 : <ui8>, i8 -> !ascvec.mask
          %429 = ascvec.min_scalar %427, %c126_i8, %428 : <ui8>, i8
          %430 = ascvec.reinterpret %429 : <ui8> -> <ui32>
          %431 = ascvec.or %430, %359, %321 : <ui32>, <ui32> -> <ui32>
          %432 = ascvec.reinterpret %370 : <f32> -> <ui32>
          %433 = ascvec.cmp_eq_scalar %432, %c-8388608_i32, %321 : <ui32>, i32 -> !ascvec.mask
          %434 = ascvec.duplicate_scalar %c2139062143_i32, %433 merge %431 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
          %435 = ascvec.reinterpret %434 : <ui32> -> <ui8>
          ascvec.storealign %arg11[%337], %435, %322 : memref<13312xui8, 1>, index, <ui8>
          %436 = ascvec.reinterpret %364 : <ui32> -> <ui8>
          %437 = ascvec.squeeze_store_reg %436, %335 : <ui8>, !ascvec.mask -> <ui8>
          ascvec.store_unalign %arg7[%c0_17], %437 : memref<832xui8, 1>, <ui8>
        }
        %c0_19 = arith.constant 0 : index
        %c1_20 = arith.constant 1 : index
        scf.for %arg13 = %c0_19 to %c1_20 step %c1_20 {
          ascvec.store_unalign_post %arg7[%c0_17] : memref<832xui8, 1>
        }
        ascvec.mem_bar vst_vld
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
      %233 = arith.trunci %84 : i64 to i32
      %234 = arith.remsi %233, %c64_i32 : i32
      %235 = arith.cmpi ne, %234, %c0_i32 : i32
      %236 = arith.cmpi slt, %234, %c0_i32 : i32
      %237 = arith.andi %235, %236 : i1
      %238 = arith.divsi %233, %c64_i32 : i32
      %239 = arith.extui %237 : i1 to i32
      %240 = arith.subi %238, %239 : i32
      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
      %241 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<832xui8, 1> -> memref<?xui16, 1>
      %242 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<832xui8, 1> -> memref<?xui16, 1>
      ascvec.vf(%alloc_11, %alloc_7, %alloc_13, %233, %241, %242, %alloc_14, %240 : memref<13312xui8, 1>, memref<13312xui8, 1>, memref<26624xui8, 1>, i32, memref<?xui16, 1>, memref<?xui16, 1>, memref<832xui16, 1>, i32) {
      ^bb0(%arg7: memref<13312xui8, 1>, %arg8: memref<13312xui8, 1>, %arg9: memref<26624xui8, 1>, %arg10: i32, %arg11: memref<?xui16, 1>, %arg12: memref<?xui16, 1>, %arg13: memref<832xui16, 1>, %arg14: i32):
        %c128_i32 = arith.constant 128 : i32
        %c256_i32 = arith.constant 256 : i32
        %c0_i32_15 = arith.constant 0 : i32
        %c2_i32 = arith.constant 2 : i32
        scf.for %arg15 = %c0_i32_15 to %arg10 step %c256_i32  : i32 {
          %321 = arith.index_cast %arg15 : i32 to index
          %322 = ascvec.loadalign %arg7[%321] : memref<13312xui8, 1>, index -> <ui8>
          %323 = ascvec.loadalign %arg8[%321] : memref<13312xui8, 1>, index -> <ui8>
          %324 = arith.muli %arg15, %c2_i32 : i32
          %325 = arith.index_cast %324 : i32 to index
          ascvec.storealign_intlv %arg9[%325], %322, %323 : memref<26624xui8, 1>, <ui8>, <ui8>
        }
        scf.for %arg15 = %c0_i32_15 to %arg14 step %c128_i32  : i32 {
          %321 = arith.index_cast %arg15 : i32 to index
          %322 = ascvec.loadalign %arg11[%321] : memref<?xui16, 1>, index -> <ui16>
          %323 = ascvec.loadalign %arg12[%321] : memref<?xui16, 1>, index -> <ui16>
          %324 = arith.muli %arg15, %c2_i32 : i32
          %325 = arith.index_cast %324 : i32 to index
          ascvec.storealign_intlv %arg13[%325], %322, %323 : memref<832xui16, 1>, <ui16>, <ui16>
        }
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
      %243 = arith.muli %54, %c512_i64 : i64
      %244 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %245 = cannir.extract_shape %244 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %246 = cannir.extract_dim %245[0] : !cannir.shape<(-1, -1)>
      %247 = arith.minsi %78, %246 : i64
      %248 = arith.subi %246, %247 : i64
      %249 = arith.minsi %248, %c1_i64 : i64
      %250 = arith.cmpi sle, %78, %246 : i64
      %251 = arith.select %250, %249, %c0_i64 : i64
      %252 = arith.cmpi sgt, %251, %c1_i64 : i64
      %253 = arith.select %252, %c1_i64, %251 : i64
      %254 = cannir.extract_dim %245[1] : !cannir.shape<(-1, -1)>
      %255 = arith.maxsi %243, %c1_i64 : i64
      %256 = arith.divsi %254, %255 : i64
      %257 = arith.minsi %80, %256 : i64
      %258 = arith.muli %257, %243 : i64
      %259 = arith.subi %254, %258 : i64
      %260 = arith.minsi %243, %259 : i64
      %261 = arith.cmpi sle, %80, %256 : i64
      %262 = arith.select %261, %260, %c0_i64 : i64
      %263 = arith.cmpi sgt, %262, %c26624_i64 : i64
      %264 = arith.select %263, %c26624_i64, %262 : i64
      %265 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %266 = cannir.extract_stride %265 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %267 = cannir.extract_dim %266[0] : !cannir.stride<(-1, 1)>
      %268 = arith.muli %78, %267 : i64
      %269 = arith.muli %80, %243 : i64
      %270 = arith.addi %268, %269 : i64
      %271 = arith.muli %270, %c8_i64 : i64
      %272 = ascscalar.memref_address %alloc_1 : memref<1xui8>
      %273 = arith.divui %271, %c8_i64 : i64
      %274 = arith.addi %272, %273 : i64
      %275 = ascscalar.bind_gm_ptr %274 : memref<1xui8>
      %276 = ascscalar.bitcast_memref %alloc_13 byte_offset = %c0_i64 : memref<26624xui8, 1> -> memref<?xui8, 1>
      %277 = cannir.extract_stride %244 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %278 = cannir.extract_dim %277[0] : !cannir.stride<(-1, 1)>
      %279 = arith.cmpi ugt, %253, %c0_i64 : i64
      %280 = arith.cmpi ugt, %264, %c0_i64 : i64
      %281 = arith.andi %279, %280 : i1
      scf.if %281 {
        ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
        %321 = arith.trunci %253 : i64 to i32
        %322 = arith.trunci %264 : i64 to i32
        ascvec.copy_ub2gm %275[%c0], %276[%c0], %321, %322, %278, %c26624_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
        ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
      }
      %282 = arith.muli %54, %c16_i64 : i64
      %283 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %284 = cannir.extract_shape %283 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %285 = cannir.extract_dim %284[0] : !cannir.shape<(-1, -1)>
      %286 = arith.minsi %78, %285 : i64
      %287 = arith.subi %285, %286 : i64
      %288 = arith.minsi %287, %c1_i64 : i64
      %289 = arith.cmpi sle, %78, %285 : i64
      %290 = arith.select %289, %288, %c0_i64 : i64
      %291 = arith.cmpi sgt, %290, %c1_i64 : i64
      %292 = arith.select %291, %c1_i64, %290 : i64
      %293 = cannir.extract_dim %284[1] : !cannir.shape<(-1, -1)>
      %294 = arith.maxsi %282, %c1_i64 : i64
      %295 = arith.divsi %293, %294 : i64
      %296 = arith.minsi %80, %295 : i64
      %297 = arith.muli %296, %282 : i64
      %298 = arith.subi %293, %297 : i64
      %299 = arith.minsi %282, %298 : i64
      %300 = arith.cmpi sle, %80, %295 : i64
      %301 = arith.select %300, %299, %c0_i64 : i64
      %302 = arith.cmpi sgt, %301, %c1664_i64 : i64
      %303 = arith.select %302, %c1664_i64, %301 : i64
      %304 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %305 = cannir.extract_stride %304 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %306 = cannir.extract_dim %305[0] : !cannir.stride<(-1, 1)>
      %307 = arith.muli %78, %306 : i64
      %308 = arith.muli %80, %282 : i64
      %309 = arith.addi %307, %308 : i64
      %310 = arith.muli %309, %c8_i64 : i64
      %311 = ascscalar.memref_address %alloc_2 : memref<1xui8>
      %312 = arith.divui %310, %c8_i64 : i64
      %313 = arith.addi %311, %312 : i64
      %314 = ascscalar.bind_gm_ptr %313 : memref<1xui8>
      %315 = ascscalar.bitcast_memref %alloc_14 byte_offset = %c0_i64 : memref<832xui16, 1> -> memref<?xui8, 1>
      %316 = cannir.extract_stride %283 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %317 = cannir.extract_dim %316[0] : !cannir.stride<(-1, 1)>
      %318 = arith.cmpi ugt, %292, %c0_i64 : i64
      %319 = arith.cmpi ugt, %303, %c0_i64 : i64
      %320 = arith.andi %318, %319 : i1
      scf.if %320 {
        ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
        %321 = arith.trunci %292 : i64 to i32
        %322 = arith.trunci %303 : i64 to i32
        ascvec.copy_ub2gm %314[%c0], %315[%c0], %321, %322, %317, %c1664_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
        ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
      }
    }
    return
  }
}
