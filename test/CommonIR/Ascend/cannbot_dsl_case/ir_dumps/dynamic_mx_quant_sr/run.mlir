module {
  func.func @run_eac66d6188e0242c6df03a6ea5de137b(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>) attributes {cannbotdsl.contract_digest = "eac66d6188e0242c6df03a6ea5de137b0605dec8b640c9ff11c638801fb69ca9", cannbotdsl.public_slot_count = 6 : i64, cannbotdsl.slot_dtype_bits = [16, 8, 8, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 1, 1, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 2, 2, 1], cannbotdsl.slot_type_digests = ["b61381ca951a0f603976832b4955166996f3e8a7d1460da4be1b89bae4e92f47", "481dbcd23d79e136c72cda6b88f83676c3549f81b3cd3be5caf1bc2d99f3440f", "f3b37a0bd876490cb7c40dd697b264fbf7c668be5608a770181d9307f4b9c55d", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "4126671a83041647cd98afed881383228865da287ea44573939f32b1d956765b"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel___call___0(%arg4) [%arg0, %arg1, %arg2, %arg3, %arg5] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 4>, name = "_DynamicMxQuantSrKernel___call__", op_type = "_DynamicMxQuantSrKernel___call__", roles = array<i64: 0, 1, 1, 0>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel___call___0(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel___call___0", cannir.resource.ids_prepared} {
    %c2048_i32 = arith.constant 2048 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c1 = arith.constant 1 : index
    %c64_i64 = arith.constant 64 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c15_i64 = arith.constant 15 : i64
    %c16_i64 = arith.constant 16 : i64
    %c0_i32 = arith.constant 0 : i32
    %c65536_i64 = arith.constant 65536 : i64
    %c256_i64 = arith.constant 256 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c1023_i64 = arith.constant 1023 : i64
    %c4294967296_i64 = arith.constant 4294967296 : i64
    %c32_i64 = arith.constant 32 : i64
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
    %c2_i32 = arith.constant 2 : i32
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c14464_i64 = arith.constant 14464 : i64
    %c13568_i64 = arith.constant 13568 : i64
    %c63_i64 = arith.constant 63 : i64
    %c512_i64 = arith.constant 512 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c128_i64 = arith.constant 128 : i64
    %c6_i64 = arith.constant 6 : i64
    %c7_i64 = arith.constant 7 : i64
    %c255_i64 = arith.constant 255 : i64
    %c8_i64 = arith.constant 8 : i64
    %c32767_i64 = arith.constant 32767 : i64
    %c511_i64 = arith.constant 511 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %false = arith.constant false
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xui8>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xui8>
    %alloc_2 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<2xi64>
    %alloc_3 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<32768xbf16, 1>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<32768xbf16, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 131072 : i64} : memref<32768xui8, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 163840 : i64} : memref<2048xui8, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 165888 : i64} : memref<1024xui32, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 169984 : i64} : memref<8192xi32, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 202752 : i64} : memref<2048xui16, 1>
    %0 = ascscalar.scalar_ptr %alloc_2[%c0] : memref<2xi64>, index -> <gm, i64>
    %1 = ascscalar.load_dev %0 : !ascscalar.scalar_ptr<gm, i64> -> i64
    %2 = ascscalar.scalar_ptr %alloc_2[%c1] : memref<2xi64>, index -> <gm, i64>
    %3 = ascscalar.load_dev %2 : !ascscalar.scalar_ptr<gm, i64> -> i64
    %4 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
    %5 = cannir.extract_shape %4 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
    %6 = cannir.extract_dim %5[0] : !cannir.shape<(-1, -1)>
    %7 = arith.cmpi eq, %6, %c1_i64 : i64
    %8 = scf.if %7 -> (i1) {
      %9 = cannir.extract_dim %5[1] : !cannir.shape<(-1, -1)>
      %10 = arith.remsi %9, %c64_i64 : i64
      %11 = arith.cmpi ne, %10, %c0_i64 : i64
      %12 = arith.cmpi slt, %10, %c0_i64 : i64
      %13 = arith.andi %11, %12 : i1
      %14 = arith.select %13, %c64_i64, %c0_i64 : i64
      %15 = arith.addi %10, %14 overflow<nsw> : i64
      %16 = arith.cmpi eq, %15, %c0_i64 : i64
      %17 = scf.if %16 -> (i1) {
        %18 = ascscalar.get_block_num
        %19 = arith.muli %18, %c32768_i64 : i64
        %20 = arith.cmpi sgt, %9, %19 : i64
        %21 = scf.if %20 -> (i1) {
          %22 = arith.remsi %9, %19 : i64
          %23 = arith.cmpi ne, %22, %c0_i64 : i64
          %24 = arith.cmpi slt, %22, %c0_i64 : i64
          %25 = arith.cmpi slt, %19, %c0_i64 : i64
          %26 = arith.xori %24, %25 : i1
          %27 = arith.andi %23, %26 : i1
          %28 = arith.select %27, %19, %c0_i64 : i64
          %29 = arith.addi %22, %28 : i64
          %30 = arith.cmpi ne, %29, %c0_i64 : i64
          scf.yield %30 : i1
        } else {
          scf.yield %false : i1
        }
        scf.yield %21 : i1
      } else {
        scf.yield %false : i1
      }
      scf.yield %17 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %8 {
      %9 = cannir.extract_dim %5[1] : !cannir.shape<(-1, -1)>
      %10 = arith.remsi %9, %c64_i64 : i64
      %11 = arith.cmpi ne, %10, %c0_i64 : i64
      %12 = arith.cmpi slt, %10, %c0_i64 : i64
      %13 = arith.andi %11, %12 : i1
      %14 = arith.divsi %9, %c64_i64 : i64
      %15 = arith.extui %13 : i1 to i64
      %16 = arith.subi %14, %15 : i64
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
      %29 = arith.cmpi sge, %28, %c4_i64 : i64
      %30 = scf.if %29 -> (i64) {
        %50 = arith.addi %28, %c3_i64 : i64
        %51 = arith.remsi %50, %c4_i64 : i64
        %52 = arith.cmpi ne, %51, %c0_i64 : i64
        %53 = arith.cmpi slt, %51, %c0_i64 : i64
        %54 = arith.andi %52, %53 : i1
        %55 = arith.divsi %50, %c4_i64 : i64
        %56 = arith.extui %54 : i1 to i64
        %57 = arith.subi %55, %56 : i64
        %58 = arith.muli %57, %c4_i64 : i64
        scf.yield %58 : i64
      } else {
        scf.yield %28 : i64
      }
      %31 = arith.muli %17, %c64_i64 : i64
      %32 = arith.cmpi sge, %16, %31 : i64
      %33 = scf.if %32 -> (i64) {
        %50 = arith.addi %30, %c15_i64 : i64
        %51 = arith.remsi %50, %c16_i64 : i64
        %52 = arith.cmpi ne, %51, %c0_i64 : i64
        %53 = arith.cmpi slt, %51, %c0_i64 : i64
        %54 = arith.andi %52, %53 : i1
        %55 = arith.divsi %50, %c16_i64 : i64
        %56 = arith.extui %54 : i1 to i64
        %57 = arith.subi %55, %56 : i64
        %58 = arith.muli %57, %c16_i64 : i64
        scf.yield %58 : i64
      } else {
        scf.yield %30 : i64
      }
      %34 = arith.muli %33, %c64_i64 : i64
      %35 = ascscalar.get_block_idx
      %36 = arith.muli %35, %34 : i64
      %37 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %38 = cannir.extract_shape %37 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %39 = cannir.extract_dim %38[1] : !cannir.shape<(-1, -1)>
      %40 = arith.remsi %9, %c16_i64 : i64
      %41 = arith.cmpi ne, %40, %c0_i64 : i64
      %42 = arith.cmpi slt, %40, %c0_i64 : i64
      %43 = arith.andi %41, %42 : i1
      %44 = arith.divsi %9, %c16_i64 : i64
      %45 = arith.extui %43 : i1 to i64
      %46 = arith.subi %44, %45 : i64
      %47 = arith.cmpi eq, %39, %46 : i64
      %48 = arith.select %47, %c4_i64, %c2_i64 : i64
      %49 = arith.cmpi slt, %36, %9 : i64
      scf.if %49 {
        %50 = arith.muli %33, %48 : i64
        %51 = arith.subi %9, %36 : i64
        %52 = arith.cmpi sgt, %51, %34 : i64
        %53 = arith.select %52, %34, %51 : i64
        %54 = arith.addi %53, %c32767_i64 : i64
        %55 = arith.remsi %54, %c32768_i64 : i64
        %56 = arith.cmpi ne, %55, %c0_i64 : i64
        %57 = arith.cmpi slt, %55, %c0_i64 : i64
        %58 = arith.andi %56, %57 : i1
        %59 = arith.divsi %54, %c32768_i64 : i64
        %60 = arith.extui %58 : i1 to i64
        %61 = arith.subi %59, %60 : i64
        %62 = scf.for %arg5 = %c0_i64 to %61 step %c1_i64 iter_args(%arg6 = %c0_i64) -> (i64)  : i64 {
          %63 = arith.muli %arg5, %c32768_i64 : i64
          %64 = arith.subi %53, %63 : i64
          %65 = arith.cmpi sgt, %64, %c32768_i64 : i64
          %66 = arith.select %65, %c32768_i64, %64 : i64
          %67 = arith.cmpi eq, %arg6, %c0_i64 : i64
          %68 = ascscalar.memref_address %alloc_3 : memref<32768xbf16, 1>
          %69 = ascscalar.memref_address %alloc_4 : memref<32768xbf16, 1>
          %70 = arith.select %67, %68, %69 : i64
          %71 = arith.cmpi ne, %arg6, %c0_i64 : i64
          %72 = arith.extui %71 : i1 to i64
          %73 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
          %74 = arith.remui %73, %c2_i64 : i64
          %75 = arith.minsi %6, %c0_i64 : i64
          %76 = arith.subi %6, %75 : i64
          %77 = arith.minsi %76, %c1_i64 : i64
          %78 = arith.cmpi sge, %6, %c0_i64 : i64
          %79 = arith.select %78, %77, %c0_i64 : i64
          %80 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %81 = cannir.extract_stride %80 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %82 = cannir.extract_dim %81[0] : !cannir.stride<(-1, 1)>
          %83 = arith.muli %82, %c2_i64 : i64
          %84 = arith.maxsi %34, %c1_i64 : i64
          %85 = arith.divsi %9, %84 : i64
          %86 = arith.minsi %35, %85 : i64
          %87 = arith.muli %86, %34 : i64
          %88 = arith.subi %9, %87 : i64
          %89 = arith.minsi %34, %88 : i64
          %90 = arith.cmpi sle, %35, %85 : i64
          %91 = arith.select %90, %89, %c0_i64 : i64
          %92 = arith.divsi %91, %c32768_i64 : i64
          %93 = arith.minsi %arg5, %92 : i64
          %94 = arith.muli %93, %c32768_i64 : i64
          %95 = arith.subi %91, %94 : i64
          %96 = arith.minsi %95, %c32768_i64 : i64
          %97 = arith.cmpi sle, %arg5, %92 : i64
          %98 = arith.select %97, %96, %c0_i64 : i64
          %99 = arith.muli %arg5, %c65536_i64 : i64
          %100 = arith.muli %36, %c2_i64 : i64
          %101 = arith.addi %99, %100 : i64
          %102 = arith.muli %98, %c2_i64 : i64
          %103 = arith.cmpi eq, %102, %c0_i64 : i64
          %104 = arith.select %103, %c0_i64, %79 : i64
          %105 = arith.select %103, %c2_i64, %102 : i64
          %106 = ascscalar.bind_local_ptr %70 : memref<1xbf16, 1>
          ascvec.get_buf mte2, %72, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%106, %alloc, %c0_i32, %c0_i32, %c0_i64, %101, %104, %105, %83, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %72, 0 {cannir.buffer.generated_sync}
          %107 = ascscalar.bind_local_ptr %70 : memref<32768xbf16, 1>
          %108 = arith.addi %36, %63 : i64
          %109 = arith.cmpi sge, %66, %c256_i64 : i64
          %110 = arith.remsi %66, %c256_i64 : i64
          %111 = arith.cmpi eq, %110, %c0_i64 : i64
          %112 = arith.select %109, %111, %false : i1
          scf.if %112 {
            %192 = arith.remsi %arg3, %c256_i64 : i64
            %193 = arith.cmpi ne, %192, %c0_i64 : i64
            %194 = arith.cmpi slt, %192, %c0_i64 : i64
            %195 = arith.andi %193, %194 : i1
            %196 = arith.select %195, %c256_i64, %c0_i64 : i64
            %197 = arith.addi %192, %196 overflow<nsw> : i64
            %198 = arith.cmpi eq, %197, %c0_i64 : i64
            scf.if %198 {
              %199 = arith.cmpi sge, %66, %c1024_i64 : i64
              %200 = arith.remsi %66, %c1024_i64 : i64
              %201 = arith.cmpi eq, %200, %c0_i64 : i64
              %202 = arith.select %199, %201, %false : i1
              scf.if %202 {
                %203 = arith.divsi %66, %c256_i64 : i64
                %204 = arith.addi %66, %c1023_i64 : i64
                %205 = arith.divsi %204, %c1024_i64 : i64
                %206 = arith.trunci %1 : i64 to i32
                %207 = arith.extui %206 : i32 to i64
                %208 = arith.subi %1, %207 : i64
                %209 = arith.remsi %208, %c4294967296_i64 : i64
                %210 = arith.cmpi ne, %209, %c0_i64 : i64
                %211 = arith.cmpi slt, %209, %c0_i64 : i64
                %212 = arith.andi %210, %211 : i1
                %213 = arith.divsi %208, %c4294967296_i64 : i64
                %214 = arith.extui %212 : i1 to i64
                %215 = arith.subi %213, %214 : i64
                %216 = arith.trunci %215 : i64 to i32
                %217 = arith.divui %108, %c32_i64 : i64
                %218 = arith.trunci %217 : i64 to i32
                %219 = arith.extui %218 : i32 to i64
                %220 = arith.subi %217, %219 : i64
                %221 = arith.remsi %220, %c4294967296_i64 : i64
                %222 = arith.cmpi ne, %221, %c0_i64 : i64
                %223 = arith.cmpi slt, %221, %c0_i64 : i64
                %224 = arith.andi %222, %223 : i1
                %225 = arith.divsi %220, %c4294967296_i64 : i64
                %226 = arith.extui %224 : i1 to i64
                %227 = arith.subi %225, %226 : i64
                %228 = arith.trunci %227 : i64 to i32
                %229 = arith.trunci %3 : i64 to i32
                %230 = arith.extui %229 : i32 to i64
                %231 = arith.subi %3, %230 : i64
                %232 = arith.remsi %231, %c4294967296_i64 : i64
                %233 = arith.cmpi ne, %232, %c0_i64 : i64
                %234 = arith.cmpi slt, %232, %c0_i64 : i64
                %235 = arith.andi %233, %234 : i1
                %236 = arith.divsi %231, %c4294967296_i64 : i64
                %237 = arith.extui %235 : i1 to i64
                %238 = arith.subi %236, %237 : i64
                %239 = arith.trunci %238 : i64 to i32
                %240 = arith.divui %229, %c4_i32 : i32
                %241 = arith.remui %239, %c4_i32 : i32
                %242 = arith.muli %241, %c1073741824_i32 : i32
                %243 = arith.addi %240, %242 : i32
                %244 = arith.divui %239, %c4_i32 : i32
                %245 = arith.muli %205, %c32_i64 : i64
                %246 = arith.divui %217, %c16384_i64 : i64
                %247 = arith.addi %217, %245 : i64
                %248 = arith.subi %247, %c1_i64 : i64
                %249 = arith.divui %248, %c16384_i64 : i64
                %250 = arith.cmpi eq, %246, %249 : i64
                scf.if %250 {
                  %260 = arith.trunci %246 : i64 to i32
                  %261 = arith.extui %260 : i32 to i64
                  %262 = arith.subi %246, %261 : i64
                  %263 = arith.remsi %262, %c4294967296_i64 : i64
                  %264 = arith.cmpi ne, %263, %c0_i64 : i64
                  %265 = arith.cmpi slt, %263, %c0_i64 : i64
                  %266 = arith.andi %264, %265 : i1
                  %267 = arith.divsi %262, %c4294967296_i64 : i64
                  %268 = arith.extui %266 : i1 to i64
                  %269 = arith.subi %267, %268 : i64
                  %270 = arith.trunci %269 : i64 to i32
                  %271 = arith.addi %243, %260 : i32
                  %272 = arith.addi %244, %270 : i32
                  %273 = arith.cmpi ult, %271, %243 : i32
                  %274 = arith.extui %273 : i1 to i32
                  %275 = arith.addi %272, %274 : i32
                  %276 = arith.extui %271 : i32 to i64
                  %277 = arith.muli %276, %c3449720151_i64 : i64
                  %278 = arith.trunci %277 : i64 to i32
                  %279 = arith.extui %278 : i32 to i64
                  %280 = arith.subi %277, %279 : i64
                  %281 = arith.remsi %280, %c4294967296_i64 : i64
                  %282 = arith.cmpi ne, %281, %c0_i64 : i64
                  %283 = arith.cmpi slt, %281, %c0_i64 : i64
                  %284 = arith.andi %282, %283 : i1
                  %285 = arith.divsi %280, %c4294967296_i64 : i64
                  %286 = arith.extui %284 : i1 to i64
                  %287 = arith.subi %285, %286 : i64
                  %288 = arith.trunci %287 : i64 to i32
                  %289 = arith.xori %275, %216 : i32
                  %290 = arith.extui %289 : i32 to i64
                  %291 = arith.muli %290, %c3449720151_i64 : i64
                  %292 = arith.trunci %291 : i64 to i32
                  %293 = arith.extui %292 : i32 to i64
                  %294 = arith.subi %291, %293 : i64
                  %295 = arith.remsi %294, %c4294967296_i64 : i64
                  %296 = arith.cmpi ne, %295, %c0_i64 : i64
                  %297 = arith.cmpi slt, %295, %c0_i64 : i64
                  %298 = arith.andi %296, %297 : i1
                  %299 = arith.divsi %294, %c4294967296_i64 : i64
                  %300 = arith.extui %298 : i1 to i64
                  %301 = arith.subi %299, %300 : i64
                  %302 = arith.trunci %301 : i64 to i32
                  %303 = arith.addi %206, %c-1640531527_i32 : i32
                  %304 = arith.addi %216, %c-1150833019_i32 : i32
                  %305 = arith.xori %302, %278 : i32
                  %306 = arith.xori %305, %303 : i32
                  %307 = arith.extui %306 : i32 to i64
                  %308 = arith.muli %307, %c3528531795_i64 : i64
                  %309 = arith.trunci %308 : i64 to i32
                  %310 = arith.extui %309 : i32 to i64
                  %311 = arith.subi %308, %310 : i64
                  %312 = arith.remsi %311, %c4294967296_i64 : i64
                  %313 = arith.cmpi ne, %312, %c0_i64 : i64
                  %314 = arith.cmpi slt, %312, %c0_i64 : i64
                  %315 = arith.andi %313, %314 : i1
                  %316 = arith.divsi %311, %c4294967296_i64 : i64
                  %317 = arith.extui %315 : i1 to i64
                  %318 = arith.subi %316, %317 : i64
                  %319 = arith.trunci %318 : i64 to i32
                  %320 = arith.xori %288, %206 : i32
                  %321 = arith.addi %206, %c1013904242_i32 : i32
                  %322 = arith.xori %292, %321 : i32
                  %323 = arith.addi %216, %c1993301258_i32 : i32
                  %324 = arith.xori %319, %323 : i32
                  ascsimt.vf(%218, %320, %304, %322, %324, %206, %216, %309, %alloc_8, %245 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
                    %c256_i64_10 = arith.constant 256 : i64
                    %c3_i64_11 = arith.constant 3 : i64
                    %c2_i64_12 = arith.constant 2 : i64
                    %c1_i64_13 = arith.constant 1 : i64
                    %c4_i64_14 = arith.constant 4 : i64
                    %c1_i32_15 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c3_i32 = arith.constant 3 : i32
                    %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %325 = ascsimt.thread_idx {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %326 = arith.extsi %325 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg17 = %326 to %arg16 step %c256_i64_10  : i64 {
                      %327 = arith.trunci %arg17 : i64 to i32
                      %328 = arith.addi %arg7, %327 : i32
                      %329 = arith.remui %328, %c16384_i32 : i32
                      %330 = arith.xori %329, %arg8 : i32
                      %331 = arith.muli %330, %c-766435501_i32 : i32
                      %332 = "ascsimt.bitcast"(%330) : (i32) -> ui32
                      %333 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %334 = ascsimt.math_binary %332, %333, "mulhi" : ui32, ui32 -> ui32
                      %335 = "ascsimt.bitcast"(%334) : (ui32) -> i32
                      %336 = arith.xori %335, %arg9 : i32
                      %337 = arith.muli %336, %c-845247145_i32 : i32
                      %338 = "ascsimt.bitcast"(%336) : (i32) -> ui32
                      %339 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %340 = ascsimt.math_binary %338, %339, "mulhi" : ui32, ui32 -> ui32
                      %341 = "ascsimt.bitcast"(%340) : (ui32) -> i32
                      %342 = arith.xori %341, %arg10 : i32
                      %343 = arith.xori %331, %arg11 : i32
                      %344:4 = scf.for %arg18 = %c3_i32 to %c10_i32 step %c1_i32_15 iter_args(%arg19 = %342, %arg20 = %337, %arg21 = %343, %arg22 = %arg14) -> (i32, i32, i32, i32)  : i32 {
                        %353 = arith.muli %arg19, %c-766435501_i32 : i32
                        %354 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %355 = ascsimt.math_binary %354, %333, "mulhi" : ui32, ui32 -> ui32
                        %356 = "ascsimt.bitcast"(%355) : (ui32) -> i32
                        %357 = arith.muli %arg21, %c-845247145_i32 : i32
                        %358 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %359 = ascsimt.math_binary %358, %339, "mulhi" : ui32, ui32 -> ui32
                        %360 = "ascsimt.bitcast"(%359) : (ui32) -> i32
                        %361 = arith.muli %arg18, %c-1640531527_i32_17 : i32
                        %362 = arith.addi %arg12, %361 : i32
                        %363 = arith.muli %arg18, %c-1150833019_i32_16 : i32
                        %364 = arith.addi %arg13, %363 : i32
                        %365 = arith.xori %360, %arg20 : i32
                        %366 = arith.xori %365, %362 : i32
                        %367 = arith.xori %356, %arg22 : i32
                        %368 = arith.xori %367, %364 : i32
                        scf.yield %366, %357, %368, %353 : i32, i32, i32, i32
                      }
                      %345 = arith.muli %arg17, %c4_i64_14 : i64
                      %346 = arith.index_cast %345 : i64 to index
                      ascsimt.store %arg15[%346], %344#0 : memref<8192xi32, 1>, i32
                      %347 = arith.addi %345, %c1_i64_13 : i64
                      %348 = arith.index_cast %347 : i64 to index
                      ascsimt.store %arg15[%348], %344#1 : memref<8192xi32, 1>, i32
                      %349 = arith.addi %345, %c2_i64_12 : i64
                      %350 = arith.index_cast %349 : i64 to index
                      ascsimt.store %arg15[%350], %344#2 : memref<8192xi32, 1>, i32
                      %351 = arith.addi %345, %c3_i64_11 : i64
                      %352 = arith.index_cast %351 : i64 to index
                      ascsimt.store %arg15[%352], %344#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%218, %228, %243, %244, %206, %216, %alloc_8, %245 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: memref<8192xi32, 1>, %arg14: i64):
                    %c256_i64_10 = arith.constant 256 : i64
                    %c3_i64_11 = arith.constant 3 : i64
                    %c2_i64_12 = arith.constant 2 : i64
                    %c1_i64_13 = arith.constant 1 : i64
                    %c4_i64_14 = arith.constant 4 : i64
                    %c1_i32_15 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c0_i32_16 = arith.constant 0 : i32
                    %c-1150833019_i32_17 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_18 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c262144_i32 = arith.constant 262144 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %260 = ascsimt.thread_idx {ascsimt.scope = 2 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %261 = arith.extsi %260 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg15 = %261 to %arg14 step %c256_i64_10  : i64 {
                      %262 = arith.trunci %arg15 : i64 to i32
                      %263 = arith.addi %arg7, %262 : i32
                      %264 = arith.cmpi ult, %263, %arg7 : i32
                      %265 = arith.extui %264 : i1 to i32
                      %266 = arith.addi %arg8, %265 : i32
                      %267 = arith.divui %263, %c16384_i32 : i32
                      %268 = arith.remui %266, %c16384_i32 : i32
                      %269 = arith.muli %268, %c262144_i32 : i32
                      %270 = arith.addi %267, %269 : i32
                      %271 = arith.divui %266, %c16384_i32 : i32
                      %272 = arith.addi %arg9, %270 : i32
                      %273 = arith.addi %arg10, %271 : i32
                      %274 = arith.cmpi ult, %272, %arg9 : i32
                      %275 = arith.extui %274 : i1 to i32
                      %276 = arith.addi %273, %275 : i32
                      %277 = arith.remui %263, %c16384_i32 : i32
                      %278:4 = scf.for %arg16 = %c0_i32_16 to %c10_i32 step %c1_i32_15 iter_args(%arg17 = %c0_i32_16, %arg18 = %277, %arg19 = %272, %arg20 = %276) -> (i32, i32, i32, i32)  : i32 {
                        %287 = arith.muli %arg17, %c-766435501_i32 : i32
                        %288 = "ascsimt.bitcast"(%arg17) : (i32) -> ui32
                        %289 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %290 = ascsimt.math_binary %288, %289, "mulhi" : ui32, ui32 -> ui32
                        %291 = "ascsimt.bitcast"(%290) : (ui32) -> i32
                        %292 = arith.muli %arg19, %c-845247145_i32 : i32
                        %293 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %294 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %295 = ascsimt.math_binary %293, %294, "mulhi" : ui32, ui32 -> ui32
                        %296 = "ascsimt.bitcast"(%295) : (ui32) -> i32
                        %297 = arith.muli %arg16, %c-1640531527_i32_18 : i32
                        %298 = arith.addi %arg11, %297 : i32
                        %299 = arith.muli %arg16, %c-1150833019_i32_17 : i32
                        %300 = arith.addi %arg12, %299 : i32
                        %301 = arith.xori %296, %arg18 : i32
                        %302 = arith.xori %301, %298 : i32
                        %303 = arith.xori %291, %arg20 : i32
                        %304 = arith.xori %303, %300 : i32
                        scf.yield %302, %292, %304, %287 : i32, i32, i32, i32
                      }
                      %279 = arith.muli %arg15, %c4_i64_14 : i64
                      %280 = arith.index_cast %279 : i64 to index
                      ascsimt.store %arg13[%280], %278#0 : memref<8192xi32, 1>, i32
                      %281 = arith.addi %279, %c1_i64_13 : i64
                      %282 = arith.index_cast %281 : i64 to index
                      ascsimt.store %arg13[%282], %278#1 : memref<8192xi32, 1>, i32
                      %283 = arith.addi %279, %c2_i64_12 : i64
                      %284 = arith.index_cast %283 : i64 to index
                      ascsimt.store %arg13[%284], %278#2 : memref<8192xi32, 1>, i32
                      %285 = arith.addi %279, %c3_i64_11 : i64
                      %286 = arith.index_cast %285 : i64 to index
                      ascsimt.store %arg13[%286], %278#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %251 = arith.trunci %48 : i64 to i32
                %252 = arith.divsi %251, %c2_i32 : i32
                %253 = arith.trunci %252 : i32 to i8
                %254 = arith.trunci %203 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %255 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%253, %alloc_6, %107, %255, %alloc_9, %254 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, memref<2048xui16, 1>, i32) {
                ^bb0(%arg7: i8, %arg8: memref<2048xui8, 1>, %arg9: memref<32768xbf16, 1>, %arg10: memref<?xui16, 1>, %arg11: memref<2048xui16, 1>, %arg12: i32):
                  %c16_10 = arith.constant 16 : index
                  %c4_i32_11 = arith.constant 4 : i32
                  %c0_i32_12 = arith.constant 0 : i32
                  %c3_i32 = arith.constant 3 : i32
                  %c2_i32_13 = arith.constant 2 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c1_i32_14 = arith.constant 1 : i32
                  %c0_15 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %260 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %261 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %262 = ascvec.duplicate_scalar %c32767_i16, %260 : i16 -> <ui16>
                  %263 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %264 = ascvec.shr %263, 4, %260 : <ui16> -> <ui16>
                  %265 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %266 = ascvec.duplicate_scalar %c1_i8, %261 : i8 -> <ui8>
                  %267 = ascvec.and %265, %266, %261 : <ui8>, <ui8> -> <ui8>
                  %268 = ascvec.cmp_lt_scalar %265, %c16_i8, %261 : <ui8>, i8 -> !ascvec.mask
                  %269 = ascvec.cmp_lt_scalar %267, %arg7, %261 : <ui8>, i8 -> !ascvec.mask
                  %270 = ascvec.mask_and %268, %269, %261
                  ascvec.store_unalign_begin %arg8 : memref<2048xui8, 1>
                  %mask_16, %remaining_out_17 = ascvec.update_mask %c1_i32_14 elem_bits 16 : !ascvec.mask
                  %271 = ascvec.duplicate_scalar %c-1_i16, %mask_16 : i16 -> <ui16>
                  %272 = scf.for %arg13 = %c0_i32_12 to %arg12 step %c4_i32_11 iter_args(%arg14 = %271) -> (!ascvec.vreg<ui16>)  : i32 {
                    %273 = ascvec.duplicate_scalar %c-1_i16, %260 : i16 -> <ui16>
                    %274 = ascvec.duplicate_scalar %c0_i16, %260 : i16 -> <ui16>
                    %275 = arith.muli %arg13, %c256_i32 : i32
                    %276 = arith.index_cast %275 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg9[%276] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %277 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %278 = ascvec.and %277, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %279 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %280 = ascvec.and %279, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %281 = ascvec.max %278, %280, %260 : <ui16>
                    %282 = ascvec.reduce_max_datablock %281, %260 : <ui16>
                    %283 = ascvec.shr %282, 7, %260 : <ui16> -> <ui16>
                    %284 = ascvec.cmp_ge_scalar %283, %c255_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %285 = ascvec.max_scalar %283, %c15_i16, %260 : <ui16>, i16
                    %286 = ascvec.add_scalar %285, %c-15_i16, %260 : <ui16>, i16
                    %287 = ascvec.duplicate_scalar %c255_i16, %284 merge %286 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %288 = ascvec.duplicate_scalar %c254_i16, %260 : i16 -> <ui16>
                    %289 = ascvec.sub %288, %287, %260 : <ui16>
                    %290 = ascvec.cmp_ne_scalar %287, %c0_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %291 = ascvec.shl %289, 7, %290 : <ui16> -> <ui16>
                    %292 = ascvec.max %274, %287, %mask : <ui16>
                    %293 = ascvec.gather_reg %291, %264 : <ui16>, <ui16> -> <ui16>
                    %294 = ascvec.reinterpret %293 : <ui16> -> <bf16>
                    %295 = ascvec.mul %result0, %294, %260 : <bf16>
                    %296 = ascvec.mul %result1, %294, %260 : <bf16>
                    ascvec.storealign_intlv %arg9[%276], %295, %296 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %297 = ascvec.reinterpret %295 : <bf16> -> <ui16>
                    %298 = ascvec.and %297, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %299 = ascvec.reinterpret %296 : <bf16> -> <ui16>
                    %300 = ascvec.and %299, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %301 = ascvec.min %298, %300, %260 : <ui16>
                    %302 = ascvec.min %273, %301, %260 : <ui16>
                    %303 = arith.muli %arg13, %c16_i32 : i32
                    %304 = arith.index_cast %303 : i32 to index
                    ascvec.storealign %arg10[%304], %287, %mask : memref<?xui16, 1>, index, <ui16>
                    %305 = ascvec.reinterpret %287 : <ui16> -> <ui8>
                    %306 = ascvec.squeeze_store_reg %305, %270 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_15], %306 : memref<2048xui8, 1>, <ui8>
                    %307 = arith.addi %arg13, %c1_i32_14 : i32
                    %308 = arith.muli %307, %c256_i32 : i32
                    %309 = arith.index_cast %308 : i32 to index
                    %result0_20, %result1_21 = ascvec.loadalign_deintlv %arg9[%309] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %310 = ascvec.reinterpret %result0_20 : <bf16> -> <ui16>
                    %311 = ascvec.and %310, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %312 = ascvec.reinterpret %result1_21 : <bf16> -> <ui16>
                    %313 = ascvec.and %312, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %314 = ascvec.max %311, %313, %260 : <ui16>
                    %315 = ascvec.reduce_max_datablock %314, %260 : <ui16>
                    %316 = ascvec.shr %315, 7, %260 : <ui16> -> <ui16>
                    %317 = ascvec.cmp_ge_scalar %316, %c255_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %318 = ascvec.max_scalar %316, %c15_i16, %260 : <ui16>, i16
                    %319 = ascvec.add_scalar %318, %c-15_i16, %260 : <ui16>, i16
                    %320 = ascvec.duplicate_scalar %c255_i16, %317 merge %319 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %321 = ascvec.sub %288, %320, %260 : <ui16>
                    %322 = ascvec.cmp_ne_scalar %320, %c0_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %323 = ascvec.shl %321, 7, %322 : <ui16> -> <ui16>
                    %324 = ascvec.max %292, %320, %mask : <ui16>
                    %325 = ascvec.gather_reg %323, %264 : <ui16>, <ui16> -> <ui16>
                    %326 = ascvec.reinterpret %325 : <ui16> -> <bf16>
                    %327 = ascvec.mul %result0_20, %326, %260 : <bf16>
                    %328 = ascvec.mul %result1_21, %326, %260 : <bf16>
                    ascvec.storealign_intlv %arg9[%309], %327, %328 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %329 = ascvec.reinterpret %327 : <bf16> -> <ui16>
                    %330 = ascvec.and %329, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %331 = ascvec.reinterpret %328 : <bf16> -> <ui16>
                    %332 = ascvec.and %331, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %333 = ascvec.min %330, %332, %260 : <ui16>
                    %334 = ascvec.min %302, %333, %260 : <ui16>
                    %335 = arith.muli %307, %c16_i32 : i32
                    %336 = arith.index_cast %335 : i32 to index
                    ascvec.storealign %arg10[%336], %320, %mask : memref<?xui16, 1>, index, <ui16>
                    %337 = ascvec.reinterpret %320 : <ui16> -> <ui8>
                    %338 = ascvec.squeeze_store_reg %337, %270 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_15], %338 : memref<2048xui8, 1>, <ui8>
                    %339 = arith.addi %arg13, %c2_i32_13 : i32
                    %340 = arith.muli %339, %c256_i32 : i32
                    %341 = arith.index_cast %340 : i32 to index
                    %result0_22, %result1_23 = ascvec.loadalign_deintlv %arg9[%341] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %342 = ascvec.reinterpret %result0_22 : <bf16> -> <ui16>
                    %343 = ascvec.and %342, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %344 = ascvec.reinterpret %result1_23 : <bf16> -> <ui16>
                    %345 = ascvec.and %344, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %346 = ascvec.max %343, %345, %260 : <ui16>
                    %347 = ascvec.reduce_max_datablock %346, %260 : <ui16>
                    %348 = ascvec.shr %347, 7, %260 : <ui16> -> <ui16>
                    %349 = ascvec.cmp_ge_scalar %348, %c255_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %350 = ascvec.max_scalar %348, %c15_i16, %260 : <ui16>, i16
                    %351 = ascvec.add_scalar %350, %c-15_i16, %260 : <ui16>, i16
                    %352 = ascvec.duplicate_scalar %c255_i16, %349 merge %351 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %353 = ascvec.sub %288, %352, %260 : <ui16>
                    %354 = ascvec.cmp_ne_scalar %352, %c0_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %355 = ascvec.shl %353, 7, %354 : <ui16> -> <ui16>
                    %356 = ascvec.max %324, %352, %mask : <ui16>
                    %357 = ascvec.gather_reg %355, %264 : <ui16>, <ui16> -> <ui16>
                    %358 = ascvec.reinterpret %357 : <ui16> -> <bf16>
                    %359 = ascvec.mul %result0_22, %358, %260 : <bf16>
                    %360 = ascvec.mul %result1_23, %358, %260 : <bf16>
                    ascvec.storealign_intlv %arg9[%341], %359, %360 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %361 = ascvec.reinterpret %359 : <bf16> -> <ui16>
                    %362 = ascvec.and %361, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %363 = ascvec.reinterpret %360 : <bf16> -> <ui16>
                    %364 = ascvec.and %363, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %365 = ascvec.min %362, %364, %260 : <ui16>
                    %366 = ascvec.min %334, %365, %260 : <ui16>
                    %367 = arith.muli %339, %c16_i32 : i32
                    %368 = arith.index_cast %367 : i32 to index
                    ascvec.storealign %arg10[%368], %352, %mask : memref<?xui16, 1>, index, <ui16>
                    %369 = ascvec.reinterpret %352 : <ui16> -> <ui8>
                    %370 = ascvec.squeeze_store_reg %369, %270 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_15], %370 : memref<2048xui8, 1>, <ui8>
                    %371 = arith.addi %arg13, %c3_i32 : i32
                    %372 = arith.muli %371, %c256_i32 : i32
                    %373 = arith.index_cast %372 : i32 to index
                    %result0_24, %result1_25 = ascvec.loadalign_deintlv %arg9[%373] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %374 = ascvec.reinterpret %result0_24 : <bf16> -> <ui16>
                    %375 = ascvec.and %374, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %376 = ascvec.reinterpret %result1_25 : <bf16> -> <ui16>
                    %377 = ascvec.and %376, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %378 = ascvec.max %375, %377, %260 : <ui16>
                    %379 = ascvec.reduce_max_datablock %378, %260 : <ui16>
                    %380 = ascvec.shr %379, 7, %260 : <ui16> -> <ui16>
                    %381 = ascvec.cmp_ge_scalar %380, %c255_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %382 = ascvec.max_scalar %380, %c15_i16, %260 : <ui16>, i16
                    %383 = ascvec.add_scalar %382, %c-15_i16, %260 : <ui16>, i16
                    %384 = ascvec.duplicate_scalar %c255_i16, %381 merge %383 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %385 = ascvec.sub %288, %384, %260 : <ui16>
                    %386 = ascvec.cmp_ne_scalar %384, %c0_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %387 = ascvec.shl %385, 7, %386 : <ui16> -> <ui16>
                    %388 = ascvec.max %356, %384, %mask : <ui16>
                    %389 = ascvec.gather_reg %387, %264 : <ui16>, <ui16> -> <ui16>
                    %390 = ascvec.reinterpret %389 : <ui16> -> <bf16>
                    %391 = ascvec.mul %result0_24, %390, %260 : <bf16>
                    %392 = ascvec.mul %result1_25, %390, %260 : <bf16>
                    ascvec.storealign_intlv %arg9[%373], %391, %392 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %393 = ascvec.reinterpret %391 : <bf16> -> <ui16>
                    %394 = ascvec.and %393, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %395 = ascvec.reinterpret %392 : <bf16> -> <ui16>
                    %396 = ascvec.and %395, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %397 = ascvec.min %394, %396, %260 : <ui16>
                    %398 = ascvec.min %366, %397, %260 : <ui16>
                    %399 = arith.muli %371, %c16_i32 : i32
                    %400 = arith.index_cast %399 : i32 to index
                    ascvec.storealign %arg10[%400], %384, %mask : memref<?xui16, 1>, index, <ui16>
                    %401 = ascvec.reinterpret %384 : <ui16> -> <ui8>
                    %402 = ascvec.squeeze_store_reg %401, %270 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_15], %402 : memref<2048xui8, 1>, <ui8>
                    %403 = ascvec.reduce_min %398, %260 : <ui16>
                    %404 = ascvec.reduce_max %388, %mask : <ui16>
                    %405 = ascvec.cmp_eq_scalar %404, %c255_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %406 = ascvec.duplicate_scalar %c0_i16, %405 merge %403 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.storealign_1st %arg11[%304], %406 : memref<2048xui16, 1>, index, <ui16>
                    %407 = ascvec.min %arg14, %406, %mask_16 : <ui16>
                    scf.yield %407 : !ascvec.vreg<ui16>
                  }
                  %c0_18 = arith.constant 0 : index
                  %c1_19 = arith.constant 1 : index
                  scf.for %arg13 = %c0_18 to %c1_19 step %c1_19 {
                    ascvec.store_unalign_post %arg8[%c0_15] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg11[%c16_10], %272 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %256 = memref.load %alloc_9[%c16] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %257 = cannir.scalar_storage_cast %256 : ui16 -> i16
                %258 = arith.extui %257 : i16 to i64
                %259 = arith.cmpi sge, %258, %c14464_i64 : i64
                scf.if %259 {
                  ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %260 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                  %261 = ascscalar.bitcast_memref %107 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%107, %260, %261, %alloc_5, %254 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg7: memref<32768xbf16, 1>, %arg8: memref<?xui8, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<32768xui8, 1>, %arg11: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_10 = arith.constant 1 : i32
                    %c0_i32_11 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c128_i32 = arith.constant 128 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c31_i16 = arith.constant 31 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %262 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %263 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %264 = ascvec.duplicate_scalar %c255_i16, %262 : i16 -> <ui16>
                    %265 = ascvec.duplicate_scalar %c-128_i8, %263 : i8 -> <ui8>
                    %266 = ascvec.duplicate_scalar %c31_i16, %262 : i16 -> <ui16>
                    scf.for %arg12 = %c0_i32_11 to %arg11 step %c1_i32_10  : i32 {
                      %267 = arith.muli %arg12, %c256_i32 : i32
                      %268 = arith.index_cast %267 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg7[%268] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %269 = arith.muli %arg12, %c128_i32 : i32
                      %270 = arith.index_cast %269 : i32 to index
                      %271 = ascvec.loadalign %arg8[%270] : memref<?xui8, 1>, index -> <ui8>
                      %272 = ascvec.unpack %271 {part = "lower"} : <ui8> -> <ui16>
                      %273 = ascvec.and %272, %266, %262 : <ui16>, <ui16> -> <ui16>
                      %274 = ascvec.shr %272, 3, %262 : <ui16> -> <ui16>
                      %275 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %276 = ascvec.add %275, %273, %262 : <ui16>
                      %277 = ascvec.shr %276, 5, %262 : <ui16> -> <ui16>
                      %278 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %279 = ascvec.add %278, %274, %262 : <ui16>
                      %280 = ascvec.shr %279, 5, %262 : <ui16> -> <ui16>
                      %281 = ascvec.and %277, %264, %262 : <ui16>, <ui16> -> <ui16>
                      %282 = ascvec.shl %280, 8, %262 : <ui16> -> <ui16>
                      %283 = ascvec.or %281, %282, %262 : <ui16>, <ui16> -> <ui16>
                      %284 = ascvec.reinterpret %283 : <ui16> -> <ui8>
                      %285 = ascvec.add_scalar %284, %c64_i8, %263 : <ui8>, i8
                      %286 = ascvec.min_scalar %285, %c123_i8, %263 : <ui8>, i8
                      %287 = arith.muli %arg12, %c512_i32 : i32
                      %288 = arith.index_cast %287 : i32 to index
                      %result0_12, %result1_13 = ascvec.loadalign_deintlv %arg9[%288] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %289 = ascvec.and %result1_13, %265, %263 : <ui8>, <ui8> -> <ui8>
                      %290 = ascvec.or %286, %289, %263 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg10[%268], %290, %263 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
                } else {
                  scf.for %arg7 = %c0_i64 to %203 step %c16_i64  : i64 {
                    %260 = arith.addi %arg7, %c16_i64 : i64
                    %261 = arith.cmpi sgt, %260, %203 : i64
                    %262 = arith.select %261, %203, %260 : i64
                    %263 = arith.muli %arg7, %c16_i64 : i64
                    ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %264 = arith.index_cast %263 : i64 to index
                    %265 = memref.load %alloc_9[%264] : memref<2048xui16, 1>
                    ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %266 = cannir.scalar_storage_cast %265 : ui16 -> i16
                    %267 = arith.extui %266 : i16 to i64
                    %268 = arith.addi %arg7, %c4_i64 : i64
                    %269 = scf.for %arg8 = %268 to %262 step %c4_i64 iter_args(%arg9 = %267) -> (i64)  : i64 {
                      %271 = arith.muli %arg8, %c16_i64 : i64
                      ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %272 = arith.index_cast %271 : i64 to index
                      %273 = memref.load %alloc_9[%272] : memref<2048xui16, 1>
                      ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %274 = cannir.scalar_storage_cast %273 : ui16 -> i16
                      %275 = arith.extui %274 : i16 to i64
                      %276 = arith.cmpi slt, %275, %arg9 : i64
                      %277 = arith.select %276, %275, %arg9 : i64
                      scf.yield %277 : i64
                    }
                    %270 = arith.cmpi sge, %269, %c14464_i64 : i64
                    scf.if %270 {
                      %271 = arith.trunci %arg7 : i64 to i32
                      %272 = arith.trunci %262 : i64 to i32
                      ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %273 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                      %274 = ascscalar.bitcast_memref %107 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                      ascvec.vf(%107, %273, %274, %alloc_5, %271, %272 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                      ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32, %arg13: i32):
                        %c512_i32 = arith.constant 512 : i32
                        %c1_i32_10 = arith.constant 1 : i32
                        %c123_i8 = arith.constant 123 : i8
                        %c64_i8 = arith.constant 64 : i8
                        %c128_i32 = arith.constant 128 : i32
                        %c256_i32 = arith.constant 256 : i32
                        %c31_i16 = arith.constant 31 : i16
                        %c-128_i8 = arith.constant -128 : i8
                        %c255_i16 = arith.constant 255 : i16
                        %275 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                        %276 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                        %277 = ascvec.duplicate_scalar %c255_i16, %275 : i16 -> <ui16>
                        %278 = ascvec.duplicate_scalar %c-128_i8, %276 : i8 -> <ui8>
                        %279 = ascvec.duplicate_scalar %c31_i16, %275 : i16 -> <ui16>
                        scf.for %arg14 = %arg12 to %arg13 step %c1_i32_10  : i32 {
                          %280 = arith.muli %arg14, %c256_i32 : i32
                          %281 = arith.index_cast %280 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg8[%281] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                          %282 = arith.muli %arg14, %c128_i32 : i32
                          %283 = arith.index_cast %282 : i32 to index
                          %284 = ascvec.loadalign %arg9[%283] : memref<?xui8, 1>, index -> <ui8>
                          %285 = ascvec.unpack %284 {part = "lower"} : <ui8> -> <ui16>
                          %286 = ascvec.and %285, %279, %275 : <ui16>, <ui16> -> <ui16>
                          %287 = ascvec.shr %285, 3, %275 : <ui16> -> <ui16>
                          %288 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                          %289 = ascvec.add %288, %286, %275 : <ui16>
                          %290 = ascvec.shr %289, 5, %275 : <ui16> -> <ui16>
                          %291 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                          %292 = ascvec.add %291, %287, %275 : <ui16>
                          %293 = ascvec.shr %292, 5, %275 : <ui16> -> <ui16>
                          %294 = ascvec.and %290, %277, %275 : <ui16>, <ui16> -> <ui16>
                          %295 = ascvec.shl %293, 8, %275 : <ui16> -> <ui16>
                          %296 = ascvec.or %294, %295, %275 : <ui16>, <ui16> -> <ui16>
                          %297 = ascvec.reinterpret %296 : <ui16> -> <ui8>
                          %298 = ascvec.add_scalar %297, %c64_i8, %276 : <ui8>, i8
                          %299 = ascvec.min_scalar %298, %c123_i8, %276 : <ui8>, i8
                          %300 = arith.muli %arg14, %c512_i32 : i32
                          %301 = arith.index_cast %300 : i32 to index
                          %result0_11, %result1_12 = ascvec.loadalign_deintlv %arg10[%301] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                          %302 = ascvec.and %result1_12, %278, %276 : <ui8>, <ui8> -> <ui8>
                          %303 = ascvec.or %299, %302, %276 : <ui8>, <ui8> -> <ui8>
                          ascvec.storealign %arg11[%281], %303, %276 : memref<32768xui8, 1>, index, <ui8>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
                    } else {
                      %271 = arith.cmpi sge, %269, %c13568_i64 : i64
                      scf.if %271 {
                        %272 = arith.trunci %arg7 : i64 to i32
                        %273 = arith.trunci %262 : i64 to i32
                        ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %274 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                        %275 = ascscalar.bitcast_memref %107 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%107, %274, %275, %alloc_5, %272, %273 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32, %arg13: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_10 = arith.constant 1 : i32
                          %c123_i8 = arith.constant 123 : i8
                          %c64_i8 = arith.constant 64 : i8
                          %c448_i16 = arith.constant 448 : i16
                          %c14464_i16 = arith.constant 14464 : i16
                          %c128_i32 = arith.constant 128 : i32
                          %c256_i32 = arith.constant 256 : i32
                          %cst = arith.constant 2.684350e+08 : bf16
                          %c32767_i16 = arith.constant 32767 : i16
                          %c4095_i16 = arith.constant 4095 : i16
                          %c3_i8 = arith.constant 3 : i8
                          %c1_i8 = arith.constant 1 : i8
                          %c0_i8 = arith.constant 0 : i8
                          %c-128_i8 = arith.constant -128 : i8
                          %c255_i16 = arith.constant 255 : i16
                          %276 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %277 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %278 = ascvec.duplicate_scalar %c255_i16, %276 : i16 -> <ui16>
                          %279 = ascvec.duplicate_scalar %c-128_i8, %277 : i8 -> <ui8>
                          %280 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %281 = ascvec.shr %280, 1, %277 : <ui8> -> <ui8>
                          %282 = ascvec.shr %281, 2, %277 : <ui8> -> <ui8>
                          %283 = ascvec.shl %282, 2, %277 : <ui8> -> <ui8>
                          %284 = ascvec.duplicate_scalar %c1_i8, %277 : i8 -> <ui8>
                          %285 = ascvec.and %280, %284, %277 : <ui8>, <ui8> -> <ui8>
                          %286 = ascvec.add_scalar %281, %c3_i8, %277 : <ui8>, i8
                          %287 = ascvec.add %286, %285, %277 : <ui8>
                          %288 = ascvec.duplicate_scalar %c3_i8, %277 : i8 -> <ui8>
                          %289 = ascvec.and %287, %288, %277 : <ui8>, <ui8> -> <ui8>
                          %290 = ascvec.or %283, %289, %277 : <ui8>, <ui8> -> <ui8>
                          %291 = ascvec.duplicate_scalar %c4095_i16, %276 : i16 -> <ui16>
                          %292 = ascvec.duplicate_scalar %c32767_i16, %276 : i16 -> <ui16>
                          %293 = ascvec.duplicate_scalar %cst, %276 : bf16 -> <bf16>
                          scf.for %arg14 = %arg12 to %arg13 step %c1_i32_10  : i32 {
                            %294 = arith.muli %arg14, %c256_i32 : i32
                            %295 = arith.index_cast %294 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg8[%295] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %296 = arith.muli %arg14, %c128_i32 : i32
                            %297 = arith.index_cast %296 : i32 to index
                            %298 = ascvec.loadalign %arg9[%297] : memref<?xui8, 1>, index -> <ui8>
                            %299 = ascvec.gather_reg %298, %290 : <ui8>, <ui8> -> <ui8>
                            %300 = ascvec.reinterpret %299 : <ui8> -> <ui16>
                            %301 = ascvec.shr %300, 1, %276 : <ui16> -> <ui16>
                            %302 = ascvec.and %301, %291, %276 : <ui16>, <ui16> -> <ui16>
                            %303 = ascvec.shr %300, 4, %276 : <ui16> -> <ui16>
                            %304 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %305 = ascvec.and %304, %292, %276 : <ui16>, <ui16> -> <ui16>
                            %306 = ascvec.cmp_lt_scalar %305, %c14464_i16, %276 : <ui16>, i16 -> !ascvec.mask
                            %307 = ascvec.shr %302, 7, %276 : <ui16> -> <ui16>
                            %308 = ascvec.add %305, %307, %276 : <ui16>
                            %309 = ascvec.shr %308, 5, %276 : <ui16> -> <ui16>
                            %310 = ascvec.reinterpret %305 : <ui16> -> <bf16>
                            %311 = ascvec.mul %310, %293, %306 : <bf16>
                            %312 = ascvec.cast %311, %276 {rounding = 4 : i32} : <bf16> -> <f16>
                            %313 = ascvec.cast %312, %276 {rounding = 4 : i32} : <f16> -> <i16>
                            %314 = ascvec.reinterpret %313 : <i16> -> <ui16>
                            %315 = ascvec.add %314, %302, %276 : <ui16>
                            %316 = ascvec.shr %315, 12, %276 : <ui16> -> <ui16>
                            %317 = ascvec.add_scalar %316, %c448_i16, %276 : <ui16>, i16
                            %318 = ascvec.select %317, %309, %306 : <ui16>
                            %319 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %320 = ascvec.and %319, %292, %276 : <ui16>, <ui16> -> <ui16>
                            %321 = ascvec.cmp_lt_scalar %320, %c14464_i16, %276 : <ui16>, i16 -> !ascvec.mask
                            %322 = ascvec.shr %303, 7, %276 : <ui16> -> <ui16>
                            %323 = ascvec.add %320, %322, %276 : <ui16>
                            %324 = ascvec.shr %323, 5, %276 : <ui16> -> <ui16>
                            %325 = ascvec.reinterpret %320 : <ui16> -> <bf16>
                            %326 = ascvec.mul %325, %293, %321 : <bf16>
                            %327 = ascvec.cast %326, %276 {rounding = 4 : i32} : <bf16> -> <f16>
                            %328 = ascvec.cast %327, %276 {rounding = 4 : i32} : <f16> -> <i16>
                            %329 = ascvec.reinterpret %328 : <i16> -> <ui16>
                            %330 = ascvec.add %329, %303, %276 : <ui16>
                            %331 = ascvec.shr %330, 12, %276 : <ui16> -> <ui16>
                            %332 = ascvec.add_scalar %331, %c448_i16, %276 : <ui16>, i16
                            %333 = ascvec.select %332, %324, %321 : <ui16>
                            %334 = ascvec.and %318, %278, %276 : <ui16>, <ui16> -> <ui16>
                            %335 = ascvec.shl %333, 8, %276 : <ui16> -> <ui16>
                            %336 = ascvec.or %334, %335, %276 : <ui16>, <ui16> -> <ui16>
                            %337 = ascvec.reinterpret %336 : <ui16> -> <ui8>
                            %338 = ascvec.add_scalar %337, %c64_i8, %277 : <ui8>, i8
                            %339 = ascvec.min_scalar %338, %c123_i8, %277 : <ui8>, i8
                            %340 = arith.muli %arg14, %c512_i32 : i32
                            %341 = arith.index_cast %340 : i32 to index
                            %result0_11, %result1_12 = ascvec.loadalign_deintlv %arg10[%341] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %342 = ascvec.and %result1_12, %279, %277 : <ui8>, <ui8> -> <ui8>
                            %343 = ascvec.or %339, %342, %277 : <ui8>, <ui8> -> <ui8>
                            ascvec.storealign %arg11[%295], %343, %277 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
                      } else {
                        %272 = arith.trunci %arg7 : i64 to i32
                        %273 = arith.trunci %262 : i64 to i32
                        ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %274 = ascscalar.bitcast_memref %107 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%107, %alloc_8, %274, %255, %alloc_5, %272, %273 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<8192xi32, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<?xui16, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32, %arg14: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c127_i8 = arith.constant 127 : i8
                          %c16_i32 = arith.constant 16 : i32
                          %c0_i16 = arith.constant 0 : i16
                          %c123_i8 = arith.constant 123 : i8
                          %c-1_i8 = arith.constant -1 : i8
                          %c60_i8 = arith.constant 60 : i8
                          %cst = arith.constant 6.10351526E-5 : f32
                          %c3_i32 = arith.constant 3 : i32
                          %c65535_i32 = arith.constant 65535 : i32
                          %c4_i32_10 = arith.constant 4 : i32
                          %c2_i32_11 = arith.constant 2 : i32
                          %c1_i32_12 = arith.constant 1 : i32
                          %c7_i32 = arith.constant 7 : i32
                          %c32_i32 = arith.constant 32 : i32
                          %c0_i32_13 = arith.constant 0 : i32
                          %c32767_i16 = arith.constant 32767 : i16
                          %c0_14 = arith.constant 0 : index
                          %c256_i32 = arith.constant 256 : i32
                          %c3_i8 = arith.constant 3 : i8
                          %c-128_i8 = arith.constant -128 : i8
                          %c0_i8 = arith.constant 0 : i8
                          %c8_i32 = arith.constant 8 : i32
                          %275 = ascvec.create_mask : !ascvec.mask
                          %276 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %277 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                          %278 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %279 = ascvec.duplicate_scalar %c-128_i8, %277 : i8 -> <ui8>
                          %280 = ascvec.shr %278, 5, %277 : <ui8> -> <ui8>
                          %281 = ascvec.duplicate_scalar %c3_i8, %277 : i8 -> <ui8>
                          %282 = ascvec.and %278, %281, %277 : <ui8>, <ui8> -> <ui8>
                          %283 = ascvec.shl %282, 6, %277 : <ui8> -> <ui8>
                          %284 = ascvec.shr %278, 2, %277 : <ui8> -> <ui8>
                          %285 = ascvec.or %283, %284, %277 : <ui8>, <ui8> -> <ui8>
                          scf.for %arg15 = %arg13 to %arg14 step %c1_i32_12  : i32 {
                            %286 = arith.muli %arg15, %c256_i32 : i32
                            %287 = arith.index_cast %286 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg8[%287] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %288 = ascvec.duplicate_scalar %c32767_i16, %276 : i16 -> <ui16>
                            %289 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %290 = ascvec.and %289, %288, %276 : <ui16>, <ui16> -> <ui16>
                            %291 = ascvec.reinterpret %290 : <ui16> -> <bf16>
                            %292 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %293 = ascvec.and %292, %288, %276 : <ui16>, <ui16> -> <ui16>
                            %294 = ascvec.reinterpret %293 : <ui16> -> <bf16>
                            %295 = ascvec.cast %291, %276 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %296 = ascvec.cast %291, %276 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %297 = ascvec.cast %294, %276 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %298 = ascvec.cast %294, %276 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %299 = ascvec.arange %c0_i32_13 : i32 -> <ui32>
                            %300 = ascvec.shl %299, 2, %275 : <ui32> -> <ui32>
                            %301 = ascvec.add_scalar %300, %c0_i32_13, %275 : <ui32>, i32
                            %302 = ascvec.shr %301, 3, %275 : <ui32> -> <ui32>
                            %303 = arith.muli %arg15, %c32_i32 : i32
                            %304 = ascvec.add_scalar %302, %303, %275 : <ui32>, i32
                            %305 = ascvec.gather %arg9[%c0_14], %304, %275 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %306 = ascvec.reinterpret %305 : <i32> -> <ui32>
                            %307 = ascvec.duplicate_scalar %c7_i32, %275 : i32 -> <ui32>
                            %308 = ascvec.and %301, %307, %275 : <ui32>, <ui32> -> <ui32>
                            %309 = ascvec.shl %306, 11, %275 : <ui32> -> <ui32>
                            %310 = ascvec.shr %306, 21, %275 : <ui32> -> <ui32>
                            %311 = ascvec.or %309, %310, %275 : <ui32>, <ui32> -> <ui32>
                            %312 = ascvec.shr %311, 3, %275 : <ui32> -> <ui32>
                            %313 = ascvec.shl %311, 29, %275 : <ui32> -> <ui32>
                            %314 = ascvec.or %312, %313, %275 : <ui32>, <ui32> -> <ui32>
                            %315 = ascvec.duplicate_scalar %c1_i32_12, %275 : i32 -> <ui32>
                            %316 = ascvec.and %308, %315, %275 : <ui32>, <ui32> -> <ui32>
                            %317 = ascvec.cmp_ne_scalar %316, %c0_i32_13, %275 : <ui32>, i32 -> !ascvec.mask
                            %318 = ascvec.select %314, %311, %317 : <ui32>
                            %319 = ascvec.shr %318, 8, %275 : <ui32> -> <ui32>
                            %320 = ascvec.shl %318, 24, %275 : <ui32> -> <ui32>
                            %321 = ascvec.or %319, %320, %275 : <ui32>, <ui32> -> <ui32>
                            %322 = ascvec.duplicate_scalar %c2_i32_11, %275 : i32 -> <ui32>
                            %323 = ascvec.and %308, %322, %275 : <ui32>, <ui32> -> <ui32>
                            %324 = ascvec.cmp_ne_scalar %323, %c0_i32_13, %275 : <ui32>, i32 -> !ascvec.mask
                            %325 = ascvec.select %321, %318, %324 : <ui32>
                            %326 = ascvec.shr %325, 16, %275 : <ui32> -> <ui32>
                            %327 = ascvec.shl %325, 16, %275 : <ui32> -> <ui32>
                            %328 = ascvec.or %326, %327, %275 : <ui32>, <ui32> -> <ui32>
                            %329 = ascvec.duplicate_scalar %c4_i32_10, %275 : i32 -> <ui32>
                            %330 = ascvec.and %308, %329, %275 : <ui32>, <ui32> -> <ui32>
                            %331 = ascvec.cmp_ne_scalar %330, %c0_i32_13, %275 : <ui32>, i32 -> !ascvec.mask
                            %332 = ascvec.select %328, %325, %331 : <ui32>
                            %333 = ascvec.duplicate_scalar %c65535_i32, %275 : i32 -> <ui32>
                            %334 = ascvec.and %332, %333, %275 : <ui32>, <ui32> -> <ui32>
                            %335 = ascvec.add_scalar %300, %c1_i32_12, %275 : <ui32>, i32
                            %336 = ascvec.shr %335, 3, %275 : <ui32> -> <ui32>
                            %337 = ascvec.add_scalar %336, %303, %275 : <ui32>, i32
                            %338 = ascvec.gather %arg9[%c0_14], %337, %275 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %339 = ascvec.reinterpret %338 : <i32> -> <ui32>
                            %340 = ascvec.and %335, %307, %275 : <ui32>, <ui32> -> <ui32>
                            %341 = ascvec.shl %339, 11, %275 : <ui32> -> <ui32>
                            %342 = ascvec.shr %339, 21, %275 : <ui32> -> <ui32>
                            %343 = ascvec.or %341, %342, %275 : <ui32>, <ui32> -> <ui32>
                            %344 = ascvec.shr %343, 3, %275 : <ui32> -> <ui32>
                            %345 = ascvec.shl %343, 29, %275 : <ui32> -> <ui32>
                            %346 = ascvec.or %344, %345, %275 : <ui32>, <ui32> -> <ui32>
                            %347 = ascvec.and %340, %315, %275 : <ui32>, <ui32> -> <ui32>
                            %348 = ascvec.cmp_ne_scalar %347, %c0_i32_13, %275 : <ui32>, i32 -> !ascvec.mask
                            %349 = ascvec.select %346, %343, %348 : <ui32>
                            %350 = ascvec.shr %349, 8, %275 : <ui32> -> <ui32>
                            %351 = ascvec.shl %349, 24, %275 : <ui32> -> <ui32>
                            %352 = ascvec.or %350, %351, %275 : <ui32>, <ui32> -> <ui32>
                            %353 = ascvec.and %340, %322, %275 : <ui32>, <ui32> -> <ui32>
                            %354 = ascvec.cmp_ne_scalar %353, %c0_i32_13, %275 : <ui32>, i32 -> !ascvec.mask
                            %355 = ascvec.select %352, %349, %354 : <ui32>
                            %356 = ascvec.shr %355, 16, %275 : <ui32> -> <ui32>
                            %357 = ascvec.shl %355, 16, %275 : <ui32> -> <ui32>
                            %358 = ascvec.or %356, %357, %275 : <ui32>, <ui32> -> <ui32>
                            %359 = ascvec.and %340, %329, %275 : <ui32>, <ui32> -> <ui32>
                            %360 = ascvec.cmp_ne_scalar %359, %c0_i32_13, %275 : <ui32>, i32 -> !ascvec.mask
                            %361 = ascvec.select %358, %355, %360 : <ui32>
                            %362 = ascvec.and %361, %333, %275 : <ui32>, <ui32> -> <ui32>
                            %363 = ascvec.add_scalar %300, %c2_i32_11, %275 : <ui32>, i32
                            %364 = ascvec.shr %363, 3, %275 : <ui32> -> <ui32>
                            %365 = ascvec.add_scalar %364, %303, %275 : <ui32>, i32
                            %366 = ascvec.gather %arg9[%c0_14], %365, %275 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %367 = ascvec.reinterpret %366 : <i32> -> <ui32>
                            %368 = ascvec.and %363, %307, %275 : <ui32>, <ui32> -> <ui32>
                            %369 = ascvec.shl %367, 11, %275 : <ui32> -> <ui32>
                            %370 = ascvec.shr %367, 21, %275 : <ui32> -> <ui32>
                            %371 = ascvec.or %369, %370, %275 : <ui32>, <ui32> -> <ui32>
                            %372 = ascvec.shr %371, 3, %275 : <ui32> -> <ui32>
                            %373 = ascvec.shl %371, 29, %275 : <ui32> -> <ui32>
                            %374 = ascvec.or %372, %373, %275 : <ui32>, <ui32> -> <ui32>
                            %375 = ascvec.and %368, %315, %275 : <ui32>, <ui32> -> <ui32>
                            %376 = ascvec.cmp_ne_scalar %375, %c0_i32_13, %275 : <ui32>, i32 -> !ascvec.mask
                            %377 = ascvec.select %374, %371, %376 : <ui32>
                            %378 = ascvec.shr %377, 8, %275 : <ui32> -> <ui32>
                            %379 = ascvec.shl %377, 24, %275 : <ui32> -> <ui32>
                            %380 = ascvec.or %378, %379, %275 : <ui32>, <ui32> -> <ui32>
                            %381 = ascvec.and %368, %322, %275 : <ui32>, <ui32> -> <ui32>
                            %382 = ascvec.cmp_ne_scalar %381, %c0_i32_13, %275 : <ui32>, i32 -> !ascvec.mask
                            %383 = ascvec.select %380, %377, %382 : <ui32>
                            %384 = ascvec.shr %383, 16, %275 : <ui32> -> <ui32>
                            %385 = ascvec.shl %383, 16, %275 : <ui32> -> <ui32>
                            %386 = ascvec.or %384, %385, %275 : <ui32>, <ui32> -> <ui32>
                            %387 = ascvec.and %368, %329, %275 : <ui32>, <ui32> -> <ui32>
                            %388 = ascvec.cmp_ne_scalar %387, %c0_i32_13, %275 : <ui32>, i32 -> !ascvec.mask
                            %389 = ascvec.select %386, %383, %388 : <ui32>
                            %390 = ascvec.and %389, %333, %275 : <ui32>, <ui32> -> <ui32>
                            %391 = ascvec.add_scalar %300, %c3_i32, %275 : <ui32>, i32
                            %392 = ascvec.shr %391, 3, %275 : <ui32> -> <ui32>
                            %393 = ascvec.add_scalar %392, %303, %275 : <ui32>, i32
                            %394 = ascvec.gather %arg9[%c0_14], %393, %275 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %395 = ascvec.reinterpret %394 : <i32> -> <ui32>
                            %396 = ascvec.and %391, %307, %275 : <ui32>, <ui32> -> <ui32>
                            %397 = ascvec.shl %395, 11, %275 : <ui32> -> <ui32>
                            %398 = ascvec.shr %395, 21, %275 : <ui32> -> <ui32>
                            %399 = ascvec.or %397, %398, %275 : <ui32>, <ui32> -> <ui32>
                            %400 = ascvec.shr %399, 3, %275 : <ui32> -> <ui32>
                            %401 = ascvec.shl %399, 29, %275 : <ui32> -> <ui32>
                            %402 = ascvec.or %400, %401, %275 : <ui32>, <ui32> -> <ui32>
                            %403 = ascvec.and %396, %315, %275 : <ui32>, <ui32> -> <ui32>
                            %404 = ascvec.cmp_ne_scalar %403, %c0_i32_13, %275 : <ui32>, i32 -> !ascvec.mask
                            %405 = ascvec.select %402, %399, %404 : <ui32>
                            %406 = ascvec.shr %405, 8, %275 : <ui32> -> <ui32>
                            %407 = ascvec.shl %405, 24, %275 : <ui32> -> <ui32>
                            %408 = ascvec.or %406, %407, %275 : <ui32>, <ui32> -> <ui32>
                            %409 = ascvec.and %396, %322, %275 : <ui32>, <ui32> -> <ui32>
                            %410 = ascvec.cmp_ne_scalar %409, %c0_i32_13, %275 : <ui32>, i32 -> !ascvec.mask
                            %411 = ascvec.select %408, %405, %410 : <ui32>
                            %412 = ascvec.shr %411, 16, %275 : <ui32> -> <ui32>
                            %413 = ascvec.shl %411, 16, %275 : <ui32> -> <ui32>
                            %414 = ascvec.or %412, %413, %275 : <ui32>, <ui32> -> <ui32>
                            %415 = ascvec.and %396, %329, %275 : <ui32>, <ui32> -> <ui32>
                            %416 = ascvec.cmp_ne_scalar %415, %c0_i32_13, %275 : <ui32>, i32 -> !ascvec.mask
                            %417 = ascvec.select %414, %411, %416 : <ui32>
                            %418 = ascvec.and %417, %333, %275 : <ui32>, <ui32> -> <ui32>
                            %419 = ascvec.reinterpret %295 : <f32> -> <ui32>
                            %420 = ascvec.reinterpret %419 : <ui32> -> <f32>
                            %421 = ascvec.max_scalar %420, %cst, %275 : <f32>, f32
                            %422 = ascvec.add %420, %421, %275 : <f32>
                            %423 = ascvec.reinterpret %422 : <f32> -> <ui32>
                            %424 = ascvec.shr %423, 5, %275 : <ui32> -> <ui32>
                            %425 = ascvec.add %424, %334, %275 : <ui32>
                            %426 = ascvec.reinterpret %297 : <f32> -> <ui32>
                            %427 = ascvec.reinterpret %426 : <ui32> -> <f32>
                            %428 = ascvec.max_scalar %427, %cst, %275 : <f32>, f32
                            %429 = ascvec.add %427, %428, %275 : <f32>
                            %430 = ascvec.reinterpret %429 : <f32> -> <ui32>
                            %431 = ascvec.shr %430, 5, %275 : <ui32> -> <ui32>
                            %432 = ascvec.add %431, %362, %275 : <ui32>
                            %433 = ascvec.reinterpret %296 : <f32> -> <ui32>
                            %434 = ascvec.reinterpret %433 : <ui32> -> <f32>
                            %435 = ascvec.max_scalar %434, %cst, %275 : <f32>, f32
                            %436 = ascvec.add %434, %435, %275 : <f32>
                            %437 = ascvec.reinterpret %436 : <f32> -> <ui32>
                            %438 = ascvec.shr %437, 5, %275 : <ui32> -> <ui32>
                            %439 = ascvec.add %438, %390, %275 : <ui32>
                            %440 = ascvec.reinterpret %298 : <f32> -> <ui32>
                            %441 = ascvec.reinterpret %440 : <ui32> -> <f32>
                            %442 = ascvec.max_scalar %441, %cst, %275 : <f32>, f32
                            %443 = ascvec.add %441, %442, %275 : <f32>
                            %444 = ascvec.reinterpret %443 : <f32> -> <ui32>
                            %445 = ascvec.shr %444, 5, %275 : <ui32> -> <ui32>
                            %446 = ascvec.add %445, %418, %275 : <ui32>
                            %447 = ascvec.reinterpret %425 : <ui32> -> <ui16>
                            %448 = ascvec.reinterpret %432 : <ui32> -> <ui16>
                            %result0_15, %result1_16 = ascvec.deintlv %447, %448 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %449 = ascvec.reinterpret %439 : <ui32> -> <ui16>
                            %450 = ascvec.reinterpret %446 : <ui32> -> <ui16>
                            %result0_17, %result1_18 = ascvec.deintlv %449, %450 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %451 = ascvec.reinterpret %result1_16 : <ui16> -> <ui8>
                            %452 = ascvec.reinterpret %result1_18 : <ui16> -> <ui8>
                            %result0_19, %result1_20 = ascvec.deintlv %451, %452 : <ui8>, <ui8> -> <ui8>, <ui8>
                            %453 = ascvec.gather_reg %result0_19, %285 : <ui8>, <ui8> -> <ui8>
                            %454 = ascvec.add_scalar %453, %c60_i8, %277 : <ui8>, i8
                            %455 = ascvec.cmp_ne_scalar %454, %c-1_i8, %277 : <ui8>, i8 -> !ascvec.mask
                            %456 = ascvec.min_scalar %454, %c123_i8, %455 : <ui8>, i8
                            %457 = arith.muli %arg15, %c512_i32 : i32
                            %458 = arith.index_cast %457 : i32 to index
                            %result0_21, %result1_22 = ascvec.loadalign_deintlv %arg10[%458] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %459 = ascvec.and %result1_22, %279, %277 : <ui8>, <ui8> -> <ui8>
                            %460 = ascvec.or %456, %459, %277 : <ui8>, <ui8> -> <ui8>
                            %461 = ascvec.arange %c0_i16 : i16 -> <ui16>
                            %462 = arith.muli %arg15, %c16_i32 : i32
                            %463 = arith.trunci %462 : i32 to i16
                            %464 = ascvec.add_scalar %461, %463, %276 : <ui16>, i16
                            %465 = ascvec.gather %arg11[%c0_14], %464, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                            %466 = ascvec.pack %465 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                            %467 = ascvec.gather_reg %466, %280 : <ui8>, <ui8> -> <ui8>
                            %468 = ascvec.cmp_eq_scalar %467, %c-1_i8, %277 : <ui8>, i8 -> !ascvec.mask
                            %469 = ascvec.duplicate_scalar %c127_i8, %468 merge %460 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                            ascvec.storealign %arg12[%287], %469, %277 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
                      }
                    }
                  }
                }
              } else {
                %203 = arith.divsi %66, %c256_i64 : i64
                %204 = arith.addi %66, %c1023_i64 : i64
                %205 = arith.divsi %204, %c1024_i64 : i64
                %206 = arith.trunci %1 : i64 to i32
                %207 = arith.extui %206 : i32 to i64
                %208 = arith.subi %1, %207 : i64
                %209 = arith.remsi %208, %c4294967296_i64 : i64
                %210 = arith.cmpi ne, %209, %c0_i64 : i64
                %211 = arith.cmpi slt, %209, %c0_i64 : i64
                %212 = arith.andi %210, %211 : i1
                %213 = arith.divsi %208, %c4294967296_i64 : i64
                %214 = arith.extui %212 : i1 to i64
                %215 = arith.subi %213, %214 : i64
                %216 = arith.trunci %215 : i64 to i32
                %217 = arith.divui %108, %c32_i64 : i64
                %218 = arith.trunci %217 : i64 to i32
                %219 = arith.extui %218 : i32 to i64
                %220 = arith.subi %217, %219 : i64
                %221 = arith.remsi %220, %c4294967296_i64 : i64
                %222 = arith.cmpi ne, %221, %c0_i64 : i64
                %223 = arith.cmpi slt, %221, %c0_i64 : i64
                %224 = arith.andi %222, %223 : i1
                %225 = arith.divsi %220, %c4294967296_i64 : i64
                %226 = arith.extui %224 : i1 to i64
                %227 = arith.subi %225, %226 : i64
                %228 = arith.trunci %227 : i64 to i32
                %229 = arith.trunci %3 : i64 to i32
                %230 = arith.extui %229 : i32 to i64
                %231 = arith.subi %3, %230 : i64
                %232 = arith.remsi %231, %c4294967296_i64 : i64
                %233 = arith.cmpi ne, %232, %c0_i64 : i64
                %234 = arith.cmpi slt, %232, %c0_i64 : i64
                %235 = arith.andi %233, %234 : i1
                %236 = arith.divsi %231, %c4294967296_i64 : i64
                %237 = arith.extui %235 : i1 to i64
                %238 = arith.subi %236, %237 : i64
                %239 = arith.trunci %238 : i64 to i32
                %240 = arith.divui %229, %c4_i32 : i32
                %241 = arith.remui %239, %c4_i32 : i32
                %242 = arith.muli %241, %c1073741824_i32 : i32
                %243 = arith.addi %240, %242 : i32
                %244 = arith.divui %239, %c4_i32 : i32
                %245 = arith.muli %205, %c32_i64 : i64
                %246 = arith.divui %217, %c16384_i64 : i64
                %247 = arith.addi %217, %245 : i64
                %248 = arith.subi %247, %c1_i64 : i64
                %249 = arith.divui %248, %c16384_i64 : i64
                %250 = arith.cmpi eq, %246, %249 : i64
                scf.if %250 {
                  %260 = arith.trunci %246 : i64 to i32
                  %261 = arith.extui %260 : i32 to i64
                  %262 = arith.subi %246, %261 : i64
                  %263 = arith.remsi %262, %c4294967296_i64 : i64
                  %264 = arith.cmpi ne, %263, %c0_i64 : i64
                  %265 = arith.cmpi slt, %263, %c0_i64 : i64
                  %266 = arith.andi %264, %265 : i1
                  %267 = arith.divsi %262, %c4294967296_i64 : i64
                  %268 = arith.extui %266 : i1 to i64
                  %269 = arith.subi %267, %268 : i64
                  %270 = arith.trunci %269 : i64 to i32
                  %271 = arith.addi %243, %260 : i32
                  %272 = arith.addi %244, %270 : i32
                  %273 = arith.cmpi ult, %271, %243 : i32
                  %274 = arith.extui %273 : i1 to i32
                  %275 = arith.addi %272, %274 : i32
                  %276 = arith.extui %271 : i32 to i64
                  %277 = arith.muli %276, %c3449720151_i64 : i64
                  %278 = arith.trunci %277 : i64 to i32
                  %279 = arith.extui %278 : i32 to i64
                  %280 = arith.subi %277, %279 : i64
                  %281 = arith.remsi %280, %c4294967296_i64 : i64
                  %282 = arith.cmpi ne, %281, %c0_i64 : i64
                  %283 = arith.cmpi slt, %281, %c0_i64 : i64
                  %284 = arith.andi %282, %283 : i1
                  %285 = arith.divsi %280, %c4294967296_i64 : i64
                  %286 = arith.extui %284 : i1 to i64
                  %287 = arith.subi %285, %286 : i64
                  %288 = arith.trunci %287 : i64 to i32
                  %289 = arith.xori %275, %216 : i32
                  %290 = arith.extui %289 : i32 to i64
                  %291 = arith.muli %290, %c3449720151_i64 : i64
                  %292 = arith.trunci %291 : i64 to i32
                  %293 = arith.extui %292 : i32 to i64
                  %294 = arith.subi %291, %293 : i64
                  %295 = arith.remsi %294, %c4294967296_i64 : i64
                  %296 = arith.cmpi ne, %295, %c0_i64 : i64
                  %297 = arith.cmpi slt, %295, %c0_i64 : i64
                  %298 = arith.andi %296, %297 : i1
                  %299 = arith.divsi %294, %c4294967296_i64 : i64
                  %300 = arith.extui %298 : i1 to i64
                  %301 = arith.subi %299, %300 : i64
                  %302 = arith.trunci %301 : i64 to i32
                  %303 = arith.addi %206, %c-1640531527_i32 : i32
                  %304 = arith.addi %216, %c-1150833019_i32 : i32
                  %305 = arith.xori %302, %278 : i32
                  %306 = arith.xori %305, %303 : i32
                  %307 = arith.extui %306 : i32 to i64
                  %308 = arith.muli %307, %c3528531795_i64 : i64
                  %309 = arith.trunci %308 : i64 to i32
                  %310 = arith.extui %309 : i32 to i64
                  %311 = arith.subi %308, %310 : i64
                  %312 = arith.remsi %311, %c4294967296_i64 : i64
                  %313 = arith.cmpi ne, %312, %c0_i64 : i64
                  %314 = arith.cmpi slt, %312, %c0_i64 : i64
                  %315 = arith.andi %313, %314 : i1
                  %316 = arith.divsi %311, %c4294967296_i64 : i64
                  %317 = arith.extui %315 : i1 to i64
                  %318 = arith.subi %316, %317 : i64
                  %319 = arith.trunci %318 : i64 to i32
                  %320 = arith.xori %288, %206 : i32
                  %321 = arith.addi %206, %c1013904242_i32 : i32
                  %322 = arith.xori %292, %321 : i32
                  %323 = arith.addi %216, %c1993301258_i32 : i32
                  %324 = arith.xori %319, %323 : i32
                  ascsimt.vf(%218, %320, %304, %322, %324, %206, %216, %309, %alloc_8, %245 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
                    %c256_i64_10 = arith.constant 256 : i64
                    %c3_i64_11 = arith.constant 3 : i64
                    %c2_i64_12 = arith.constant 2 : i64
                    %c1_i64_13 = arith.constant 1 : i64
                    %c4_i64_14 = arith.constant 4 : i64
                    %c1_i32_15 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c3_i32 = arith.constant 3 : i32
                    %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %325 = ascsimt.thread_idx {ascsimt.scope = 8 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %326 = arith.extsi %325 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg17 = %326 to %arg16 step %c256_i64_10  : i64 {
                      %327 = arith.trunci %arg17 : i64 to i32
                      %328 = arith.addi %arg7, %327 : i32
                      %329 = arith.remui %328, %c16384_i32 : i32
                      %330 = arith.xori %329, %arg8 : i32
                      %331 = arith.muli %330, %c-766435501_i32 : i32
                      %332 = "ascsimt.bitcast"(%330) : (i32) -> ui32
                      %333 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %334 = ascsimt.math_binary %332, %333, "mulhi" : ui32, ui32 -> ui32
                      %335 = "ascsimt.bitcast"(%334) : (ui32) -> i32
                      %336 = arith.xori %335, %arg9 : i32
                      %337 = arith.muli %336, %c-845247145_i32 : i32
                      %338 = "ascsimt.bitcast"(%336) : (i32) -> ui32
                      %339 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %340 = ascsimt.math_binary %338, %339, "mulhi" : ui32, ui32 -> ui32
                      %341 = "ascsimt.bitcast"(%340) : (ui32) -> i32
                      %342 = arith.xori %341, %arg10 : i32
                      %343 = arith.xori %331, %arg11 : i32
                      %344:4 = scf.for %arg18 = %c3_i32 to %c10_i32 step %c1_i32_15 iter_args(%arg19 = %342, %arg20 = %337, %arg21 = %343, %arg22 = %arg14) -> (i32, i32, i32, i32)  : i32 {
                        %353 = arith.muli %arg19, %c-766435501_i32 : i32
                        %354 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %355 = ascsimt.math_binary %354, %333, "mulhi" : ui32, ui32 -> ui32
                        %356 = "ascsimt.bitcast"(%355) : (ui32) -> i32
                        %357 = arith.muli %arg21, %c-845247145_i32 : i32
                        %358 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %359 = ascsimt.math_binary %358, %339, "mulhi" : ui32, ui32 -> ui32
                        %360 = "ascsimt.bitcast"(%359) : (ui32) -> i32
                        %361 = arith.muli %arg18, %c-1640531527_i32_17 : i32
                        %362 = arith.addi %arg12, %361 : i32
                        %363 = arith.muli %arg18, %c-1150833019_i32_16 : i32
                        %364 = arith.addi %arg13, %363 : i32
                        %365 = arith.xori %360, %arg20 : i32
                        %366 = arith.xori %365, %362 : i32
                        %367 = arith.xori %356, %arg22 : i32
                        %368 = arith.xori %367, %364 : i32
                        scf.yield %366, %357, %368, %353 : i32, i32, i32, i32
                      }
                      %345 = arith.muli %arg17, %c4_i64_14 : i64
                      %346 = arith.index_cast %345 : i64 to index
                      ascsimt.store %arg15[%346], %344#0 : memref<8192xi32, 1>, i32
                      %347 = arith.addi %345, %c1_i64_13 : i64
                      %348 = arith.index_cast %347 : i64 to index
                      ascsimt.store %arg15[%348], %344#1 : memref<8192xi32, 1>, i32
                      %349 = arith.addi %345, %c2_i64_12 : i64
                      %350 = arith.index_cast %349 : i64 to index
                      ascsimt.store %arg15[%350], %344#2 : memref<8192xi32, 1>, i32
                      %351 = arith.addi %345, %c3_i64_11 : i64
                      %352 = arith.index_cast %351 : i64 to index
                      ascsimt.store %arg15[%352], %344#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%218, %228, %243, %244, %206, %216, %alloc_8, %245 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: memref<8192xi32, 1>, %arg14: i64):
                    %c256_i64_10 = arith.constant 256 : i64
                    %c3_i64_11 = arith.constant 3 : i64
                    %c2_i64_12 = arith.constant 2 : i64
                    %c1_i64_13 = arith.constant 1 : i64
                    %c4_i64_14 = arith.constant 4 : i64
                    %c1_i32_15 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c0_i32_16 = arith.constant 0 : i32
                    %c-1150833019_i32_17 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_18 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c262144_i32 = arith.constant 262144 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %260 = ascsimt.thread_idx {ascsimt.scope = 9 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %261 = arith.extsi %260 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg15 = %261 to %arg14 step %c256_i64_10  : i64 {
                      %262 = arith.trunci %arg15 : i64 to i32
                      %263 = arith.addi %arg7, %262 : i32
                      %264 = arith.cmpi ult, %263, %arg7 : i32
                      %265 = arith.extui %264 : i1 to i32
                      %266 = arith.addi %arg8, %265 : i32
                      %267 = arith.divui %263, %c16384_i32 : i32
                      %268 = arith.remui %266, %c16384_i32 : i32
                      %269 = arith.muli %268, %c262144_i32 : i32
                      %270 = arith.addi %267, %269 : i32
                      %271 = arith.divui %266, %c16384_i32 : i32
                      %272 = arith.addi %arg9, %270 : i32
                      %273 = arith.addi %arg10, %271 : i32
                      %274 = arith.cmpi ult, %272, %arg9 : i32
                      %275 = arith.extui %274 : i1 to i32
                      %276 = arith.addi %273, %275 : i32
                      %277 = arith.remui %263, %c16384_i32 : i32
                      %278:4 = scf.for %arg16 = %c0_i32_16 to %c10_i32 step %c1_i32_15 iter_args(%arg17 = %c0_i32_16, %arg18 = %277, %arg19 = %272, %arg20 = %276) -> (i32, i32, i32, i32)  : i32 {
                        %287 = arith.muli %arg17, %c-766435501_i32 : i32
                        %288 = "ascsimt.bitcast"(%arg17) : (i32) -> ui32
                        %289 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %290 = ascsimt.math_binary %288, %289, "mulhi" : ui32, ui32 -> ui32
                        %291 = "ascsimt.bitcast"(%290) : (ui32) -> i32
                        %292 = arith.muli %arg19, %c-845247145_i32 : i32
                        %293 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %294 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %295 = ascsimt.math_binary %293, %294, "mulhi" : ui32, ui32 -> ui32
                        %296 = "ascsimt.bitcast"(%295) : (ui32) -> i32
                        %297 = arith.muli %arg16, %c-1640531527_i32_18 : i32
                        %298 = arith.addi %arg11, %297 : i32
                        %299 = arith.muli %arg16, %c-1150833019_i32_17 : i32
                        %300 = arith.addi %arg12, %299 : i32
                        %301 = arith.xori %296, %arg18 : i32
                        %302 = arith.xori %301, %298 : i32
                        %303 = arith.xori %291, %arg20 : i32
                        %304 = arith.xori %303, %300 : i32
                        scf.yield %302, %292, %304, %287 : i32, i32, i32, i32
                      }
                      %279 = arith.muli %arg15, %c4_i64_14 : i64
                      %280 = arith.index_cast %279 : i64 to index
                      ascsimt.store %arg13[%280], %278#0 : memref<8192xi32, 1>, i32
                      %281 = arith.addi %279, %c1_i64_13 : i64
                      %282 = arith.index_cast %281 : i64 to index
                      ascsimt.store %arg13[%282], %278#1 : memref<8192xi32, 1>, i32
                      %283 = arith.addi %279, %c2_i64_12 : i64
                      %284 = arith.index_cast %283 : i64 to index
                      ascsimt.store %arg13[%284], %278#2 : memref<8192xi32, 1>, i32
                      %285 = arith.addi %279, %c3_i64_11 : i64
                      %286 = arith.index_cast %285 : i64 to index
                      ascsimt.store %arg13[%286], %278#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %251 = arith.trunci %48 : i64 to i32
                %252 = arith.divsi %251, %c2_i32 : i32
                %253 = arith.trunci %252 : i32 to i8
                %254 = arith.trunci %203 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %255 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%253, %alloc_6, %107, %255, %254, %alloc_9 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, i32, memref<2048xui16, 1>) {
                ^bb0(%arg7: i8, %arg8: memref<2048xui8, 1>, %arg9: memref<32768xbf16, 1>, %arg10: memref<?xui16, 1>, %arg11: i32, %arg12: memref<2048xui16, 1>):
                  %c1_i32_10 = arith.constant 1 : i32
                  %c0_i32_11 = arith.constant 0 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c0_12 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %260 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %261 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %262 = ascvec.duplicate_scalar %c32767_i16, %260 : i16 -> <ui16>
                  %263 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %264 = ascvec.shr %263, 4, %260 : <ui16> -> <ui16>
                  %265 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %266 = ascvec.duplicate_scalar %c1_i8, %261 : i8 -> <ui8>
                  %267 = ascvec.and %265, %266, %261 : <ui8>, <ui8> -> <ui8>
                  %268 = ascvec.cmp_lt_scalar %265, %c16_i8, %261 : <ui8>, i8 -> !ascvec.mask
                  %269 = ascvec.cmp_lt_scalar %267, %arg7, %261 : <ui8>, i8 -> !ascvec.mask
                  %270 = ascvec.mask_and %268, %269, %261
                  ascvec.store_unalign_begin %arg8 : memref<2048xui8, 1>
                  %271 = ascvec.duplicate_scalar %c-1_i16, %260 : i16 -> <ui16>
                  %272 = ascvec.duplicate_scalar %c0_i16, %260 : i16 -> <ui16>
                  %273:2 = scf.for %arg13 = %c0_i32_11 to %arg11 step %c1_i32_10 iter_args(%arg14 = %272, %arg15 = %271) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i32 {
                    %274 = arith.muli %arg13, %c256_i32 : i32
                    %275 = arith.index_cast %274 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg9[%275] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %276 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %277 = ascvec.and %276, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %278 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %279 = ascvec.and %278, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %280 = ascvec.max %277, %279, %260 : <ui16>
                    %281 = ascvec.reduce_max_datablock %280, %260 : <ui16>
                    %282 = ascvec.shr %281, 7, %260 : <ui16> -> <ui16>
                    %283 = ascvec.cmp_ge_scalar %282, %c255_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %284 = ascvec.max_scalar %282, %c15_i16, %260 : <ui16>, i16
                    %285 = ascvec.add_scalar %284, %c-15_i16, %260 : <ui16>, i16
                    %286 = ascvec.duplicate_scalar %c255_i16, %283 merge %285 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %287 = ascvec.duplicate_scalar %c254_i16, %260 : i16 -> <ui16>
                    %288 = ascvec.sub %287, %286, %260 : <ui16>
                    %289 = ascvec.cmp_ne_scalar %286, %c0_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %290 = ascvec.shl %288, 7, %289 : <ui16> -> <ui16>
                    %291 = ascvec.max %arg14, %286, %mask : <ui16>
                    %292 = ascvec.gather_reg %290, %264 : <ui16>, <ui16> -> <ui16>
                    %293 = ascvec.reinterpret %292 : <ui16> -> <bf16>
                    %294 = ascvec.mul %result0, %293, %260 : <bf16>
                    %295 = ascvec.mul %result1, %293, %260 : <bf16>
                    ascvec.storealign_intlv %arg9[%275], %294, %295 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %296 = ascvec.reinterpret %294 : <bf16> -> <ui16>
                    %297 = ascvec.and %296, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %298 = ascvec.reinterpret %295 : <bf16> -> <ui16>
                    %299 = ascvec.and %298, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %300 = ascvec.min %297, %299, %260 : <ui16>
                    %301 = ascvec.min %arg15, %300, %260 : <ui16>
                    %302 = arith.muli %arg13, %c16_i32 : i32
                    %303 = arith.index_cast %302 : i32 to index
                    ascvec.storealign %arg10[%303], %286, %mask : memref<?xui16, 1>, index, <ui16>
                    %304 = ascvec.reinterpret %286 : <ui16> -> <ui8>
                    %305 = ascvec.squeeze_store_reg %304, %270 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_12], %305 : memref<2048xui8, 1>, <ui8>
                    scf.yield %291, %301 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                  }
                  %c0_13 = arith.constant 0 : index
                  %c1_14 = arith.constant 1 : index
                  scf.for %arg13 = %c0_13 to %c1_14 step %c1_14 {
                    %274 = ascvec.reduce_min %273#1, %260 : <ui16>
                    %275 = ascvec.reduce_max %273#0, %mask : <ui16>
                    %276 = ascvec.cmp_eq_scalar %275, %c255_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %277 = ascvec.duplicate_scalar %c0_i16, %276 merge %274 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.store_unalign_post %arg8[%c0_12] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg12[%c0_12], %277 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %256 = memref.load %alloc_9[%c0] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %257 = cannir.scalar_storage_cast %256 : ui16 -> i16
                %258 = arith.extui %257 : i16 to i64
                %259 = arith.cmpi sge, %258, %c14464_i64 : i64
                scf.if %259 {
                  ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %260 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                  %261 = ascscalar.bitcast_memref %107 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%107, %260, %261, %alloc_5, %254 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg7: memref<32768xbf16, 1>, %arg8: memref<?xui8, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<32768xui8, 1>, %arg11: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_10 = arith.constant 1 : i32
                    %c0_i32_11 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c128_i32 = arith.constant 128 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c31_i16 = arith.constant 31 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %262 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %263 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %264 = ascvec.duplicate_scalar %c255_i16, %262 : i16 -> <ui16>
                    %265 = ascvec.duplicate_scalar %c-128_i8, %263 : i8 -> <ui8>
                    %266 = ascvec.duplicate_scalar %c31_i16, %262 : i16 -> <ui16>
                    scf.for %arg12 = %c0_i32_11 to %arg11 step %c1_i32_10  : i32 {
                      %267 = arith.muli %arg12, %c256_i32 : i32
                      %268 = arith.index_cast %267 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg7[%268] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %269 = arith.muli %arg12, %c128_i32 : i32
                      %270 = arith.index_cast %269 : i32 to index
                      %271 = ascvec.loadalign %arg8[%270] : memref<?xui8, 1>, index -> <ui8>
                      %272 = ascvec.unpack %271 {part = "lower"} : <ui8> -> <ui16>
                      %273 = ascvec.and %272, %266, %262 : <ui16>, <ui16> -> <ui16>
                      %274 = ascvec.shr %272, 3, %262 : <ui16> -> <ui16>
                      %275 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %276 = ascvec.add %275, %273, %262 : <ui16>
                      %277 = ascvec.shr %276, 5, %262 : <ui16> -> <ui16>
                      %278 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %279 = ascvec.add %278, %274, %262 : <ui16>
                      %280 = ascvec.shr %279, 5, %262 : <ui16> -> <ui16>
                      %281 = ascvec.and %277, %264, %262 : <ui16>, <ui16> -> <ui16>
                      %282 = ascvec.shl %280, 8, %262 : <ui16> -> <ui16>
                      %283 = ascvec.or %281, %282, %262 : <ui16>, <ui16> -> <ui16>
                      %284 = ascvec.reinterpret %283 : <ui16> -> <ui8>
                      %285 = ascvec.add_scalar %284, %c64_i8, %263 : <ui8>, i8
                      %286 = ascvec.min_scalar %285, %c123_i8, %263 : <ui8>, i8
                      %287 = arith.muli %arg12, %c512_i32 : i32
                      %288 = arith.index_cast %287 : i32 to index
                      %result0_12, %result1_13 = ascvec.loadalign_deintlv %arg9[%288] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %289 = ascvec.and %result1_13, %265, %263 : <ui8>, <ui8> -> <ui8>
                      %290 = ascvec.or %286, %289, %263 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg10[%268], %290, %263 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
                } else {
                  %260 = arith.cmpi sge, %258, %c13568_i64 : i64
                  scf.if %260 {
                    ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %261 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                    %262 = ascscalar.bitcast_memref %107 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                    ascvec.vf(%107, %261, %262, %alloc_5, %254 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                    ^bb0(%arg7: memref<32768xbf16, 1>, %arg8: memref<?xui8, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<32768xui8, 1>, %arg11: i32):
                      %c512_i32 = arith.constant 512 : i32
                      %c1_i32_10 = arith.constant 1 : i32
                      %c0_i32_11 = arith.constant 0 : i32
                      %c123_i8 = arith.constant 123 : i8
                      %c64_i8 = arith.constant 64 : i8
                      %c448_i16 = arith.constant 448 : i16
                      %c14464_i16 = arith.constant 14464 : i16
                      %c128_i32 = arith.constant 128 : i32
                      %c256_i32 = arith.constant 256 : i32
                      %cst = arith.constant 2.684350e+08 : bf16
                      %c32767_i16 = arith.constant 32767 : i16
                      %c4095_i16 = arith.constant 4095 : i16
                      %c3_i8 = arith.constant 3 : i8
                      %c1_i8 = arith.constant 1 : i8
                      %c0_i8 = arith.constant 0 : i8
                      %c-128_i8 = arith.constant -128 : i8
                      %c255_i16 = arith.constant 255 : i16
                      %263 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                      %264 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                      %265 = ascvec.duplicate_scalar %c255_i16, %263 : i16 -> <ui16>
                      %266 = ascvec.duplicate_scalar %c-128_i8, %264 : i8 -> <ui8>
                      %267 = ascvec.arange %c0_i8 : i8 -> <ui8>
                      %268 = ascvec.shr %267, 1, %264 : <ui8> -> <ui8>
                      %269 = ascvec.shr %268, 2, %264 : <ui8> -> <ui8>
                      %270 = ascvec.shl %269, 2, %264 : <ui8> -> <ui8>
                      %271 = ascvec.duplicate_scalar %c1_i8, %264 : i8 -> <ui8>
                      %272 = ascvec.and %267, %271, %264 : <ui8>, <ui8> -> <ui8>
                      %273 = ascvec.add_scalar %268, %c3_i8, %264 : <ui8>, i8
                      %274 = ascvec.add %273, %272, %264 : <ui8>
                      %275 = ascvec.duplicate_scalar %c3_i8, %264 : i8 -> <ui8>
                      %276 = ascvec.and %274, %275, %264 : <ui8>, <ui8> -> <ui8>
                      %277 = ascvec.or %270, %276, %264 : <ui8>, <ui8> -> <ui8>
                      %278 = ascvec.duplicate_scalar %c4095_i16, %263 : i16 -> <ui16>
                      %279 = ascvec.duplicate_scalar %c32767_i16, %263 : i16 -> <ui16>
                      %280 = ascvec.duplicate_scalar %cst, %263 : bf16 -> <bf16>
                      scf.for %arg12 = %c0_i32_11 to %arg11 step %c1_i32_10  : i32 {
                        %281 = arith.muli %arg12, %c256_i32 : i32
                        %282 = arith.index_cast %281 : i32 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg7[%282] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                        %283 = arith.muli %arg12, %c128_i32 : i32
                        %284 = arith.index_cast %283 : i32 to index
                        %285 = ascvec.loadalign %arg8[%284] : memref<?xui8, 1>, index -> <ui8>
                        %286 = ascvec.gather_reg %285, %277 : <ui8>, <ui8> -> <ui8>
                        %287 = ascvec.reinterpret %286 : <ui8> -> <ui16>
                        %288 = ascvec.shr %287, 1, %263 : <ui16> -> <ui16>
                        %289 = ascvec.and %288, %278, %263 : <ui16>, <ui16> -> <ui16>
                        %290 = ascvec.shr %287, 4, %263 : <ui16> -> <ui16>
                        %291 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                        %292 = ascvec.and %291, %279, %263 : <ui16>, <ui16> -> <ui16>
                        %293 = ascvec.cmp_lt_scalar %292, %c14464_i16, %263 : <ui16>, i16 -> !ascvec.mask
                        %294 = ascvec.shr %289, 7, %263 : <ui16> -> <ui16>
                        %295 = ascvec.add %292, %294, %263 : <ui16>
                        %296 = ascvec.shr %295, 5, %263 : <ui16> -> <ui16>
                        %297 = ascvec.reinterpret %292 : <ui16> -> <bf16>
                        %298 = ascvec.mul %297, %280, %293 : <bf16>
                        %299 = ascvec.cast %298, %263 {rounding = 4 : i32} : <bf16> -> <f16>
                        %300 = ascvec.cast %299, %263 {rounding = 4 : i32} : <f16> -> <i16>
                        %301 = ascvec.reinterpret %300 : <i16> -> <ui16>
                        %302 = ascvec.add %301, %289, %263 : <ui16>
                        %303 = ascvec.shr %302, 12, %263 : <ui16> -> <ui16>
                        %304 = ascvec.add_scalar %303, %c448_i16, %263 : <ui16>, i16
                        %305 = ascvec.select %304, %296, %293 : <ui16>
                        %306 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                        %307 = ascvec.and %306, %279, %263 : <ui16>, <ui16> -> <ui16>
                        %308 = ascvec.cmp_lt_scalar %307, %c14464_i16, %263 : <ui16>, i16 -> !ascvec.mask
                        %309 = ascvec.shr %290, 7, %263 : <ui16> -> <ui16>
                        %310 = ascvec.add %307, %309, %263 : <ui16>
                        %311 = ascvec.shr %310, 5, %263 : <ui16> -> <ui16>
                        %312 = ascvec.reinterpret %307 : <ui16> -> <bf16>
                        %313 = ascvec.mul %312, %280, %308 : <bf16>
                        %314 = ascvec.cast %313, %263 {rounding = 4 : i32} : <bf16> -> <f16>
                        %315 = ascvec.cast %314, %263 {rounding = 4 : i32} : <f16> -> <i16>
                        %316 = ascvec.reinterpret %315 : <i16> -> <ui16>
                        %317 = ascvec.add %316, %290, %263 : <ui16>
                        %318 = ascvec.shr %317, 12, %263 : <ui16> -> <ui16>
                        %319 = ascvec.add_scalar %318, %c448_i16, %263 : <ui16>, i16
                        %320 = ascvec.select %319, %311, %308 : <ui16>
                        %321 = ascvec.and %305, %265, %263 : <ui16>, <ui16> -> <ui16>
                        %322 = ascvec.shl %320, 8, %263 : <ui16> -> <ui16>
                        %323 = ascvec.or %321, %322, %263 : <ui16>, <ui16> -> <ui16>
                        %324 = ascvec.reinterpret %323 : <ui16> -> <ui8>
                        %325 = ascvec.add_scalar %324, %c64_i8, %264 : <ui8>, i8
                        %326 = ascvec.min_scalar %325, %c123_i8, %264 : <ui8>, i8
                        %327 = arith.muli %arg12, %c512_i32 : i32
                        %328 = arith.index_cast %327 : i32 to index
                        %result0_12, %result1_13 = ascvec.loadalign_deintlv %arg9[%328] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                        %329 = ascvec.and %result1_13, %266, %264 : <ui8>, <ui8> -> <ui8>
                        %330 = ascvec.or %326, %329, %264 : <ui8>, <ui8> -> <ui8>
                        ascvec.storealign %arg10[%282], %330, %264 : memref<32768xui8, 1>, index, <ui8>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
                  } else {
                    ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %261 = ascscalar.bitcast_memref %107 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                    ascvec.vf(%107, %alloc_8, %261, %255, %alloc_5, %254 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32) {
                    ^bb0(%arg7: memref<32768xbf16, 1>, %arg8: memref<8192xi32, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<?xui16, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32):
                      %c512_i32 = arith.constant 512 : i32
                      %c127_i8 = arith.constant 127 : i8
                      %c16_i32 = arith.constant 16 : i32
                      %c0_i16 = arith.constant 0 : i16
                      %c123_i8 = arith.constant 123 : i8
                      %c-1_i8 = arith.constant -1 : i8
                      %c60_i8 = arith.constant 60 : i8
                      %cst = arith.constant 6.10351526E-5 : f32
                      %c3_i32 = arith.constant 3 : i32
                      %c65535_i32 = arith.constant 65535 : i32
                      %c4_i32_10 = arith.constant 4 : i32
                      %c2_i32_11 = arith.constant 2 : i32
                      %c1_i32_12 = arith.constant 1 : i32
                      %c7_i32 = arith.constant 7 : i32
                      %c32_i32 = arith.constant 32 : i32
                      %c0_i32_13 = arith.constant 0 : i32
                      %c32767_i16 = arith.constant 32767 : i16
                      %c0_14 = arith.constant 0 : index
                      %c256_i32 = arith.constant 256 : i32
                      %c3_i8 = arith.constant 3 : i8
                      %c-128_i8 = arith.constant -128 : i8
                      %c0_i8 = arith.constant 0 : i8
                      %c8_i32 = arith.constant 8 : i32
                      %262 = ascvec.create_mask : !ascvec.mask
                      %263 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                      %264 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                      %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                      %265 = ascvec.arange %c0_i8 : i8 -> <ui8>
                      %266 = ascvec.duplicate_scalar %c-128_i8, %264 : i8 -> <ui8>
                      %267 = ascvec.shr %265, 5, %264 : <ui8> -> <ui8>
                      %268 = ascvec.duplicate_scalar %c3_i8, %264 : i8 -> <ui8>
                      %269 = ascvec.and %265, %268, %264 : <ui8>, <ui8> -> <ui8>
                      %270 = ascvec.shl %269, 6, %264 : <ui8> -> <ui8>
                      %271 = ascvec.shr %265, 2, %264 : <ui8> -> <ui8>
                      %272 = ascvec.or %270, %271, %264 : <ui8>, <ui8> -> <ui8>
                      scf.for %arg13 = %c0_i32_13 to %arg12 step %c1_i32_12  : i32 {
                        %273 = arith.muli %arg13, %c256_i32 : i32
                        %274 = arith.index_cast %273 : i32 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg7[%274] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                        %275 = ascvec.duplicate_scalar %c32767_i16, %263 : i16 -> <ui16>
                        %276 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                        %277 = ascvec.and %276, %275, %263 : <ui16>, <ui16> -> <ui16>
                        %278 = ascvec.reinterpret %277 : <ui16> -> <bf16>
                        %279 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                        %280 = ascvec.and %279, %275, %263 : <ui16>, <ui16> -> <ui16>
                        %281 = ascvec.reinterpret %280 : <ui16> -> <bf16>
                        %282 = ascvec.cast %278, %263 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %283 = ascvec.cast %278, %263 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %284 = ascvec.cast %281, %263 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %285 = ascvec.cast %281, %263 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %286 = ascvec.arange %c0_i32_13 : i32 -> <ui32>
                        %287 = ascvec.shl %286, 2, %262 : <ui32> -> <ui32>
                        %288 = ascvec.add_scalar %287, %c0_i32_13, %262 : <ui32>, i32
                        %289 = ascvec.shr %288, 3, %262 : <ui32> -> <ui32>
                        %290 = arith.muli %arg13, %c32_i32 : i32
                        %291 = ascvec.add_scalar %289, %290, %262 : <ui32>, i32
                        %292 = ascvec.gather %arg8[%c0_14], %291, %262 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %293 = ascvec.reinterpret %292 : <i32> -> <ui32>
                        %294 = ascvec.duplicate_scalar %c7_i32, %262 : i32 -> <ui32>
                        %295 = ascvec.and %288, %294, %262 : <ui32>, <ui32> -> <ui32>
                        %296 = ascvec.shl %293, 11, %262 : <ui32> -> <ui32>
                        %297 = ascvec.shr %293, 21, %262 : <ui32> -> <ui32>
                        %298 = ascvec.or %296, %297, %262 : <ui32>, <ui32> -> <ui32>
                        %299 = ascvec.shr %298, 3, %262 : <ui32> -> <ui32>
                        %300 = ascvec.shl %298, 29, %262 : <ui32> -> <ui32>
                        %301 = ascvec.or %299, %300, %262 : <ui32>, <ui32> -> <ui32>
                        %302 = ascvec.duplicate_scalar %c1_i32_12, %262 : i32 -> <ui32>
                        %303 = ascvec.and %295, %302, %262 : <ui32>, <ui32> -> <ui32>
                        %304 = ascvec.cmp_ne_scalar %303, %c0_i32_13, %262 : <ui32>, i32 -> !ascvec.mask
                        %305 = ascvec.select %301, %298, %304 : <ui32>
                        %306 = ascvec.shr %305, 8, %262 : <ui32> -> <ui32>
                        %307 = ascvec.shl %305, 24, %262 : <ui32> -> <ui32>
                        %308 = ascvec.or %306, %307, %262 : <ui32>, <ui32> -> <ui32>
                        %309 = ascvec.duplicate_scalar %c2_i32_11, %262 : i32 -> <ui32>
                        %310 = ascvec.and %295, %309, %262 : <ui32>, <ui32> -> <ui32>
                        %311 = ascvec.cmp_ne_scalar %310, %c0_i32_13, %262 : <ui32>, i32 -> !ascvec.mask
                        %312 = ascvec.select %308, %305, %311 : <ui32>
                        %313 = ascvec.shr %312, 16, %262 : <ui32> -> <ui32>
                        %314 = ascvec.shl %312, 16, %262 : <ui32> -> <ui32>
                        %315 = ascvec.or %313, %314, %262 : <ui32>, <ui32> -> <ui32>
                        %316 = ascvec.duplicate_scalar %c4_i32_10, %262 : i32 -> <ui32>
                        %317 = ascvec.and %295, %316, %262 : <ui32>, <ui32> -> <ui32>
                        %318 = ascvec.cmp_ne_scalar %317, %c0_i32_13, %262 : <ui32>, i32 -> !ascvec.mask
                        %319 = ascvec.select %315, %312, %318 : <ui32>
                        %320 = ascvec.duplicate_scalar %c65535_i32, %262 : i32 -> <ui32>
                        %321 = ascvec.and %319, %320, %262 : <ui32>, <ui32> -> <ui32>
                        %322 = ascvec.add_scalar %287, %c1_i32_12, %262 : <ui32>, i32
                        %323 = ascvec.shr %322, 3, %262 : <ui32> -> <ui32>
                        %324 = ascvec.add_scalar %323, %290, %262 : <ui32>, i32
                        %325 = ascvec.gather %arg8[%c0_14], %324, %262 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %326 = ascvec.reinterpret %325 : <i32> -> <ui32>
                        %327 = ascvec.and %322, %294, %262 : <ui32>, <ui32> -> <ui32>
                        %328 = ascvec.shl %326, 11, %262 : <ui32> -> <ui32>
                        %329 = ascvec.shr %326, 21, %262 : <ui32> -> <ui32>
                        %330 = ascvec.or %328, %329, %262 : <ui32>, <ui32> -> <ui32>
                        %331 = ascvec.shr %330, 3, %262 : <ui32> -> <ui32>
                        %332 = ascvec.shl %330, 29, %262 : <ui32> -> <ui32>
                        %333 = ascvec.or %331, %332, %262 : <ui32>, <ui32> -> <ui32>
                        %334 = ascvec.and %327, %302, %262 : <ui32>, <ui32> -> <ui32>
                        %335 = ascvec.cmp_ne_scalar %334, %c0_i32_13, %262 : <ui32>, i32 -> !ascvec.mask
                        %336 = ascvec.select %333, %330, %335 : <ui32>
                        %337 = ascvec.shr %336, 8, %262 : <ui32> -> <ui32>
                        %338 = ascvec.shl %336, 24, %262 : <ui32> -> <ui32>
                        %339 = ascvec.or %337, %338, %262 : <ui32>, <ui32> -> <ui32>
                        %340 = ascvec.and %327, %309, %262 : <ui32>, <ui32> -> <ui32>
                        %341 = ascvec.cmp_ne_scalar %340, %c0_i32_13, %262 : <ui32>, i32 -> !ascvec.mask
                        %342 = ascvec.select %339, %336, %341 : <ui32>
                        %343 = ascvec.shr %342, 16, %262 : <ui32> -> <ui32>
                        %344 = ascvec.shl %342, 16, %262 : <ui32> -> <ui32>
                        %345 = ascvec.or %343, %344, %262 : <ui32>, <ui32> -> <ui32>
                        %346 = ascvec.and %327, %316, %262 : <ui32>, <ui32> -> <ui32>
                        %347 = ascvec.cmp_ne_scalar %346, %c0_i32_13, %262 : <ui32>, i32 -> !ascvec.mask
                        %348 = ascvec.select %345, %342, %347 : <ui32>
                        %349 = ascvec.and %348, %320, %262 : <ui32>, <ui32> -> <ui32>
                        %350 = ascvec.add_scalar %287, %c2_i32_11, %262 : <ui32>, i32
                        %351 = ascvec.shr %350, 3, %262 : <ui32> -> <ui32>
                        %352 = ascvec.add_scalar %351, %290, %262 : <ui32>, i32
                        %353 = ascvec.gather %arg8[%c0_14], %352, %262 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %354 = ascvec.reinterpret %353 : <i32> -> <ui32>
                        %355 = ascvec.and %350, %294, %262 : <ui32>, <ui32> -> <ui32>
                        %356 = ascvec.shl %354, 11, %262 : <ui32> -> <ui32>
                        %357 = ascvec.shr %354, 21, %262 : <ui32> -> <ui32>
                        %358 = ascvec.or %356, %357, %262 : <ui32>, <ui32> -> <ui32>
                        %359 = ascvec.shr %358, 3, %262 : <ui32> -> <ui32>
                        %360 = ascvec.shl %358, 29, %262 : <ui32> -> <ui32>
                        %361 = ascvec.or %359, %360, %262 : <ui32>, <ui32> -> <ui32>
                        %362 = ascvec.and %355, %302, %262 : <ui32>, <ui32> -> <ui32>
                        %363 = ascvec.cmp_ne_scalar %362, %c0_i32_13, %262 : <ui32>, i32 -> !ascvec.mask
                        %364 = ascvec.select %361, %358, %363 : <ui32>
                        %365 = ascvec.shr %364, 8, %262 : <ui32> -> <ui32>
                        %366 = ascvec.shl %364, 24, %262 : <ui32> -> <ui32>
                        %367 = ascvec.or %365, %366, %262 : <ui32>, <ui32> -> <ui32>
                        %368 = ascvec.and %355, %309, %262 : <ui32>, <ui32> -> <ui32>
                        %369 = ascvec.cmp_ne_scalar %368, %c0_i32_13, %262 : <ui32>, i32 -> !ascvec.mask
                        %370 = ascvec.select %367, %364, %369 : <ui32>
                        %371 = ascvec.shr %370, 16, %262 : <ui32> -> <ui32>
                        %372 = ascvec.shl %370, 16, %262 : <ui32> -> <ui32>
                        %373 = ascvec.or %371, %372, %262 : <ui32>, <ui32> -> <ui32>
                        %374 = ascvec.and %355, %316, %262 : <ui32>, <ui32> -> <ui32>
                        %375 = ascvec.cmp_ne_scalar %374, %c0_i32_13, %262 : <ui32>, i32 -> !ascvec.mask
                        %376 = ascvec.select %373, %370, %375 : <ui32>
                        %377 = ascvec.and %376, %320, %262 : <ui32>, <ui32> -> <ui32>
                        %378 = ascvec.add_scalar %287, %c3_i32, %262 : <ui32>, i32
                        %379 = ascvec.shr %378, 3, %262 : <ui32> -> <ui32>
                        %380 = ascvec.add_scalar %379, %290, %262 : <ui32>, i32
                        %381 = ascvec.gather %arg8[%c0_14], %380, %262 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %382 = ascvec.reinterpret %381 : <i32> -> <ui32>
                        %383 = ascvec.and %378, %294, %262 : <ui32>, <ui32> -> <ui32>
                        %384 = ascvec.shl %382, 11, %262 : <ui32> -> <ui32>
                        %385 = ascvec.shr %382, 21, %262 : <ui32> -> <ui32>
                        %386 = ascvec.or %384, %385, %262 : <ui32>, <ui32> -> <ui32>
                        %387 = ascvec.shr %386, 3, %262 : <ui32> -> <ui32>
                        %388 = ascvec.shl %386, 29, %262 : <ui32> -> <ui32>
                        %389 = ascvec.or %387, %388, %262 : <ui32>, <ui32> -> <ui32>
                        %390 = ascvec.and %383, %302, %262 : <ui32>, <ui32> -> <ui32>
                        %391 = ascvec.cmp_ne_scalar %390, %c0_i32_13, %262 : <ui32>, i32 -> !ascvec.mask
                        %392 = ascvec.select %389, %386, %391 : <ui32>
                        %393 = ascvec.shr %392, 8, %262 : <ui32> -> <ui32>
                        %394 = ascvec.shl %392, 24, %262 : <ui32> -> <ui32>
                        %395 = ascvec.or %393, %394, %262 : <ui32>, <ui32> -> <ui32>
                        %396 = ascvec.and %383, %309, %262 : <ui32>, <ui32> -> <ui32>
                        %397 = ascvec.cmp_ne_scalar %396, %c0_i32_13, %262 : <ui32>, i32 -> !ascvec.mask
                        %398 = ascvec.select %395, %392, %397 : <ui32>
                        %399 = ascvec.shr %398, 16, %262 : <ui32> -> <ui32>
                        %400 = ascvec.shl %398, 16, %262 : <ui32> -> <ui32>
                        %401 = ascvec.or %399, %400, %262 : <ui32>, <ui32> -> <ui32>
                        %402 = ascvec.and %383, %316, %262 : <ui32>, <ui32> -> <ui32>
                        %403 = ascvec.cmp_ne_scalar %402, %c0_i32_13, %262 : <ui32>, i32 -> !ascvec.mask
                        %404 = ascvec.select %401, %398, %403 : <ui32>
                        %405 = ascvec.and %404, %320, %262 : <ui32>, <ui32> -> <ui32>
                        %406 = ascvec.reinterpret %282 : <f32> -> <ui32>
                        %407 = ascvec.reinterpret %406 : <ui32> -> <f32>
                        %408 = ascvec.max_scalar %407, %cst, %262 : <f32>, f32
                        %409 = ascvec.add %407, %408, %262 : <f32>
                        %410 = ascvec.reinterpret %409 : <f32> -> <ui32>
                        %411 = ascvec.shr %410, 5, %262 : <ui32> -> <ui32>
                        %412 = ascvec.add %411, %321, %262 : <ui32>
                        %413 = ascvec.reinterpret %284 : <f32> -> <ui32>
                        %414 = ascvec.reinterpret %413 : <ui32> -> <f32>
                        %415 = ascvec.max_scalar %414, %cst, %262 : <f32>, f32
                        %416 = ascvec.add %414, %415, %262 : <f32>
                        %417 = ascvec.reinterpret %416 : <f32> -> <ui32>
                        %418 = ascvec.shr %417, 5, %262 : <ui32> -> <ui32>
                        %419 = ascvec.add %418, %349, %262 : <ui32>
                        %420 = ascvec.reinterpret %283 : <f32> -> <ui32>
                        %421 = ascvec.reinterpret %420 : <ui32> -> <f32>
                        %422 = ascvec.max_scalar %421, %cst, %262 : <f32>, f32
                        %423 = ascvec.add %421, %422, %262 : <f32>
                        %424 = ascvec.reinterpret %423 : <f32> -> <ui32>
                        %425 = ascvec.shr %424, 5, %262 : <ui32> -> <ui32>
                        %426 = ascvec.add %425, %377, %262 : <ui32>
                        %427 = ascvec.reinterpret %285 : <f32> -> <ui32>
                        %428 = ascvec.reinterpret %427 : <ui32> -> <f32>
                        %429 = ascvec.max_scalar %428, %cst, %262 : <f32>, f32
                        %430 = ascvec.add %428, %429, %262 : <f32>
                        %431 = ascvec.reinterpret %430 : <f32> -> <ui32>
                        %432 = ascvec.shr %431, 5, %262 : <ui32> -> <ui32>
                        %433 = ascvec.add %432, %405, %262 : <ui32>
                        %434 = ascvec.reinterpret %412 : <ui32> -> <ui16>
                        %435 = ascvec.reinterpret %419 : <ui32> -> <ui16>
                        %result0_15, %result1_16 = ascvec.deintlv %434, %435 : <ui16>, <ui16> -> <ui16>, <ui16>
                        %436 = ascvec.reinterpret %426 : <ui32> -> <ui16>
                        %437 = ascvec.reinterpret %433 : <ui32> -> <ui16>
                        %result0_17, %result1_18 = ascvec.deintlv %436, %437 : <ui16>, <ui16> -> <ui16>, <ui16>
                        %438 = ascvec.reinterpret %result1_16 : <ui16> -> <ui8>
                        %439 = ascvec.reinterpret %result1_18 : <ui16> -> <ui8>
                        %result0_19, %result1_20 = ascvec.deintlv %438, %439 : <ui8>, <ui8> -> <ui8>, <ui8>
                        %440 = ascvec.gather_reg %result0_19, %272 : <ui8>, <ui8> -> <ui8>
                        %441 = ascvec.add_scalar %440, %c60_i8, %264 : <ui8>, i8
                        %442 = ascvec.cmp_ne_scalar %441, %c-1_i8, %264 : <ui8>, i8 -> !ascvec.mask
                        %443 = ascvec.min_scalar %441, %c123_i8, %442 : <ui8>, i8
                        %444 = arith.muli %arg13, %c512_i32 : i32
                        %445 = arith.index_cast %444 : i32 to index
                        %result0_21, %result1_22 = ascvec.loadalign_deintlv %arg9[%445] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                        %446 = ascvec.and %result1_22, %266, %264 : <ui8>, <ui8> -> <ui8>
                        %447 = ascvec.or %443, %446, %264 : <ui8>, <ui8> -> <ui8>
                        %448 = ascvec.arange %c0_i16 : i16 -> <ui16>
                        %449 = arith.muli %arg13, %c16_i32 : i32
                        %450 = arith.trunci %449 : i32 to i16
                        %451 = ascvec.add_scalar %448, %450, %263 : <ui16>, i16
                        %452 = ascvec.gather %arg10[%c0_14], %451, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %453 = ascvec.pack %452 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                        %454 = ascvec.gather_reg %453, %267 : <ui8>, <ui8> -> <ui8>
                        %455 = ascvec.cmp_eq_scalar %454, %c-1_i8, %264 : <ui8>, i8 -> !ascvec.mask
                        %456 = ascvec.duplicate_scalar %c127_i8, %455 merge %447 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                        ascvec.storealign %arg11[%274], %456, %264 : memref<32768xui8, 1>, index, <ui8>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
                  }
                }
              }
            } else {
              %199 = arith.cmpi sge, %66, %c1024_i64 : i64
              %200 = arith.remsi %66, %c1024_i64 : i64
              %201 = arith.cmpi eq, %200, %c0_i64 : i64
              %202 = arith.select %199, %201, %false : i1
              scf.if %202 {
                %203 = arith.divsi %66, %c256_i64 : i64
                %204 = arith.addi %66, %c1023_i64 : i64
                %205 = arith.divsi %204, %c1024_i64 : i64
                %206 = arith.trunci %1 : i64 to i32
                %207 = arith.extui %206 : i32 to i64
                %208 = arith.subi %1, %207 : i64
                %209 = arith.remsi %208, %c4294967296_i64 : i64
                %210 = arith.cmpi ne, %209, %c0_i64 : i64
                %211 = arith.cmpi slt, %209, %c0_i64 : i64
                %212 = arith.andi %210, %211 : i1
                %213 = arith.divsi %208, %c4294967296_i64 : i64
                %214 = arith.extui %212 : i1 to i64
                %215 = arith.subi %213, %214 : i64
                %216 = arith.trunci %215 : i64 to i32
                %217 = arith.divui %108, %c16_i64 : i64
                %218 = arith.trunci %217 : i64 to i32
                %219 = arith.extui %218 : i32 to i64
                %220 = arith.subi %217, %219 : i64
                %221 = arith.remsi %220, %c4294967296_i64 : i64
                %222 = arith.cmpi ne, %221, %c0_i64 : i64
                %223 = arith.cmpi slt, %221, %c0_i64 : i64
                %224 = arith.andi %222, %223 : i1
                %225 = arith.divsi %220, %c4294967296_i64 : i64
                %226 = arith.extui %224 : i1 to i64
                %227 = arith.subi %225, %226 : i64
                %228 = arith.trunci %227 : i64 to i32
                %229 = arith.trunci %3 : i64 to i32
                %230 = arith.extui %229 : i32 to i64
                %231 = arith.subi %3, %230 : i64
                %232 = arith.remsi %231, %c4294967296_i64 : i64
                %233 = arith.cmpi ne, %232, %c0_i64 : i64
                %234 = arith.cmpi slt, %232, %c0_i64 : i64
                %235 = arith.andi %233, %234 : i1
                %236 = arith.divsi %231, %c4294967296_i64 : i64
                %237 = arith.extui %235 : i1 to i64
                %238 = arith.subi %236, %237 : i64
                %239 = arith.trunci %238 : i64 to i32
                %240 = arith.divui %229, %c4_i32 : i32
                %241 = arith.remui %239, %c4_i32 : i32
                %242 = arith.muli %241, %c1073741824_i32 : i32
                %243 = arith.addi %240, %242 : i32
                %244 = arith.divui %239, %c4_i32 : i32
                %245 = arith.muli %205, %c64_i64 : i64
                %246 = arith.divui %217, %c16384_i64 : i64
                %247 = arith.addi %217, %245 : i64
                %248 = arith.subi %247, %c1_i64 : i64
                %249 = arith.divui %248, %c16384_i64 : i64
                %250 = arith.cmpi eq, %246, %249 : i64
                scf.if %250 {
                  %260 = arith.trunci %246 : i64 to i32
                  %261 = arith.extui %260 : i32 to i64
                  %262 = arith.subi %246, %261 : i64
                  %263 = arith.remsi %262, %c4294967296_i64 : i64
                  %264 = arith.cmpi ne, %263, %c0_i64 : i64
                  %265 = arith.cmpi slt, %263, %c0_i64 : i64
                  %266 = arith.andi %264, %265 : i1
                  %267 = arith.divsi %262, %c4294967296_i64 : i64
                  %268 = arith.extui %266 : i1 to i64
                  %269 = arith.subi %267, %268 : i64
                  %270 = arith.trunci %269 : i64 to i32
                  %271 = arith.addi %243, %260 : i32
                  %272 = arith.addi %244, %270 : i32
                  %273 = arith.cmpi ult, %271, %243 : i32
                  %274 = arith.extui %273 : i1 to i32
                  %275 = arith.addi %272, %274 : i32
                  %276 = arith.extui %271 : i32 to i64
                  %277 = arith.muli %276, %c3449720151_i64 : i64
                  %278 = arith.trunci %277 : i64 to i32
                  %279 = arith.extui %278 : i32 to i64
                  %280 = arith.subi %277, %279 : i64
                  %281 = arith.remsi %280, %c4294967296_i64 : i64
                  %282 = arith.cmpi ne, %281, %c0_i64 : i64
                  %283 = arith.cmpi slt, %281, %c0_i64 : i64
                  %284 = arith.andi %282, %283 : i1
                  %285 = arith.divsi %280, %c4294967296_i64 : i64
                  %286 = arith.extui %284 : i1 to i64
                  %287 = arith.subi %285, %286 : i64
                  %288 = arith.trunci %287 : i64 to i32
                  %289 = arith.xori %275, %216 : i32
                  %290 = arith.extui %289 : i32 to i64
                  %291 = arith.muli %290, %c3449720151_i64 : i64
                  %292 = arith.trunci %291 : i64 to i32
                  %293 = arith.extui %292 : i32 to i64
                  %294 = arith.subi %291, %293 : i64
                  %295 = arith.remsi %294, %c4294967296_i64 : i64
                  %296 = arith.cmpi ne, %295, %c0_i64 : i64
                  %297 = arith.cmpi slt, %295, %c0_i64 : i64
                  %298 = arith.andi %296, %297 : i1
                  %299 = arith.divsi %294, %c4294967296_i64 : i64
                  %300 = arith.extui %298 : i1 to i64
                  %301 = arith.subi %299, %300 : i64
                  %302 = arith.trunci %301 : i64 to i32
                  %303 = arith.addi %206, %c-1640531527_i32 : i32
                  %304 = arith.addi %216, %c-1150833019_i32 : i32
                  %305 = arith.xori %302, %278 : i32
                  %306 = arith.xori %305, %303 : i32
                  %307 = arith.extui %306 : i32 to i64
                  %308 = arith.muli %307, %c3528531795_i64 : i64
                  %309 = arith.trunci %308 : i64 to i32
                  %310 = arith.extui %309 : i32 to i64
                  %311 = arith.subi %308, %310 : i64
                  %312 = arith.remsi %311, %c4294967296_i64 : i64
                  %313 = arith.cmpi ne, %312, %c0_i64 : i64
                  %314 = arith.cmpi slt, %312, %c0_i64 : i64
                  %315 = arith.andi %313, %314 : i1
                  %316 = arith.divsi %311, %c4294967296_i64 : i64
                  %317 = arith.extui %315 : i1 to i64
                  %318 = arith.subi %316, %317 : i64
                  %319 = arith.trunci %318 : i64 to i32
                  %320 = arith.xori %288, %206 : i32
                  %321 = arith.addi %206, %c1013904242_i32 : i32
                  %322 = arith.xori %292, %321 : i32
                  %323 = arith.addi %216, %c1993301258_i32 : i32
                  %324 = arith.xori %319, %323 : i32
                  ascsimt.vf(%218, %320, %304, %322, %324, %206, %216, %309, %alloc_8, %245 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
                    %c256_i64_10 = arith.constant 256 : i64
                    %c3_i64_11 = arith.constant 3 : i64
                    %c2_i64_12 = arith.constant 2 : i64
                    %c1_i64_13 = arith.constant 1 : i64
                    %c4_i64_14 = arith.constant 4 : i64
                    %c1_i32_15 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c3_i32 = arith.constant 3 : i32
                    %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %325 = ascsimt.thread_idx {ascsimt.scope = 14 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %326 = arith.extsi %325 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg17 = %326 to %arg16 step %c256_i64_10  : i64 {
                      %327 = arith.trunci %arg17 : i64 to i32
                      %328 = arith.addi %arg7, %327 : i32
                      %329 = arith.remui %328, %c16384_i32 : i32
                      %330 = arith.xori %329, %arg8 : i32
                      %331 = arith.muli %330, %c-766435501_i32 : i32
                      %332 = "ascsimt.bitcast"(%330) : (i32) -> ui32
                      %333 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %334 = ascsimt.math_binary %332, %333, "mulhi" : ui32, ui32 -> ui32
                      %335 = "ascsimt.bitcast"(%334) : (ui32) -> i32
                      %336 = arith.xori %335, %arg9 : i32
                      %337 = arith.muli %336, %c-845247145_i32 : i32
                      %338 = "ascsimt.bitcast"(%336) : (i32) -> ui32
                      %339 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %340 = ascsimt.math_binary %338, %339, "mulhi" : ui32, ui32 -> ui32
                      %341 = "ascsimt.bitcast"(%340) : (ui32) -> i32
                      %342 = arith.xori %341, %arg10 : i32
                      %343 = arith.xori %331, %arg11 : i32
                      %344:4 = scf.for %arg18 = %c3_i32 to %c10_i32 step %c1_i32_15 iter_args(%arg19 = %342, %arg20 = %337, %arg21 = %343, %arg22 = %arg14) -> (i32, i32, i32, i32)  : i32 {
                        %353 = arith.muli %arg19, %c-766435501_i32 : i32
                        %354 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %355 = ascsimt.math_binary %354, %333, "mulhi" : ui32, ui32 -> ui32
                        %356 = "ascsimt.bitcast"(%355) : (ui32) -> i32
                        %357 = arith.muli %arg21, %c-845247145_i32 : i32
                        %358 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %359 = ascsimt.math_binary %358, %339, "mulhi" : ui32, ui32 -> ui32
                        %360 = "ascsimt.bitcast"(%359) : (ui32) -> i32
                        %361 = arith.muli %arg18, %c-1640531527_i32_17 : i32
                        %362 = arith.addi %arg12, %361 : i32
                        %363 = arith.muli %arg18, %c-1150833019_i32_16 : i32
                        %364 = arith.addi %arg13, %363 : i32
                        %365 = arith.xori %360, %arg20 : i32
                        %366 = arith.xori %365, %362 : i32
                        %367 = arith.xori %356, %arg22 : i32
                        %368 = arith.xori %367, %364 : i32
                        scf.yield %366, %357, %368, %353 : i32, i32, i32, i32
                      }
                      %345 = arith.muli %arg17, %c4_i64_14 : i64
                      %346 = arith.index_cast %345 : i64 to index
                      ascsimt.store %arg15[%346], %344#0 : memref<8192xi32, 1>, i32
                      %347 = arith.addi %345, %c1_i64_13 : i64
                      %348 = arith.index_cast %347 : i64 to index
                      ascsimt.store %arg15[%348], %344#1 : memref<8192xi32, 1>, i32
                      %349 = arith.addi %345, %c2_i64_12 : i64
                      %350 = arith.index_cast %349 : i64 to index
                      ascsimt.store %arg15[%350], %344#2 : memref<8192xi32, 1>, i32
                      %351 = arith.addi %345, %c3_i64_11 : i64
                      %352 = arith.index_cast %351 : i64 to index
                      ascsimt.store %arg15[%352], %344#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%218, %228, %243, %244, %206, %216, %alloc_8, %245 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: memref<8192xi32, 1>, %arg14: i64):
                    %c256_i64_10 = arith.constant 256 : i64
                    %c3_i64_11 = arith.constant 3 : i64
                    %c2_i64_12 = arith.constant 2 : i64
                    %c1_i64_13 = arith.constant 1 : i64
                    %c4_i64_14 = arith.constant 4 : i64
                    %c1_i32_15 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c0_i32_16 = arith.constant 0 : i32
                    %c-1150833019_i32_17 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_18 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c262144_i32 = arith.constant 262144 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %260 = ascsimt.thread_idx {ascsimt.scope = 15 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %261 = arith.extsi %260 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg15 = %261 to %arg14 step %c256_i64_10  : i64 {
                      %262 = arith.trunci %arg15 : i64 to i32
                      %263 = arith.addi %arg7, %262 : i32
                      %264 = arith.cmpi ult, %263, %arg7 : i32
                      %265 = arith.extui %264 : i1 to i32
                      %266 = arith.addi %arg8, %265 : i32
                      %267 = arith.divui %263, %c16384_i32 : i32
                      %268 = arith.remui %266, %c16384_i32 : i32
                      %269 = arith.muli %268, %c262144_i32 : i32
                      %270 = arith.addi %267, %269 : i32
                      %271 = arith.divui %266, %c16384_i32 : i32
                      %272 = arith.addi %arg9, %270 : i32
                      %273 = arith.addi %arg10, %271 : i32
                      %274 = arith.cmpi ult, %272, %arg9 : i32
                      %275 = arith.extui %274 : i1 to i32
                      %276 = arith.addi %273, %275 : i32
                      %277 = arith.remui %263, %c16384_i32 : i32
                      %278:4 = scf.for %arg16 = %c0_i32_16 to %c10_i32 step %c1_i32_15 iter_args(%arg17 = %c0_i32_16, %arg18 = %277, %arg19 = %272, %arg20 = %276) -> (i32, i32, i32, i32)  : i32 {
                        %287 = arith.muli %arg17, %c-766435501_i32 : i32
                        %288 = "ascsimt.bitcast"(%arg17) : (i32) -> ui32
                        %289 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %290 = ascsimt.math_binary %288, %289, "mulhi" : ui32, ui32 -> ui32
                        %291 = "ascsimt.bitcast"(%290) : (ui32) -> i32
                        %292 = arith.muli %arg19, %c-845247145_i32 : i32
                        %293 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %294 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %295 = ascsimt.math_binary %293, %294, "mulhi" : ui32, ui32 -> ui32
                        %296 = "ascsimt.bitcast"(%295) : (ui32) -> i32
                        %297 = arith.muli %arg16, %c-1640531527_i32_18 : i32
                        %298 = arith.addi %arg11, %297 : i32
                        %299 = arith.muli %arg16, %c-1150833019_i32_17 : i32
                        %300 = arith.addi %arg12, %299 : i32
                        %301 = arith.xori %296, %arg18 : i32
                        %302 = arith.xori %301, %298 : i32
                        %303 = arith.xori %291, %arg20 : i32
                        %304 = arith.xori %303, %300 : i32
                        scf.yield %302, %292, %304, %287 : i32, i32, i32, i32
                      }
                      %279 = arith.muli %arg15, %c4_i64_14 : i64
                      %280 = arith.index_cast %279 : i64 to index
                      ascsimt.store %arg13[%280], %278#0 : memref<8192xi32, 1>, i32
                      %281 = arith.addi %279, %c1_i64_13 : i64
                      %282 = arith.index_cast %281 : i64 to index
                      ascsimt.store %arg13[%282], %278#1 : memref<8192xi32, 1>, i32
                      %283 = arith.addi %279, %c2_i64_12 : i64
                      %284 = arith.index_cast %283 : i64 to index
                      ascsimt.store %arg13[%284], %278#2 : memref<8192xi32, 1>, i32
                      %285 = arith.addi %279, %c3_i64_11 : i64
                      %286 = arith.index_cast %285 : i64 to index
                      ascsimt.store %arg13[%286], %278#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %251 = arith.trunci %48 : i64 to i32
                %252 = arith.divsi %251, %c2_i32 : i32
                %253 = arith.trunci %252 : i32 to i8
                %254 = arith.trunci %203 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %255 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%253, %alloc_6, %107, %255, %alloc_9, %254 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, memref<2048xui16, 1>, i32) {
                ^bb0(%arg7: i8, %arg8: memref<2048xui8, 1>, %arg9: memref<32768xbf16, 1>, %arg10: memref<?xui16, 1>, %arg11: memref<2048xui16, 1>, %arg12: i32):
                  %c16_10 = arith.constant 16 : index
                  %c4_i32_11 = arith.constant 4 : i32
                  %c0_i32_12 = arith.constant 0 : i32
                  %c3_i32 = arith.constant 3 : i32
                  %c2_i32_13 = arith.constant 2 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c1_i32_14 = arith.constant 1 : i32
                  %c0_15 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %260 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %261 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %262 = ascvec.duplicate_scalar %c32767_i16, %260 : i16 -> <ui16>
                  %263 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %264 = ascvec.shr %263, 4, %260 : <ui16> -> <ui16>
                  %265 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %266 = ascvec.duplicate_scalar %c1_i8, %261 : i8 -> <ui8>
                  %267 = ascvec.and %265, %266, %261 : <ui8>, <ui8> -> <ui8>
                  %268 = ascvec.cmp_lt_scalar %265, %c16_i8, %261 : <ui8>, i8 -> !ascvec.mask
                  %269 = ascvec.cmp_lt_scalar %267, %arg7, %261 : <ui8>, i8 -> !ascvec.mask
                  %270 = ascvec.mask_and %268, %269, %261
                  ascvec.store_unalign_begin %arg8 : memref<2048xui8, 1>
                  %mask_16, %remaining_out_17 = ascvec.update_mask %c1_i32_14 elem_bits 16 : !ascvec.mask
                  %271 = ascvec.duplicate_scalar %c-1_i16, %mask_16 : i16 -> <ui16>
                  %272 = scf.for %arg13 = %c0_i32_12 to %arg12 step %c4_i32_11 iter_args(%arg14 = %271) -> (!ascvec.vreg<ui16>)  : i32 {
                    %273 = ascvec.duplicate_scalar %c-1_i16, %260 : i16 -> <ui16>
                    %274 = ascvec.duplicate_scalar %c0_i16, %260 : i16 -> <ui16>
                    %275 = arith.muli %arg13, %c256_i32 : i32
                    %276 = arith.index_cast %275 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg9[%276] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %277 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %278 = ascvec.and %277, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %279 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %280 = ascvec.and %279, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %281 = ascvec.max %278, %280, %260 : <ui16>
                    %282 = ascvec.reduce_max_datablock %281, %260 : <ui16>
                    %283 = ascvec.shr %282, 7, %260 : <ui16> -> <ui16>
                    %284 = ascvec.cmp_ge_scalar %283, %c255_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %285 = ascvec.max_scalar %283, %c15_i16, %260 : <ui16>, i16
                    %286 = ascvec.add_scalar %285, %c-15_i16, %260 : <ui16>, i16
                    %287 = ascvec.duplicate_scalar %c255_i16, %284 merge %286 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %288 = ascvec.duplicate_scalar %c254_i16, %260 : i16 -> <ui16>
                    %289 = ascvec.sub %288, %287, %260 : <ui16>
                    %290 = ascvec.cmp_ne_scalar %287, %c0_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %291 = ascvec.shl %289, 7, %290 : <ui16> -> <ui16>
                    %292 = ascvec.max %274, %287, %mask : <ui16>
                    %293 = ascvec.gather_reg %291, %264 : <ui16>, <ui16> -> <ui16>
                    %294 = ascvec.reinterpret %293 : <ui16> -> <bf16>
                    %295 = ascvec.mul %result0, %294, %260 : <bf16>
                    %296 = ascvec.mul %result1, %294, %260 : <bf16>
                    ascvec.storealign_intlv %arg9[%276], %295, %296 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %297 = ascvec.reinterpret %295 : <bf16> -> <ui16>
                    %298 = ascvec.and %297, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %299 = ascvec.reinterpret %296 : <bf16> -> <ui16>
                    %300 = ascvec.and %299, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %301 = ascvec.min %298, %300, %260 : <ui16>
                    %302 = ascvec.min %273, %301, %260 : <ui16>
                    %303 = arith.muli %arg13, %c16_i32 : i32
                    %304 = arith.index_cast %303 : i32 to index
                    ascvec.storealign %arg10[%304], %287, %mask : memref<?xui16, 1>, index, <ui16>
                    %305 = ascvec.reinterpret %287 : <ui16> -> <ui8>
                    %306 = ascvec.squeeze_store_reg %305, %270 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_15], %306 : memref<2048xui8, 1>, <ui8>
                    %307 = arith.addi %arg13, %c1_i32_14 : i32
                    %308 = arith.muli %307, %c256_i32 : i32
                    %309 = arith.index_cast %308 : i32 to index
                    %result0_20, %result1_21 = ascvec.loadalign_deintlv %arg9[%309] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %310 = ascvec.reinterpret %result0_20 : <bf16> -> <ui16>
                    %311 = ascvec.and %310, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %312 = ascvec.reinterpret %result1_21 : <bf16> -> <ui16>
                    %313 = ascvec.and %312, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %314 = ascvec.max %311, %313, %260 : <ui16>
                    %315 = ascvec.reduce_max_datablock %314, %260 : <ui16>
                    %316 = ascvec.shr %315, 7, %260 : <ui16> -> <ui16>
                    %317 = ascvec.cmp_ge_scalar %316, %c255_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %318 = ascvec.max_scalar %316, %c15_i16, %260 : <ui16>, i16
                    %319 = ascvec.add_scalar %318, %c-15_i16, %260 : <ui16>, i16
                    %320 = ascvec.duplicate_scalar %c255_i16, %317 merge %319 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %321 = ascvec.sub %288, %320, %260 : <ui16>
                    %322 = ascvec.cmp_ne_scalar %320, %c0_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %323 = ascvec.shl %321, 7, %322 : <ui16> -> <ui16>
                    %324 = ascvec.max %292, %320, %mask : <ui16>
                    %325 = ascvec.gather_reg %323, %264 : <ui16>, <ui16> -> <ui16>
                    %326 = ascvec.reinterpret %325 : <ui16> -> <bf16>
                    %327 = ascvec.mul %result0_20, %326, %260 : <bf16>
                    %328 = ascvec.mul %result1_21, %326, %260 : <bf16>
                    ascvec.storealign_intlv %arg9[%309], %327, %328 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %329 = ascvec.reinterpret %327 : <bf16> -> <ui16>
                    %330 = ascvec.and %329, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %331 = ascvec.reinterpret %328 : <bf16> -> <ui16>
                    %332 = ascvec.and %331, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %333 = ascvec.min %330, %332, %260 : <ui16>
                    %334 = ascvec.min %302, %333, %260 : <ui16>
                    %335 = arith.muli %307, %c16_i32 : i32
                    %336 = arith.index_cast %335 : i32 to index
                    ascvec.storealign %arg10[%336], %320, %mask : memref<?xui16, 1>, index, <ui16>
                    %337 = ascvec.reinterpret %320 : <ui16> -> <ui8>
                    %338 = ascvec.squeeze_store_reg %337, %270 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_15], %338 : memref<2048xui8, 1>, <ui8>
                    %339 = arith.addi %arg13, %c2_i32_13 : i32
                    %340 = arith.muli %339, %c256_i32 : i32
                    %341 = arith.index_cast %340 : i32 to index
                    %result0_22, %result1_23 = ascvec.loadalign_deintlv %arg9[%341] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %342 = ascvec.reinterpret %result0_22 : <bf16> -> <ui16>
                    %343 = ascvec.and %342, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %344 = ascvec.reinterpret %result1_23 : <bf16> -> <ui16>
                    %345 = ascvec.and %344, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %346 = ascvec.max %343, %345, %260 : <ui16>
                    %347 = ascvec.reduce_max_datablock %346, %260 : <ui16>
                    %348 = ascvec.shr %347, 7, %260 : <ui16> -> <ui16>
                    %349 = ascvec.cmp_ge_scalar %348, %c255_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %350 = ascvec.max_scalar %348, %c15_i16, %260 : <ui16>, i16
                    %351 = ascvec.add_scalar %350, %c-15_i16, %260 : <ui16>, i16
                    %352 = ascvec.duplicate_scalar %c255_i16, %349 merge %351 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %353 = ascvec.sub %288, %352, %260 : <ui16>
                    %354 = ascvec.cmp_ne_scalar %352, %c0_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %355 = ascvec.shl %353, 7, %354 : <ui16> -> <ui16>
                    %356 = ascvec.max %324, %352, %mask : <ui16>
                    %357 = ascvec.gather_reg %355, %264 : <ui16>, <ui16> -> <ui16>
                    %358 = ascvec.reinterpret %357 : <ui16> -> <bf16>
                    %359 = ascvec.mul %result0_22, %358, %260 : <bf16>
                    %360 = ascvec.mul %result1_23, %358, %260 : <bf16>
                    ascvec.storealign_intlv %arg9[%341], %359, %360 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %361 = ascvec.reinterpret %359 : <bf16> -> <ui16>
                    %362 = ascvec.and %361, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %363 = ascvec.reinterpret %360 : <bf16> -> <ui16>
                    %364 = ascvec.and %363, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %365 = ascvec.min %362, %364, %260 : <ui16>
                    %366 = ascvec.min %334, %365, %260 : <ui16>
                    %367 = arith.muli %339, %c16_i32 : i32
                    %368 = arith.index_cast %367 : i32 to index
                    ascvec.storealign %arg10[%368], %352, %mask : memref<?xui16, 1>, index, <ui16>
                    %369 = ascvec.reinterpret %352 : <ui16> -> <ui8>
                    %370 = ascvec.squeeze_store_reg %369, %270 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_15], %370 : memref<2048xui8, 1>, <ui8>
                    %371 = arith.addi %arg13, %c3_i32 : i32
                    %372 = arith.muli %371, %c256_i32 : i32
                    %373 = arith.index_cast %372 : i32 to index
                    %result0_24, %result1_25 = ascvec.loadalign_deintlv %arg9[%373] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %374 = ascvec.reinterpret %result0_24 : <bf16> -> <ui16>
                    %375 = ascvec.and %374, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %376 = ascvec.reinterpret %result1_25 : <bf16> -> <ui16>
                    %377 = ascvec.and %376, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %378 = ascvec.max %375, %377, %260 : <ui16>
                    %379 = ascvec.reduce_max_datablock %378, %260 : <ui16>
                    %380 = ascvec.shr %379, 7, %260 : <ui16> -> <ui16>
                    %381 = ascvec.cmp_ge_scalar %380, %c255_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %382 = ascvec.max_scalar %380, %c15_i16, %260 : <ui16>, i16
                    %383 = ascvec.add_scalar %382, %c-15_i16, %260 : <ui16>, i16
                    %384 = ascvec.duplicate_scalar %c255_i16, %381 merge %383 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %385 = ascvec.sub %288, %384, %260 : <ui16>
                    %386 = ascvec.cmp_ne_scalar %384, %c0_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %387 = ascvec.shl %385, 7, %386 : <ui16> -> <ui16>
                    %388 = ascvec.max %356, %384, %mask : <ui16>
                    %389 = ascvec.gather_reg %387, %264 : <ui16>, <ui16> -> <ui16>
                    %390 = ascvec.reinterpret %389 : <ui16> -> <bf16>
                    %391 = ascvec.mul %result0_24, %390, %260 : <bf16>
                    %392 = ascvec.mul %result1_25, %390, %260 : <bf16>
                    ascvec.storealign_intlv %arg9[%373], %391, %392 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %393 = ascvec.reinterpret %391 : <bf16> -> <ui16>
                    %394 = ascvec.and %393, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %395 = ascvec.reinterpret %392 : <bf16> -> <ui16>
                    %396 = ascvec.and %395, %262, %260 : <ui16>, <ui16> -> <ui16>
                    %397 = ascvec.min %394, %396, %260 : <ui16>
                    %398 = ascvec.min %366, %397, %260 : <ui16>
                    %399 = arith.muli %371, %c16_i32 : i32
                    %400 = arith.index_cast %399 : i32 to index
                    ascvec.storealign %arg10[%400], %384, %mask : memref<?xui16, 1>, index, <ui16>
                    %401 = ascvec.reinterpret %384 : <ui16> -> <ui8>
                    %402 = ascvec.squeeze_store_reg %401, %270 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_15], %402 : memref<2048xui8, 1>, <ui8>
                    %403 = ascvec.reduce_min %398, %260 : <ui16>
                    %404 = ascvec.reduce_max %388, %mask : <ui16>
                    %405 = ascvec.cmp_eq_scalar %404, %c255_i16, %260 : <ui16>, i16 -> !ascvec.mask
                    %406 = ascvec.duplicate_scalar %c0_i16, %405 merge %403 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.storealign_1st %arg11[%304], %406 : memref<2048xui16, 1>, index, <ui16>
                    %407 = ascvec.min %arg14, %406, %mask_16 : <ui16>
                    scf.yield %407 : !ascvec.vreg<ui16>
                  }
                  %c0_18 = arith.constant 0 : index
                  %c1_19 = arith.constant 1 : index
                  scf.for %arg13 = %c0_18 to %c1_19 step %c1_19 {
                    ascvec.store_unalign_post %arg8[%c0_15] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg11[%c16_10], %272 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %256 = memref.load %alloc_9[%c16] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %257 = cannir.scalar_storage_cast %256 : ui16 -> i16
                %258 = arith.extui %257 : i16 to i64
                %259 = arith.cmpi sge, %258, %c14464_i64 : i64
                scf.if %259 {
                  ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %260 = ascscalar.bitcast_memref %107 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%107, %alloc_8, %260, %alloc_5, %254 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg7: memref<32768xbf16, 1>, %arg8: memref<8192xi32, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<32768xui8, 1>, %arg11: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_10 = arith.constant 1 : i32
                    %c0_i32_11 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c64_i32 = arith.constant 64 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c1_i16 = arith.constant 1 : i16
                    %c0_i16 = arith.constant 0 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %261 = ascvec.create_mask : !ascvec.mask
                    %262 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %263 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %264 = ascvec.duplicate_scalar %c255_i16, %262 : i16 -> <ui16>
                    %265 = ascvec.duplicate_scalar %c-128_i8, %263 : i8 -> <ui8>
                    %266 = ascvec.arange %c0_i16 : i16 -> <ui16>
                    %267 = ascvec.duplicate_scalar %c1_i16, %262 : i16 -> <ui16>
                    %268 = ascvec.xor %266, %267, %262 : <ui16>, <ui16> -> <ui16>
                    scf.for %arg12 = %c0_i32_11 to %arg11 step %c1_i32_10  : i32 {
                      %269 = arith.muli %arg12, %c256_i32 : i32
                      %270 = arith.index_cast %269 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg7[%270] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %271 = arith.muli %arg12, %c64_i32 : i32
                      %272 = arith.index_cast %271 : i32 to index
                      %273 = ascvec.loadalign %arg8[%272] : memref<8192xi32, 1>, index -> <i32>
                      %274 = ascvec.reinterpret %273 : <i32> -> <ui16>
                      %275 = ascvec.reinterpret %274 : <ui16> -> <ui32>
                      %276 = ascvec.shl %275, 8, %261 : <ui32> -> <ui32>
                      %277 = ascvec.shr %275, 24, %261 : <ui32> -> <ui32>
                      %278 = ascvec.or %276, %277, %261 : <ui32>, <ui32> -> <ui32>
                      %279 = ascvec.reinterpret %278 : <ui32> -> <ui16>
                      %280 = ascvec.gather_reg %274, %268 : <ui16>, <ui16> -> <ui16>
                      %281 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %282 = ascvec.shr %280, 11, %262 : <ui16> -> <ui16>
                      %283 = ascvec.add %281, %282, %262 : <ui16>
                      %284 = ascvec.shr %283, 5, %262 : <ui16> -> <ui16>
                      %285 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %286 = ascvec.shr %279, 11, %262 : <ui16> -> <ui16>
                      %287 = ascvec.add %285, %286, %262 : <ui16>
                      %288 = ascvec.shr %287, 5, %262 : <ui16> -> <ui16>
                      %289 = ascvec.and %284, %264, %262 : <ui16>, <ui16> -> <ui16>
                      %290 = ascvec.shl %288, 8, %262 : <ui16> -> <ui16>
                      %291 = ascvec.or %289, %290, %262 : <ui16>, <ui16> -> <ui16>
                      %292 = ascvec.reinterpret %291 : <ui16> -> <ui8>
                      %293 = ascvec.add_scalar %292, %c64_i8, %263 : <ui8>, i8
                      %294 = ascvec.min_scalar %293, %c123_i8, %263 : <ui8>, i8
                      %295 = arith.muli %arg12, %c512_i32 : i32
                      %296 = arith.index_cast %295 : i32 to index
                      %result0_12, %result1_13 = ascvec.loadalign_deintlv %arg9[%296] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %297 = ascvec.and %result1_13, %265, %263 : <ui8>, <ui8> -> <ui8>
                      %298 = ascvec.or %294, %297, %263 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg10[%270], %298, %263 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
                } else {
                  scf.for %arg7 = %c0_i64 to %203 step %c16_i64  : i64 {
                    %260 = arith.addi %arg7, %c16_i64 : i64
                    %261 = arith.cmpi sgt, %260, %203 : i64
                    %262 = arith.select %261, %203, %260 : i64
                    %263 = arith.muli %arg7, %c16_i64 : i64
                    ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %264 = arith.index_cast %263 : i64 to index
                    %265 = memref.load %alloc_9[%264] : memref<2048xui16, 1>
                    ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %266 = cannir.scalar_storage_cast %265 : ui16 -> i16
                    %267 = arith.extui %266 : i16 to i64
                    %268 = arith.addi %arg7, %c4_i64 : i64
                    %269 = scf.for %arg8 = %268 to %262 step %c4_i64 iter_args(%arg9 = %267) -> (i64)  : i64 {
                      %271 = arith.muli %arg8, %c16_i64 : i64
                      ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %272 = arith.index_cast %271 : i64 to index
                      %273 = memref.load %alloc_9[%272] : memref<2048xui16, 1>
                      ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %274 = cannir.scalar_storage_cast %273 : ui16 -> i16
                      %275 = arith.extui %274 : i16 to i64
                      %276 = arith.cmpi slt, %275, %arg9 : i64
                      %277 = arith.select %276, %275, %arg9 : i64
                      scf.yield %277 : i64
                    }
                    %270 = arith.cmpi sge, %269, %c14464_i64 : i64
                    scf.if %270 {
                      %271 = arith.trunci %arg7 : i64 to i32
                      %272 = arith.trunci %262 : i64 to i32
                      ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %273 = ascscalar.bitcast_memref %107 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                      ascvec.vf(%107, %alloc_8, %273, %alloc_5, %271, %272 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                      ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<8192xi32, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32, %arg13: i32):
                        %c512_i32 = arith.constant 512 : i32
                        %c1_i32_10 = arith.constant 1 : i32
                        %c123_i8 = arith.constant 123 : i8
                        %c64_i8 = arith.constant 64 : i8
                        %c64_i32 = arith.constant 64 : i32
                        %c256_i32 = arith.constant 256 : i32
                        %c1_i16 = arith.constant 1 : i16
                        %c0_i16 = arith.constant 0 : i16
                        %c-128_i8 = arith.constant -128 : i8
                        %c255_i16 = arith.constant 255 : i16
                        %274 = ascvec.create_mask : !ascvec.mask
                        %275 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                        %276 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                        %277 = ascvec.duplicate_scalar %c255_i16, %275 : i16 -> <ui16>
                        %278 = ascvec.duplicate_scalar %c-128_i8, %276 : i8 -> <ui8>
                        %279 = ascvec.arange %c0_i16 : i16 -> <ui16>
                        %280 = ascvec.duplicate_scalar %c1_i16, %275 : i16 -> <ui16>
                        %281 = ascvec.xor %279, %280, %275 : <ui16>, <ui16> -> <ui16>
                        scf.for %arg14 = %arg12 to %arg13 step %c1_i32_10  : i32 {
                          %282 = arith.muli %arg14, %c256_i32 : i32
                          %283 = arith.index_cast %282 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg8[%283] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                          %284 = arith.muli %arg14, %c64_i32 : i32
                          %285 = arith.index_cast %284 : i32 to index
                          %286 = ascvec.loadalign %arg9[%285] : memref<8192xi32, 1>, index -> <i32>
                          %287 = ascvec.reinterpret %286 : <i32> -> <ui16>
                          %288 = ascvec.reinterpret %287 : <ui16> -> <ui32>
                          %289 = ascvec.shl %288, 8, %274 : <ui32> -> <ui32>
                          %290 = ascvec.shr %288, 24, %274 : <ui32> -> <ui32>
                          %291 = ascvec.or %289, %290, %274 : <ui32>, <ui32> -> <ui32>
                          %292 = ascvec.reinterpret %291 : <ui32> -> <ui16>
                          %293 = ascvec.gather_reg %287, %281 : <ui16>, <ui16> -> <ui16>
                          %294 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                          %295 = ascvec.shr %293, 11, %275 : <ui16> -> <ui16>
                          %296 = ascvec.add %294, %295, %275 : <ui16>
                          %297 = ascvec.shr %296, 5, %275 : <ui16> -> <ui16>
                          %298 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                          %299 = ascvec.shr %292, 11, %275 : <ui16> -> <ui16>
                          %300 = ascvec.add %298, %299, %275 : <ui16>
                          %301 = ascvec.shr %300, 5, %275 : <ui16> -> <ui16>
                          %302 = ascvec.and %297, %277, %275 : <ui16>, <ui16> -> <ui16>
                          %303 = ascvec.shl %301, 8, %275 : <ui16> -> <ui16>
                          %304 = ascvec.or %302, %303, %275 : <ui16>, <ui16> -> <ui16>
                          %305 = ascvec.reinterpret %304 : <ui16> -> <ui8>
                          %306 = ascvec.add_scalar %305, %c64_i8, %276 : <ui8>, i8
                          %307 = ascvec.min_scalar %306, %c123_i8, %276 : <ui8>, i8
                          %308 = arith.muli %arg14, %c512_i32 : i32
                          %309 = arith.index_cast %308 : i32 to index
                          %result0_11, %result1_12 = ascvec.loadalign_deintlv %arg10[%309] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                          %310 = ascvec.and %result1_12, %278, %276 : <ui8>, <ui8> -> <ui8>
                          %311 = ascvec.or %307, %310, %276 : <ui8>, <ui8> -> <ui8>
                          ascvec.storealign %arg11[%283], %311, %276 : memref<32768xui8, 1>, index, <ui8>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
                    } else {
                      %271 = arith.cmpi sge, %269, %c13568_i64 : i64
                      scf.if %271 {
                        %272 = arith.trunci %arg7 : i64 to i32
                        %273 = arith.trunci %262 : i64 to i32
                        ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %274 = ascscalar.bitcast_memref %107 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%107, %alloc_8, %274, %alloc_5, %272, %273 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<8192xi32, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32, %arg13: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_10 = arith.constant 1 : i32
                          %c123_i8 = arith.constant 123 : i8
                          %c64_i8 = arith.constant 64 : i8
                          %c448_i16 = arith.constant 448 : i16
                          %c14464_i16 = arith.constant 14464 : i16
                          %c64_i32 = arith.constant 64 : i32
                          %c256_i32 = arith.constant 256 : i32
                          %cst = arith.constant 2.684350e+08 : bf16
                          %c32767_i16 = arith.constant 32767 : i16
                          %c1_i16 = arith.constant 1 : i16
                          %c0_i16 = arith.constant 0 : i16
                          %c-128_i8 = arith.constant -128 : i8
                          %c255_i16 = arith.constant 255 : i16
                          %275 = ascvec.create_mask : !ascvec.mask
                          %276 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %277 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %278 = ascvec.duplicate_scalar %c255_i16, %276 : i16 -> <ui16>
                          %279 = ascvec.duplicate_scalar %c-128_i8, %277 : i8 -> <ui8>
                          %280 = ascvec.arange %c0_i16 : i16 -> <ui16>
                          %281 = ascvec.duplicate_scalar %c1_i16, %276 : i16 -> <ui16>
                          %282 = ascvec.xor %280, %281, %276 : <ui16>, <ui16> -> <ui16>
                          %283 = ascvec.duplicate_scalar %c32767_i16, %276 : i16 -> <ui16>
                          %284 = ascvec.duplicate_scalar %cst, %276 : bf16 -> <bf16>
                          scf.for %arg14 = %arg12 to %arg13 step %c1_i32_10  : i32 {
                            %285 = arith.muli %arg14, %c256_i32 : i32
                            %286 = arith.index_cast %285 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg8[%286] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %287 = arith.muli %arg14, %c64_i32 : i32
                            %288 = arith.index_cast %287 : i32 to index
                            %289 = ascvec.loadalign %arg9[%288] : memref<8192xi32, 1>, index -> <i32>
                            %290 = ascvec.reinterpret %289 : <i32> -> <ui16>
                            %291 = ascvec.reinterpret %290 : <ui16> -> <ui32>
                            %292 = ascvec.shl %291, 8, %275 : <ui32> -> <ui32>
                            %293 = ascvec.shr %291, 24, %275 : <ui32> -> <ui32>
                            %294 = ascvec.or %292, %293, %275 : <ui32>, <ui32> -> <ui32>
                            %295 = ascvec.reinterpret %294 : <ui32> -> <ui16>
                            %296 = ascvec.gather_reg %290, %282 : <ui16>, <ui16> -> <ui16>
                            %297 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %298 = ascvec.and %297, %283, %276 : <ui16>, <ui16> -> <ui16>
                            %299 = ascvec.cmp_lt_scalar %298, %c14464_i16, %276 : <ui16>, i16 -> !ascvec.mask
                            %300 = ascvec.shr %296, 11, %276 : <ui16> -> <ui16>
                            %301 = ascvec.add %298, %300, %276 : <ui16>
                            %302 = ascvec.shr %301, 5, %276 : <ui16> -> <ui16>
                            %303 = ascvec.reinterpret %298 : <ui16> -> <bf16>
                            %304 = ascvec.mul %303, %284, %299 : <bf16>
                            %305 = ascvec.cast %304, %276 {rounding = 4 : i32} : <bf16> -> <f16>
                            %306 = ascvec.cast %305, %276 {rounding = 4 : i32} : <f16> -> <i16>
                            %307 = ascvec.reinterpret %306 : <i16> -> <ui16>
                            %308 = ascvec.shr %296, 4, %276 : <ui16> -> <ui16>
                            %309 = ascvec.add %307, %308, %276 : <ui16>
                            %310 = ascvec.shr %309, 12, %276 : <ui16> -> <ui16>
                            %311 = ascvec.add_scalar %310, %c448_i16, %276 : <ui16>, i16
                            %312 = ascvec.select %311, %302, %299 : <ui16>
                            %313 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %314 = ascvec.and %313, %283, %276 : <ui16>, <ui16> -> <ui16>
                            %315 = ascvec.cmp_lt_scalar %314, %c14464_i16, %276 : <ui16>, i16 -> !ascvec.mask
                            %316 = ascvec.shr %295, 11, %276 : <ui16> -> <ui16>
                            %317 = ascvec.add %314, %316, %276 : <ui16>
                            %318 = ascvec.shr %317, 5, %276 : <ui16> -> <ui16>
                            %319 = ascvec.reinterpret %314 : <ui16> -> <bf16>
                            %320 = ascvec.mul %319, %284, %315 : <bf16>
                            %321 = ascvec.cast %320, %276 {rounding = 4 : i32} : <bf16> -> <f16>
                            %322 = ascvec.cast %321, %276 {rounding = 4 : i32} : <f16> -> <i16>
                            %323 = ascvec.reinterpret %322 : <i16> -> <ui16>
                            %324 = ascvec.shr %295, 4, %276 : <ui16> -> <ui16>
                            %325 = ascvec.add %323, %324, %276 : <ui16>
                            %326 = ascvec.shr %325, 12, %276 : <ui16> -> <ui16>
                            %327 = ascvec.add_scalar %326, %c448_i16, %276 : <ui16>, i16
                            %328 = ascvec.select %327, %318, %315 : <ui16>
                            %329 = ascvec.and %312, %278, %276 : <ui16>, <ui16> -> <ui16>
                            %330 = ascvec.shl %328, 8, %276 : <ui16> -> <ui16>
                            %331 = ascvec.or %329, %330, %276 : <ui16>, <ui16> -> <ui16>
                            %332 = ascvec.reinterpret %331 : <ui16> -> <ui8>
                            %333 = ascvec.add_scalar %332, %c64_i8, %277 : <ui8>, i8
                            %334 = ascvec.min_scalar %333, %c123_i8, %277 : <ui8>, i8
                            %335 = arith.muli %arg14, %c512_i32 : i32
                            %336 = arith.index_cast %335 : i32 to index
                            %result0_11, %result1_12 = ascvec.loadalign_deintlv %arg10[%336] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %337 = ascvec.and %result1_12, %279, %277 : <ui8>, <ui8> -> <ui8>
                            %338 = ascvec.or %334, %337, %277 : <ui8>, <ui8> -> <ui8>
                            ascvec.storealign %arg11[%286], %338, %277 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
                      } else {
                        %272 = arith.trunci %arg7 : i64 to i32
                        %273 = arith.trunci %262 : i64 to i32
                        ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %274 = ascscalar.bitcast_memref %107 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%107, %alloc_8, %274, %255, %alloc_5, %272, %273 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<8192xi32, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<?xui16, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32, %arg14: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_10 = arith.constant 1 : i32
                          %c127_i8 = arith.constant 127 : i8
                          %c16_i32 = arith.constant 16 : i32
                          %c0_i16 = arith.constant 0 : i16
                          %c123_i8 = arith.constant 123 : i8
                          %c-1_i8 = arith.constant -1 : i8
                          %c60_i8 = arith.constant 60 : i8
                          %cst = arith.constant 6.10351526E-5 : f32
                          %c64_i32 = arith.constant 64 : i32
                          %c32767_i16 = arith.constant 32767 : i16
                          %c0_11 = arith.constant 0 : index
                          %c256_i32 = arith.constant 256 : i32
                          %c65535_i32 = arith.constant 65535 : i32
                          %c3_i8 = arith.constant 3 : i8
                          %c-128_i8 = arith.constant -128 : i8
                          %c0_i8 = arith.constant 0 : i8
                          %c8_i32 = arith.constant 8 : i32
                          %275 = ascvec.create_mask : !ascvec.mask
                          %276 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %277 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                          %278 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %279 = ascvec.duplicate_scalar %c-128_i8, %277 : i8 -> <ui8>
                          %280 = ascvec.shr %278, 5, %277 : <ui8> -> <ui8>
                          %281 = ascvec.duplicate_scalar %c3_i8, %277 : i8 -> <ui8>
                          %282 = ascvec.and %278, %281, %277 : <ui8>, <ui8> -> <ui8>
                          %283 = ascvec.shl %282, 6, %277 : <ui8> -> <ui8>
                          %284 = ascvec.shr %278, 2, %277 : <ui8> -> <ui8>
                          %285 = ascvec.or %283, %284, %277 : <ui8>, <ui8> -> <ui8>
                          %286 = ascvec.duplicate_scalar %c65535_i32, %275 : i32 -> <ui32>
                          scf.for %arg15 = %arg13 to %arg14 step %c1_i32_10  : i32 {
                            %287 = arith.muli %arg15, %c256_i32 : i32
                            %288 = arith.index_cast %287 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg8[%288] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %289 = ascvec.duplicate_scalar %c32767_i16, %276 : i16 -> <ui16>
                            %290 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %291 = ascvec.and %290, %289, %276 : <ui16>, <ui16> -> <ui16>
                            %292 = ascvec.reinterpret %291 : <ui16> -> <bf16>
                            %293 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %294 = ascvec.and %293, %289, %276 : <ui16>, <ui16> -> <ui16>
                            %295 = ascvec.reinterpret %294 : <ui16> -> <bf16>
                            %296 = ascvec.cast %292, %276 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %297 = ascvec.cast %292, %276 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %298 = ascvec.cast %295, %276 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %299 = ascvec.cast %295, %276 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %300 = arith.muli %arg15, %c64_i32 : i32
                            %301 = arith.index_cast %300 : i32 to index
                            %302 = ascvec.loadalign %arg9[%301] : memref<8192xi32, 1>, index -> <i32>
                            %303 = ascvec.reinterpret %302 : <i32> -> <ui32>
                            %304 = ascvec.shl %303, 8, %275 : <ui32> -> <ui32>
                            %305 = ascvec.shr %303, 24, %275 : <ui32> -> <ui32>
                            %306 = ascvec.or %304, %305, %275 : <ui32>, <ui32> -> <ui32>
                            %307 = ascvec.shr %303, 16, %275 : <ui32> -> <ui32>
                            %308 = ascvec.and %306, %286, %275 : <ui32>, <ui32> -> <ui32>
                            %309 = ascvec.and %303, %286, %275 : <ui32>, <ui32> -> <ui32>
                            %310 = ascvec.shr %306, 16, %275 : <ui32> -> <ui32>
                            %311 = ascvec.reinterpret %296 : <f32> -> <ui32>
                            %312 = ascvec.reinterpret %311 : <ui32> -> <f32>
                            %313 = ascvec.max_scalar %312, %cst, %275 : <f32>, f32
                            %314 = ascvec.add %312, %313, %275 : <f32>
                            %315 = ascvec.reinterpret %314 : <f32> -> <ui32>
                            %316 = ascvec.shr %315, 5, %275 : <ui32> -> <ui32>
                            %317 = ascvec.add %316, %307, %275 : <ui32>
                            %318 = ascvec.reinterpret %298 : <f32> -> <ui32>
                            %319 = ascvec.reinterpret %318 : <ui32> -> <f32>
                            %320 = ascvec.max_scalar %319, %cst, %275 : <f32>, f32
                            %321 = ascvec.add %319, %320, %275 : <f32>
                            %322 = ascvec.reinterpret %321 : <f32> -> <ui32>
                            %323 = ascvec.shr %322, 5, %275 : <ui32> -> <ui32>
                            %324 = ascvec.add %323, %308, %275 : <ui32>
                            %325 = ascvec.reinterpret %297 : <f32> -> <ui32>
                            %326 = ascvec.reinterpret %325 : <ui32> -> <f32>
                            %327 = ascvec.max_scalar %326, %cst, %275 : <f32>, f32
                            %328 = ascvec.add %326, %327, %275 : <f32>
                            %329 = ascvec.reinterpret %328 : <f32> -> <ui32>
                            %330 = ascvec.shr %329, 5, %275 : <ui32> -> <ui32>
                            %331 = ascvec.add %330, %309, %275 : <ui32>
                            %332 = ascvec.reinterpret %299 : <f32> -> <ui32>
                            %333 = ascvec.reinterpret %332 : <ui32> -> <f32>
                            %334 = ascvec.max_scalar %333, %cst, %275 : <f32>, f32
                            %335 = ascvec.add %333, %334, %275 : <f32>
                            %336 = ascvec.reinterpret %335 : <f32> -> <ui32>
                            %337 = ascvec.shr %336, 5, %275 : <ui32> -> <ui32>
                            %338 = ascvec.add %337, %310, %275 : <ui32>
                            %339 = ascvec.reinterpret %317 : <ui32> -> <ui16>
                            %340 = ascvec.reinterpret %324 : <ui32> -> <ui16>
                            %result0_12, %result1_13 = ascvec.deintlv %339, %340 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %341 = ascvec.reinterpret %331 : <ui32> -> <ui16>
                            %342 = ascvec.reinterpret %338 : <ui32> -> <ui16>
                            %result0_14, %result1_15 = ascvec.deintlv %341, %342 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %343 = ascvec.reinterpret %result1_13 : <ui16> -> <ui8>
                            %344 = ascvec.reinterpret %result1_15 : <ui16> -> <ui8>
                            %result0_16, %result1_17 = ascvec.deintlv %343, %344 : <ui8>, <ui8> -> <ui8>, <ui8>
                            %345 = ascvec.gather_reg %result0_16, %285 : <ui8>, <ui8> -> <ui8>
                            %346 = ascvec.add_scalar %345, %c60_i8, %277 : <ui8>, i8
                            %347 = ascvec.cmp_ne_scalar %346, %c-1_i8, %277 : <ui8>, i8 -> !ascvec.mask
                            %348 = ascvec.min_scalar %346, %c123_i8, %347 : <ui8>, i8
                            %349 = arith.muli %arg15, %c512_i32 : i32
                            %350 = arith.index_cast %349 : i32 to index
                            %result0_18, %result1_19 = ascvec.loadalign_deintlv %arg10[%350] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %351 = ascvec.and %result1_19, %279, %277 : <ui8>, <ui8> -> <ui8>
                            %352 = ascvec.or %348, %351, %277 : <ui8>, <ui8> -> <ui8>
                            %353 = ascvec.arange %c0_i16 : i16 -> <ui16>
                            %354 = arith.muli %arg15, %c16_i32 : i32
                            %355 = arith.trunci %354 : i32 to i16
                            %356 = ascvec.add_scalar %353, %355, %276 : <ui16>, i16
                            %357 = ascvec.gather %arg11[%c0_11], %356, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                            %358 = ascvec.pack %357 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                            %359 = ascvec.gather_reg %358, %280 : <ui8>, <ui8> -> <ui8>
                            %360 = ascvec.cmp_eq_scalar %359, %c-1_i8, %277 : <ui8>, i8 -> !ascvec.mask
                            %361 = ascvec.duplicate_scalar %c127_i8, %360 merge %352 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                            ascvec.storealign %arg12[%288], %361, %277 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
                      }
                    }
                  }
                }
              } else {
                %203 = arith.divsi %66, %c256_i64 : i64
                %204 = arith.addi %66, %c1023_i64 : i64
                %205 = arith.divsi %204, %c1024_i64 : i64
                %206 = arith.trunci %1 : i64 to i32
                %207 = arith.extui %206 : i32 to i64
                %208 = arith.subi %1, %207 : i64
                %209 = arith.remsi %208, %c4294967296_i64 : i64
                %210 = arith.cmpi ne, %209, %c0_i64 : i64
                %211 = arith.cmpi slt, %209, %c0_i64 : i64
                %212 = arith.andi %210, %211 : i1
                %213 = arith.divsi %208, %c4294967296_i64 : i64
                %214 = arith.extui %212 : i1 to i64
                %215 = arith.subi %213, %214 : i64
                %216 = arith.trunci %215 : i64 to i32
                %217 = arith.divui %108, %c16_i64 : i64
                %218 = arith.trunci %217 : i64 to i32
                %219 = arith.extui %218 : i32 to i64
                %220 = arith.subi %217, %219 : i64
                %221 = arith.remsi %220, %c4294967296_i64 : i64
                %222 = arith.cmpi ne, %221, %c0_i64 : i64
                %223 = arith.cmpi slt, %221, %c0_i64 : i64
                %224 = arith.andi %222, %223 : i1
                %225 = arith.divsi %220, %c4294967296_i64 : i64
                %226 = arith.extui %224 : i1 to i64
                %227 = arith.subi %225, %226 : i64
                %228 = arith.trunci %227 : i64 to i32
                %229 = arith.trunci %3 : i64 to i32
                %230 = arith.extui %229 : i32 to i64
                %231 = arith.subi %3, %230 : i64
                %232 = arith.remsi %231, %c4294967296_i64 : i64
                %233 = arith.cmpi ne, %232, %c0_i64 : i64
                %234 = arith.cmpi slt, %232, %c0_i64 : i64
                %235 = arith.andi %233, %234 : i1
                %236 = arith.divsi %231, %c4294967296_i64 : i64
                %237 = arith.extui %235 : i1 to i64
                %238 = arith.subi %236, %237 : i64
                %239 = arith.trunci %238 : i64 to i32
                %240 = arith.divui %229, %c4_i32 : i32
                %241 = arith.remui %239, %c4_i32 : i32
                %242 = arith.muli %241, %c1073741824_i32 : i32
                %243 = arith.addi %240, %242 : i32
                %244 = arith.divui %239, %c4_i32 : i32
                %245 = arith.muli %205, %c64_i64 : i64
                %246 = arith.divui %217, %c16384_i64 : i64
                %247 = arith.addi %217, %245 : i64
                %248 = arith.subi %247, %c1_i64 : i64
                %249 = arith.divui %248, %c16384_i64 : i64
                %250 = arith.cmpi eq, %246, %249 : i64
                scf.if %250 {
                  %256 = arith.trunci %246 : i64 to i32
                  %257 = arith.extui %256 : i32 to i64
                  %258 = arith.subi %246, %257 : i64
                  %259 = arith.remsi %258, %c4294967296_i64 : i64
                  %260 = arith.cmpi ne, %259, %c0_i64 : i64
                  %261 = arith.cmpi slt, %259, %c0_i64 : i64
                  %262 = arith.andi %260, %261 : i1
                  %263 = arith.divsi %258, %c4294967296_i64 : i64
                  %264 = arith.extui %262 : i1 to i64
                  %265 = arith.subi %263, %264 : i64
                  %266 = arith.trunci %265 : i64 to i32
                  %267 = arith.addi %243, %256 : i32
                  %268 = arith.addi %244, %266 : i32
                  %269 = arith.cmpi ult, %267, %243 : i32
                  %270 = arith.extui %269 : i1 to i32
                  %271 = arith.addi %268, %270 : i32
                  %272 = arith.extui %267 : i32 to i64
                  %273 = arith.muli %272, %c3449720151_i64 : i64
                  %274 = arith.trunci %273 : i64 to i32
                  %275 = arith.extui %274 : i32 to i64
                  %276 = arith.subi %273, %275 : i64
                  %277 = arith.remsi %276, %c4294967296_i64 : i64
                  %278 = arith.cmpi ne, %277, %c0_i64 : i64
                  %279 = arith.cmpi slt, %277, %c0_i64 : i64
                  %280 = arith.andi %278, %279 : i1
                  %281 = arith.divsi %276, %c4294967296_i64 : i64
                  %282 = arith.extui %280 : i1 to i64
                  %283 = arith.subi %281, %282 : i64
                  %284 = arith.trunci %283 : i64 to i32
                  %285 = arith.xori %271, %216 : i32
                  %286 = arith.extui %285 : i32 to i64
                  %287 = arith.muli %286, %c3449720151_i64 : i64
                  %288 = arith.trunci %287 : i64 to i32
                  %289 = arith.extui %288 : i32 to i64
                  %290 = arith.subi %287, %289 : i64
                  %291 = arith.remsi %290, %c4294967296_i64 : i64
                  %292 = arith.cmpi ne, %291, %c0_i64 : i64
                  %293 = arith.cmpi slt, %291, %c0_i64 : i64
                  %294 = arith.andi %292, %293 : i1
                  %295 = arith.divsi %290, %c4294967296_i64 : i64
                  %296 = arith.extui %294 : i1 to i64
                  %297 = arith.subi %295, %296 : i64
                  %298 = arith.trunci %297 : i64 to i32
                  %299 = arith.addi %206, %c-1640531527_i32 : i32
                  %300 = arith.addi %216, %c-1150833019_i32 : i32
                  %301 = arith.xori %298, %274 : i32
                  %302 = arith.xori %301, %299 : i32
                  %303 = arith.extui %302 : i32 to i64
                  %304 = arith.muli %303, %c3528531795_i64 : i64
                  %305 = arith.trunci %304 : i64 to i32
                  %306 = arith.extui %305 : i32 to i64
                  %307 = arith.subi %304, %306 : i64
                  %308 = arith.remsi %307, %c4294967296_i64 : i64
                  %309 = arith.cmpi ne, %308, %c0_i64 : i64
                  %310 = arith.cmpi slt, %308, %c0_i64 : i64
                  %311 = arith.andi %309, %310 : i1
                  %312 = arith.divsi %307, %c4294967296_i64 : i64
                  %313 = arith.extui %311 : i1 to i64
                  %314 = arith.subi %312, %313 : i64
                  %315 = arith.trunci %314 : i64 to i32
                  %316 = arith.xori %284, %206 : i32
                  %317 = arith.addi %206, %c1013904242_i32 : i32
                  %318 = arith.xori %288, %317 : i32
                  %319 = arith.addi %216, %c1993301258_i32 : i32
                  %320 = arith.xori %315, %319 : i32
                  ascsimt.vf(%218, %316, %300, %318, %320, %206, %216, %305, %alloc_8, %245 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
                    %c256_i64_10 = arith.constant 256 : i64
                    %c3_i64_11 = arith.constant 3 : i64
                    %c2_i64_12 = arith.constant 2 : i64
                    %c1_i64_13 = arith.constant 1 : i64
                    %c4_i64_14 = arith.constant 4 : i64
                    %c1_i32_15 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c3_i32 = arith.constant 3 : i32
                    %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %321 = ascsimt.thread_idx {ascsimt.scope = 21 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %322 = arith.extsi %321 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg17 = %322 to %arg16 step %c256_i64_10  : i64 {
                      %323 = arith.trunci %arg17 : i64 to i32
                      %324 = arith.addi %arg7, %323 : i32
                      %325 = arith.remui %324, %c16384_i32 : i32
                      %326 = arith.xori %325, %arg8 : i32
                      %327 = arith.muli %326, %c-766435501_i32 : i32
                      %328 = "ascsimt.bitcast"(%326) : (i32) -> ui32
                      %329 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %330 = ascsimt.math_binary %328, %329, "mulhi" : ui32, ui32 -> ui32
                      %331 = "ascsimt.bitcast"(%330) : (ui32) -> i32
                      %332 = arith.xori %331, %arg9 : i32
                      %333 = arith.muli %332, %c-845247145_i32 : i32
                      %334 = "ascsimt.bitcast"(%332) : (i32) -> ui32
                      %335 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %336 = ascsimt.math_binary %334, %335, "mulhi" : ui32, ui32 -> ui32
                      %337 = "ascsimt.bitcast"(%336) : (ui32) -> i32
                      %338 = arith.xori %337, %arg10 : i32
                      %339 = arith.xori %327, %arg11 : i32
                      %340:4 = scf.for %arg18 = %c3_i32 to %c10_i32 step %c1_i32_15 iter_args(%arg19 = %338, %arg20 = %333, %arg21 = %339, %arg22 = %arg14) -> (i32, i32, i32, i32)  : i32 {
                        %349 = arith.muli %arg19, %c-766435501_i32 : i32
                        %350 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %351 = ascsimt.math_binary %350, %329, "mulhi" : ui32, ui32 -> ui32
                        %352 = "ascsimt.bitcast"(%351) : (ui32) -> i32
                        %353 = arith.muli %arg21, %c-845247145_i32 : i32
                        %354 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %355 = ascsimt.math_binary %354, %335, "mulhi" : ui32, ui32 -> ui32
                        %356 = "ascsimt.bitcast"(%355) : (ui32) -> i32
                        %357 = arith.muli %arg18, %c-1640531527_i32_17 : i32
                        %358 = arith.addi %arg12, %357 : i32
                        %359 = arith.muli %arg18, %c-1150833019_i32_16 : i32
                        %360 = arith.addi %arg13, %359 : i32
                        %361 = arith.xori %356, %arg20 : i32
                        %362 = arith.xori %361, %358 : i32
                        %363 = arith.xori %352, %arg22 : i32
                        %364 = arith.xori %363, %360 : i32
                        scf.yield %362, %353, %364, %349 : i32, i32, i32, i32
                      }
                      %341 = arith.muli %arg17, %c4_i64_14 : i64
                      %342 = arith.index_cast %341 : i64 to index
                      ascsimt.store %arg15[%342], %340#0 : memref<8192xi32, 1>, i32
                      %343 = arith.addi %341, %c1_i64_13 : i64
                      %344 = arith.index_cast %343 : i64 to index
                      ascsimt.store %arg15[%344], %340#1 : memref<8192xi32, 1>, i32
                      %345 = arith.addi %341, %c2_i64_12 : i64
                      %346 = arith.index_cast %345 : i64 to index
                      ascsimt.store %arg15[%346], %340#2 : memref<8192xi32, 1>, i32
                      %347 = arith.addi %341, %c3_i64_11 : i64
                      %348 = arith.index_cast %347 : i64 to index
                      ascsimt.store %arg15[%348], %340#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%218, %228, %243, %244, %206, %216, %alloc_8, %245 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: memref<8192xi32, 1>, %arg14: i64):
                    %c256_i64_10 = arith.constant 256 : i64
                    %c3_i64_11 = arith.constant 3 : i64
                    %c2_i64_12 = arith.constant 2 : i64
                    %c1_i64_13 = arith.constant 1 : i64
                    %c4_i64_14 = arith.constant 4 : i64
                    %c1_i32_15 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c0_i32_16 = arith.constant 0 : i32
                    %c-1150833019_i32_17 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_18 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c262144_i32 = arith.constant 262144 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %256 = ascsimt.thread_idx {ascsimt.scope = 22 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %257 = arith.extsi %256 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg15 = %257 to %arg14 step %c256_i64_10  : i64 {
                      %258 = arith.trunci %arg15 : i64 to i32
                      %259 = arith.addi %arg7, %258 : i32
                      %260 = arith.cmpi ult, %259, %arg7 : i32
                      %261 = arith.extui %260 : i1 to i32
                      %262 = arith.addi %arg8, %261 : i32
                      %263 = arith.divui %259, %c16384_i32 : i32
                      %264 = arith.remui %262, %c16384_i32 : i32
                      %265 = arith.muli %264, %c262144_i32 : i32
                      %266 = arith.addi %263, %265 : i32
                      %267 = arith.divui %262, %c16384_i32 : i32
                      %268 = arith.addi %arg9, %266 : i32
                      %269 = arith.addi %arg10, %267 : i32
                      %270 = arith.cmpi ult, %268, %arg9 : i32
                      %271 = arith.extui %270 : i1 to i32
                      %272 = arith.addi %269, %271 : i32
                      %273 = arith.remui %259, %c16384_i32 : i32
                      %274:4 = scf.for %arg16 = %c0_i32_16 to %c10_i32 step %c1_i32_15 iter_args(%arg17 = %c0_i32_16, %arg18 = %273, %arg19 = %268, %arg20 = %272) -> (i32, i32, i32, i32)  : i32 {
                        %283 = arith.muli %arg17, %c-766435501_i32 : i32
                        %284 = "ascsimt.bitcast"(%arg17) : (i32) -> ui32
                        %285 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %286 = ascsimt.math_binary %284, %285, "mulhi" : ui32, ui32 -> ui32
                        %287 = "ascsimt.bitcast"(%286) : (ui32) -> i32
                        %288 = arith.muli %arg19, %c-845247145_i32 : i32
                        %289 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %290 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %291 = ascsimt.math_binary %289, %290, "mulhi" : ui32, ui32 -> ui32
                        %292 = "ascsimt.bitcast"(%291) : (ui32) -> i32
                        %293 = arith.muli %arg16, %c-1640531527_i32_18 : i32
                        %294 = arith.addi %arg11, %293 : i32
                        %295 = arith.muli %arg16, %c-1150833019_i32_17 : i32
                        %296 = arith.addi %arg12, %295 : i32
                        %297 = arith.xori %292, %arg18 : i32
                        %298 = arith.xori %297, %294 : i32
                        %299 = arith.xori %287, %arg20 : i32
                        %300 = arith.xori %299, %296 : i32
                        scf.yield %298, %288, %300, %283 : i32, i32, i32, i32
                      }
                      %275 = arith.muli %arg15, %c4_i64_14 : i64
                      %276 = arith.index_cast %275 : i64 to index
                      ascsimt.store %arg13[%276], %274#0 : memref<8192xi32, 1>, i32
                      %277 = arith.addi %275, %c1_i64_13 : i64
                      %278 = arith.index_cast %277 : i64 to index
                      ascsimt.store %arg13[%278], %274#1 : memref<8192xi32, 1>, i32
                      %279 = arith.addi %275, %c2_i64_12 : i64
                      %280 = arith.index_cast %279 : i64 to index
                      ascsimt.store %arg13[%280], %274#2 : memref<8192xi32, 1>, i32
                      %281 = arith.addi %275, %c3_i64_11 : i64
                      %282 = arith.index_cast %281 : i64 to index
                      ascsimt.store %arg13[%282], %274#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %251 = arith.trunci %48 : i64 to i32
                %252 = arith.divsi %251, %c2_i32 : i32
                %253 = arith.trunci %252 : i32 to i8
                %254 = arith.trunci %203 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                %255 = ascscalar.bitcast_memref %107 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                ascvec.vf(%253, %alloc_6, %107, %255, %alloc_8, %alloc_5, %254 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui8, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, i32) {
                ^bb0(%arg7: i8, %arg8: memref<2048xui8, 1>, %arg9: memref<32768xbf16, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<8192xi32, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32):
                  %c512_i32 = arith.constant 512 : i32
                  %c1_i32_10 = arith.constant 1 : i32
                  %c2139062143_i32 = arith.constant 2139062143 : i32
                  %c-8323200_i32 = arith.constant -8323200 : i32
                  %c123_i8 = arith.constant 123 : i8
                  %c-1_i8 = arith.constant -1 : i8
                  %c60_i8 = arith.constant 60 : i8
                  %cst = arith.constant 6.10351526E-5 : f32
                  %c64_i32 = arith.constant 64 : i32
                  %c0_i16 = arith.constant 0 : i16
                  %c0_i32_11 = arith.constant 0 : i32
                  %c254_i32 = arith.constant 254 : i32
                  %c-15_i32 = arith.constant -15 : i32
                  %c15_i32 = arith.constant 15 : i32
                  %c255_i32 = arith.constant 255 : i32
                  %c32767_i16 = arith.constant 32767 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c0_12 = arith.constant 0 : index
                  %c32_i8 = arith.constant 32 : i8
                  %c65535_i32 = arith.constant 65535 : i32
                  %c3_i8 = arith.constant 3 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c-128_i8 = arith.constant -128 : i8
                  %256 = ascvec.create_mask : !ascvec.mask
                  %257 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %258 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %259 = ascvec.duplicate_scalar %c-128_i8, %258 : i8 -> <ui8>
                  %260 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %261 = ascvec.duplicate_scalar %c3_i8, %258 : i8 -> <ui8>
                  %262 = ascvec.and %260, %261, %258 : <ui8>, <ui8> -> <ui8>
                  %263 = ascvec.shl %262, 6, %258 : <ui8> -> <ui8>
                  %264 = ascvec.shr %260, 2, %258 : <ui8> -> <ui8>
                  %265 = ascvec.or %263, %264, %258 : <ui8>, <ui8> -> <ui8>
                  %266 = ascvec.duplicate_scalar %c65535_i32, %256 : i32 -> <ui32>
                  %267 = ascvec.cmp_lt_scalar %260, %c32_i8, %258 : <ui8>, i8 -> !ascvec.mask
                  %268 = ascvec.cmp_lt_scalar %262, %arg7, %258 : <ui8>, i8 -> !ascvec.mask
                  %269 = ascvec.mask_and %267, %268, %258
                  ascvec.store_unalign_begin %arg8 : memref<2048xui8, 1>
                  scf.for %arg14 = %c0_i32_11 to %arg13 step %c1_i32_10  : i32 {
                    %270 = arith.muli %arg14, %c256_i32 : i32
                    %271 = arith.index_cast %270 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg9[%271] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %272 = ascvec.duplicate_scalar %c32767_i16, %257 : i16 -> <ui16>
                    %273 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %274 = ascvec.and %273, %272, %257 : <ui16>, <ui16> -> <ui16>
                    %275 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %276 = ascvec.and %275, %272, %257 : <ui16>, <ui16> -> <ui16>
                    %277 = ascvec.max %274, %276, %257 : <ui16>
                    %278 = ascvec.reduce_max_datablock %277, %257 : <ui16>
                    %279 = ascvec.unpack %278 {part = "lower"} : <ui16> -> <ui32>
                    %280 = ascvec.shl %279, 16, %256 : <ui32> -> <ui32>
                    %281 = arith.muli %arg14, %c512_i32 : i32
                    %282 = arith.index_cast %281 : i32 to index
                    %result0_15, %result1_16 = ascvec.loadalign_deintlv %arg10[%282] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                    %283 = ascvec.and %result1_16, %259, %258 : <ui8>, <ui8> -> <ui8>
                    %284 = ascvec.reinterpret %283 : <ui8> -> <ui32>
                    %285 = ascvec.shr %280, 23, %256 : <ui32> -> <ui32>
                    %286 = ascvec.cmp_ge_scalar %285, %c255_i32, %256 : <ui32>, i32 -> !ascvec.mask
                    %287 = ascvec.max_scalar %285, %c15_i32, %256 : <ui32>, i32
                    %288 = ascvec.add_scalar %287, %c-15_i32, %256 : <ui32>, i32
                    %289 = ascvec.duplicate_scalar %c255_i32, %286 merge %288 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                    %290 = ascvec.duplicate_scalar %c254_i32, %256 : i32 -> <ui32>
                    %291 = ascvec.sub %290, %289, %256 : <ui32>
                    %292 = ascvec.cmp_ne_scalar %289, %c0_i32_11, %256 : <ui32>, i32 -> !ascvec.mask
                    %293 = ascvec.shl %291, 23, %292 : <ui32> -> <ui32>
                    %294 = ascvec.reinterpret %293 : <ui32> -> <f32>
                    %295 = ascvec.reinterpret %294 : <f32> -> <ui32>
                    %296 = ascvec.shr %295, 16, %256 : <ui32> -> <ui32>
                    %297 = ascvec.pack %296 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                    %298 = ascvec.arange %c0_i16 : i16 -> <ui16>
                    %299 = ascvec.shr %298, 4, %257 : <ui16> -> <ui16>
                    %300 = ascvec.gather_reg %297, %299 : <ui16>, <ui16> -> <ui16>
                    %301 = ascvec.reinterpret %300 : <ui16> -> <bf16>
                    %302 = ascvec.reinterpret %274 : <ui16> -> <bf16>
                    %303 = ascvec.mul %302, %301, %257 : <bf16>
                    %304 = ascvec.reinterpret %276 : <ui16> -> <bf16>
                    %305 = ascvec.mul %304, %301, %257 : <bf16>
                    %306 = ascvec.cast %303, %257 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %307 = ascvec.cast %303, %257 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %308 = ascvec.cast %305, %257 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %309 = ascvec.cast %305, %257 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %310 = ascvec.reinterpret %301 : <bf16> -> <ui32>
                    %311 = arith.muli %arg14, %c64_i32 : i32
                    %312 = arith.index_cast %311 : i32 to index
                    %313 = ascvec.loadalign %arg11[%312] : memref<8192xi32, 1>, index -> <i32>
                    %314 = ascvec.reinterpret %313 : <i32> -> <ui32>
                    %315 = ascvec.shl %314, 8, %256 : <ui32> -> <ui32>
                    %316 = ascvec.shr %314, 24, %256 : <ui32> -> <ui32>
                    %317 = ascvec.or %315, %316, %256 : <ui32>, <ui32> -> <ui32>
                    %318 = ascvec.shr %314, 16, %256 : <ui32> -> <ui32>
                    %319 = ascvec.and %317, %266, %256 : <ui32>, <ui32> -> <ui32>
                    %320 = ascvec.and %314, %266, %256 : <ui32>, <ui32> -> <ui32>
                    %321 = ascvec.shr %317, 16, %256 : <ui32> -> <ui32>
                    %322 = ascvec.reinterpret %306 : <f32> -> <ui32>
                    %323 = ascvec.reinterpret %322 : <ui32> -> <f32>
                    %324 = ascvec.max_scalar %323, %cst, %256 : <f32>, f32
                    %325 = ascvec.add %323, %324, %256 : <f32>
                    %326 = ascvec.reinterpret %325 : <f32> -> <ui32>
                    %327 = ascvec.shr %326, 5, %256 : <ui32> -> <ui32>
                    %328 = ascvec.add %327, %318, %256 : <ui32>
                    %329 = ascvec.reinterpret %308 : <f32> -> <ui32>
                    %330 = ascvec.reinterpret %329 : <ui32> -> <f32>
                    %331 = ascvec.max_scalar %330, %cst, %256 : <f32>, f32
                    %332 = ascvec.add %330, %331, %256 : <f32>
                    %333 = ascvec.reinterpret %332 : <f32> -> <ui32>
                    %334 = ascvec.shr %333, 5, %256 : <ui32> -> <ui32>
                    %335 = ascvec.add %334, %319, %256 : <ui32>
                    %336 = ascvec.reinterpret %307 : <f32> -> <ui32>
                    %337 = ascvec.reinterpret %336 : <ui32> -> <f32>
                    %338 = ascvec.max_scalar %337, %cst, %256 : <f32>, f32
                    %339 = ascvec.add %337, %338, %256 : <f32>
                    %340 = ascvec.reinterpret %339 : <f32> -> <ui32>
                    %341 = ascvec.shr %340, 5, %256 : <ui32> -> <ui32>
                    %342 = ascvec.add %341, %320, %256 : <ui32>
                    %343 = ascvec.reinterpret %309 : <f32> -> <ui32>
                    %344 = ascvec.reinterpret %343 : <ui32> -> <f32>
                    %345 = ascvec.max_scalar %344, %cst, %256 : <f32>, f32
                    %346 = ascvec.add %344, %345, %256 : <f32>
                    %347 = ascvec.reinterpret %346 : <f32> -> <ui32>
                    %348 = ascvec.shr %347, 5, %256 : <ui32> -> <ui32>
                    %349 = ascvec.add %348, %321, %256 : <ui32>
                    %350 = ascvec.reinterpret %328 : <ui32> -> <ui16>
                    %351 = ascvec.reinterpret %335 : <ui32> -> <ui16>
                    %result0_17, %result1_18 = ascvec.deintlv %350, %351 : <ui16>, <ui16> -> <ui16>, <ui16>
                    %352 = ascvec.reinterpret %342 : <ui32> -> <ui16>
                    %353 = ascvec.reinterpret %349 : <ui32> -> <ui16>
                    %result0_19, %result1_20 = ascvec.deintlv %352, %353 : <ui16>, <ui16> -> <ui16>, <ui16>
                    %354 = ascvec.reinterpret %result1_18 : <ui16> -> <ui8>
                    %355 = ascvec.reinterpret %result1_20 : <ui16> -> <ui8>
                    %result0_21, %result1_22 = ascvec.deintlv %354, %355 : <ui8>, <ui8> -> <ui8>, <ui8>
                    %356 = ascvec.gather_reg %result0_21, %265 : <ui8>, <ui8> -> <ui8>
                    %357 = ascvec.reinterpret %356 : <ui8> -> <ui32>
                    %358 = ascvec.reinterpret %357 : <ui32> -> <ui8>
                    %359 = ascvec.add_scalar %358, %c60_i8, %258 : <ui8>, i8
                    %360 = ascvec.cmp_ne_scalar %359, %c-1_i8, %258 : <ui8>, i8 -> !ascvec.mask
                    %361 = ascvec.min_scalar %359, %c123_i8, %360 : <ui8>, i8
                    %362 = ascvec.reinterpret %361 : <ui8> -> <ui32>
                    %363 = ascvec.or %362, %284, %256 : <ui32>, <ui32> -> <ui32>
                    %364 = ascvec.reinterpret %310 : <ui32> -> <ui32>
                    %365 = ascvec.cmp_eq_scalar %364, %c-8323200_i32, %256 : <ui32>, i32 -> !ascvec.mask
                    %366 = ascvec.duplicate_scalar %c2139062143_i32, %365 merge %363 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                    %367 = ascvec.reinterpret %366 : <ui32> -> <ui8>
                    ascvec.storealign %arg12[%271], %367, %258 : memref<32768xui8, 1>, index, <ui8>
                    %368 = ascvec.reinterpret %289 : <ui32> -> <ui8>
                    %369 = ascvec.squeeze_store_reg %368, %269 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_12], %369 : memref<2048xui8, 1>, <ui8>
                  }
                  %c0_13 = arith.constant 0 : index
                  %c1_14 = arith.constant 1 : index
                  scf.for %arg14 = %c0_13 to %c1_14 step %c1_14 {
                    ascvec.store_unalign_post %arg8[%c0_12] : memref<2048xui8, 1>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              }
            }
          } else {
            %192 = arith.remsi %arg3, %c256_i64 : i64
            %193 = arith.cmpi ne, %192, %c0_i64 : i64
            %194 = arith.cmpi slt, %192, %c0_i64 : i64
            %195 = arith.andi %193, %194 : i1
            %196 = arith.select %195, %c256_i64, %c0_i64 : i64
            %197 = arith.addi %192, %196 overflow<nsw> : i64
            %198 = arith.cmpi eq, %197, %c0_i64 : i64
            scf.if %198 {
              %199 = arith.addi %66, %c63_i64 : i64
              %200 = arith.remsi %199, %c64_i64 : i64
              %201 = arith.cmpi ne, %200, %c0_i64 : i64
              %202 = arith.cmpi slt, %200, %c0_i64 : i64
              %203 = arith.andi %201, %202 : i1
              %204 = arith.divsi %199, %c64_i64 : i64
              %205 = arith.extui %203 : i1 to i64
              %206 = arith.subi %204, %205 : i64
              %207 = arith.addi %66, %c1023_i64 : i64
              %208 = arith.remsi %207, %c1024_i64 : i64
              %209 = arith.cmpi ne, %208, %c0_i64 : i64
              %210 = arith.cmpi slt, %208, %c0_i64 : i64
              %211 = arith.andi %209, %210 : i1
              %212 = arith.divsi %207, %c1024_i64 : i64
              %213 = arith.extui %211 : i1 to i64
              %214 = arith.subi %212, %213 : i64
              %215 = arith.trunci %1 : i64 to i32
              %216 = arith.extui %215 : i32 to i64
              %217 = arith.subi %1, %216 : i64
              %218 = arith.remsi %217, %c4294967296_i64 : i64
              %219 = arith.cmpi ne, %218, %c0_i64 : i64
              %220 = arith.cmpi slt, %218, %c0_i64 : i64
              %221 = arith.andi %219, %220 : i1
              %222 = arith.divsi %217, %c4294967296_i64 : i64
              %223 = arith.extui %221 : i1 to i64
              %224 = arith.subi %222, %223 : i64
              %225 = arith.trunci %224 : i64 to i32
              %226 = arith.divui %108, %c32_i64 : i64
              %227 = arith.trunci %226 : i64 to i32
              %228 = arith.extui %227 : i32 to i64
              %229 = arith.subi %226, %228 : i64
              %230 = arith.remsi %229, %c4294967296_i64 : i64
              %231 = arith.cmpi ne, %230, %c0_i64 : i64
              %232 = arith.cmpi slt, %230, %c0_i64 : i64
              %233 = arith.andi %231, %232 : i1
              %234 = arith.divsi %229, %c4294967296_i64 : i64
              %235 = arith.extui %233 : i1 to i64
              %236 = arith.subi %234, %235 : i64
              %237 = arith.trunci %236 : i64 to i32
              %238 = arith.trunci %3 : i64 to i32
              %239 = arith.extui %238 : i32 to i64
              %240 = arith.subi %3, %239 : i64
              %241 = arith.remsi %240, %c4294967296_i64 : i64
              %242 = arith.cmpi ne, %241, %c0_i64 : i64
              %243 = arith.cmpi slt, %241, %c0_i64 : i64
              %244 = arith.andi %242, %243 : i1
              %245 = arith.divsi %240, %c4294967296_i64 : i64
              %246 = arith.extui %244 : i1 to i64
              %247 = arith.subi %245, %246 : i64
              %248 = arith.trunci %247 : i64 to i32
              %249 = arith.divui %238, %c4_i32 : i32
              %250 = arith.remui %248, %c4_i32 : i32
              %251 = arith.muli %250, %c1073741824_i32 : i32
              %252 = arith.addi %249, %251 : i32
              %253 = arith.divui %248, %c4_i32 : i32
              %254 = arith.muli %214, %c32_i64 : i64
              %255 = arith.divui %226, %c16384_i64 : i64
              %256 = arith.addi %226, %254 : i64
              %257 = arith.subi %256, %c1_i64 : i64
              %258 = arith.divui %257, %c16384_i64 : i64
              %259 = arith.cmpi eq, %255, %258 : i64
              scf.if %259 {
                %265 = arith.trunci %255 : i64 to i32
                %266 = arith.extui %265 : i32 to i64
                %267 = arith.subi %255, %266 : i64
                %268 = arith.remsi %267, %c4294967296_i64 : i64
                %269 = arith.cmpi ne, %268, %c0_i64 : i64
                %270 = arith.cmpi slt, %268, %c0_i64 : i64
                %271 = arith.andi %269, %270 : i1
                %272 = arith.divsi %267, %c4294967296_i64 : i64
                %273 = arith.extui %271 : i1 to i64
                %274 = arith.subi %272, %273 : i64
                %275 = arith.trunci %274 : i64 to i32
                %276 = arith.addi %252, %265 : i32
                %277 = arith.addi %253, %275 : i32
                %278 = arith.cmpi ult, %276, %252 : i32
                %279 = arith.extui %278 : i1 to i32
                %280 = arith.addi %277, %279 : i32
                %281 = arith.extui %276 : i32 to i64
                %282 = arith.muli %281, %c3449720151_i64 : i64
                %283 = arith.trunci %282 : i64 to i32
                %284 = arith.extui %283 : i32 to i64
                %285 = arith.subi %282, %284 : i64
                %286 = arith.remsi %285, %c4294967296_i64 : i64
                %287 = arith.cmpi ne, %286, %c0_i64 : i64
                %288 = arith.cmpi slt, %286, %c0_i64 : i64
                %289 = arith.andi %287, %288 : i1
                %290 = arith.divsi %285, %c4294967296_i64 : i64
                %291 = arith.extui %289 : i1 to i64
                %292 = arith.subi %290, %291 : i64
                %293 = arith.trunci %292 : i64 to i32
                %294 = arith.xori %280, %225 : i32
                %295 = arith.extui %294 : i32 to i64
                %296 = arith.muli %295, %c3449720151_i64 : i64
                %297 = arith.trunci %296 : i64 to i32
                %298 = arith.extui %297 : i32 to i64
                %299 = arith.subi %296, %298 : i64
                %300 = arith.remsi %299, %c4294967296_i64 : i64
                %301 = arith.cmpi ne, %300, %c0_i64 : i64
                %302 = arith.cmpi slt, %300, %c0_i64 : i64
                %303 = arith.andi %301, %302 : i1
                %304 = arith.divsi %299, %c4294967296_i64 : i64
                %305 = arith.extui %303 : i1 to i64
                %306 = arith.subi %304, %305 : i64
                %307 = arith.trunci %306 : i64 to i32
                %308 = arith.addi %215, %c-1640531527_i32 : i32
                %309 = arith.addi %225, %c-1150833019_i32 : i32
                %310 = arith.xori %307, %283 : i32
                %311 = arith.xori %310, %308 : i32
                %312 = arith.extui %311 : i32 to i64
                %313 = arith.muli %312, %c3528531795_i64 : i64
                %314 = arith.trunci %313 : i64 to i32
                %315 = arith.extui %314 : i32 to i64
                %316 = arith.subi %313, %315 : i64
                %317 = arith.remsi %316, %c4294967296_i64 : i64
                %318 = arith.cmpi ne, %317, %c0_i64 : i64
                %319 = arith.cmpi slt, %317, %c0_i64 : i64
                %320 = arith.andi %318, %319 : i1
                %321 = arith.divsi %316, %c4294967296_i64 : i64
                %322 = arith.extui %320 : i1 to i64
                %323 = arith.subi %321, %322 : i64
                %324 = arith.trunci %323 : i64 to i32
                %325 = arith.xori %293, %215 : i32
                %326 = arith.addi %215, %c1013904242_i32 : i32
                %327 = arith.xori %297, %326 : i32
                %328 = arith.addi %225, %c1993301258_i32 : i32
                %329 = arith.xori %324, %328 : i32
                ascsimt.vf(%227, %325, %309, %327, %329, %215, %225, %314, %alloc_8, %254 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
                  %c256_i64_10 = arith.constant 256 : i64
                  %c3_i64_11 = arith.constant 3 : i64
                  %c2_i64_12 = arith.constant 2 : i64
                  %c1_i64_13 = arith.constant 1 : i64
                  %c4_i64_14 = arith.constant 4 : i64
                  %c1_i32_15 = arith.constant 1 : i32
                  %c10_i32 = arith.constant 10 : i32
                  %c3_i32 = arith.constant 3 : i32
                  %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                  %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                  %c-845247145_i32 = arith.constant -845247145 : i32
                  %c-766435501_i32 = arith.constant -766435501 : i32
                  %c16384_i32 = arith.constant 16384 : i32
                  %330 = ascsimt.thread_idx {ascsimt.scope = 24 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                  %331 = arith.extsi %330 {cannir.vf_threads = 256 : i64} : i32 to i64
                  scf.for %arg17 = %331 to %arg16 step %c256_i64_10  : i64 {
                    %332 = arith.trunci %arg17 : i64 to i32
                    %333 = arith.addi %arg7, %332 : i32
                    %334 = arith.remui %333, %c16384_i32 : i32
                    %335 = arith.xori %334, %arg8 : i32
                    %336 = arith.muli %335, %c-766435501_i32 : i32
                    %337 = "ascsimt.bitcast"(%335) : (i32) -> ui32
                    %338 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                    %339 = ascsimt.math_binary %337, %338, "mulhi" : ui32, ui32 -> ui32
                    %340 = "ascsimt.bitcast"(%339) : (ui32) -> i32
                    %341 = arith.xori %340, %arg9 : i32
                    %342 = arith.muli %341, %c-845247145_i32 : i32
                    %343 = "ascsimt.bitcast"(%341) : (i32) -> ui32
                    %344 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                    %345 = ascsimt.math_binary %343, %344, "mulhi" : ui32, ui32 -> ui32
                    %346 = "ascsimt.bitcast"(%345) : (ui32) -> i32
                    %347 = arith.xori %346, %arg10 : i32
                    %348 = arith.xori %336, %arg11 : i32
                    %349:4 = scf.for %arg18 = %c3_i32 to %c10_i32 step %c1_i32_15 iter_args(%arg19 = %347, %arg20 = %342, %arg21 = %348, %arg22 = %arg14) -> (i32, i32, i32, i32)  : i32 {
                      %358 = arith.muli %arg19, %c-766435501_i32 : i32
                      %359 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                      %360 = ascsimt.math_binary %359, %338, "mulhi" : ui32, ui32 -> ui32
                      %361 = "ascsimt.bitcast"(%360) : (ui32) -> i32
                      %362 = arith.muli %arg21, %c-845247145_i32 : i32
                      %363 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                      %364 = ascsimt.math_binary %363, %344, "mulhi" : ui32, ui32 -> ui32
                      %365 = "ascsimt.bitcast"(%364) : (ui32) -> i32
                      %366 = arith.muli %arg18, %c-1640531527_i32_17 : i32
                      %367 = arith.addi %arg12, %366 : i32
                      %368 = arith.muli %arg18, %c-1150833019_i32_16 : i32
                      %369 = arith.addi %arg13, %368 : i32
                      %370 = arith.xori %365, %arg20 : i32
                      %371 = arith.xori %370, %367 : i32
                      %372 = arith.xori %361, %arg22 : i32
                      %373 = arith.xori %372, %369 : i32
                      scf.yield %371, %362, %373, %358 : i32, i32, i32, i32
                    }
                    %350 = arith.muli %arg17, %c4_i64_14 : i64
                    %351 = arith.index_cast %350 : i64 to index
                    ascsimt.store %arg15[%351], %349#0 : memref<8192xi32, 1>, i32
                    %352 = arith.addi %350, %c1_i64_13 : i64
                    %353 = arith.index_cast %352 : i64 to index
                    ascsimt.store %arg15[%353], %349#1 : memref<8192xi32, 1>, i32
                    %354 = arith.addi %350, %c2_i64_12 : i64
                    %355 = arith.index_cast %354 : i64 to index
                    ascsimt.store %arg15[%355], %349#2 : memref<8192xi32, 1>, i32
                    %356 = arith.addi %350, %c3_i64_11 : i64
                    %357 = arith.index_cast %356 : i64 to index
                    ascsimt.store %arg15[%357], %349#3 : memref<8192xi32, 1>, i32
                  } {cannir.vf_threads = 256 : i64}
                } {threads = 256 : i64}
              } else {
                ascsimt.vf(%227, %237, %252, %253, %215, %225, %alloc_8, %254 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: memref<8192xi32, 1>, %arg14: i64):
                  %c256_i64_10 = arith.constant 256 : i64
                  %c3_i64_11 = arith.constant 3 : i64
                  %c2_i64_12 = arith.constant 2 : i64
                  %c1_i64_13 = arith.constant 1 : i64
                  %c4_i64_14 = arith.constant 4 : i64
                  %c1_i32_15 = arith.constant 1 : i32
                  %c10_i32 = arith.constant 10 : i32
                  %c0_i32_16 = arith.constant 0 : i32
                  %c-1150833019_i32_17 = arith.constant -1150833019 : i32
                  %c-1640531527_i32_18 = arith.constant -1640531527 : i32
                  %c-845247145_i32 = arith.constant -845247145 : i32
                  %c-766435501_i32 = arith.constant -766435501 : i32
                  %c262144_i32 = arith.constant 262144 : i32
                  %c16384_i32 = arith.constant 16384 : i32
                  %265 = ascsimt.thread_idx {ascsimt.scope = 25 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                  %266 = arith.extsi %265 {cannir.vf_threads = 256 : i64} : i32 to i64
                  scf.for %arg15 = %266 to %arg14 step %c256_i64_10  : i64 {
                    %267 = arith.trunci %arg15 : i64 to i32
                    %268 = arith.addi %arg7, %267 : i32
                    %269 = arith.cmpi ult, %268, %arg7 : i32
                    %270 = arith.extui %269 : i1 to i32
                    %271 = arith.addi %arg8, %270 : i32
                    %272 = arith.divui %268, %c16384_i32 : i32
                    %273 = arith.remui %271, %c16384_i32 : i32
                    %274 = arith.muli %273, %c262144_i32 : i32
                    %275 = arith.addi %272, %274 : i32
                    %276 = arith.divui %271, %c16384_i32 : i32
                    %277 = arith.addi %arg9, %275 : i32
                    %278 = arith.addi %arg10, %276 : i32
                    %279 = arith.cmpi ult, %277, %arg9 : i32
                    %280 = arith.extui %279 : i1 to i32
                    %281 = arith.addi %278, %280 : i32
                    %282 = arith.remui %268, %c16384_i32 : i32
                    %283:4 = scf.for %arg16 = %c0_i32_16 to %c10_i32 step %c1_i32_15 iter_args(%arg17 = %c0_i32_16, %arg18 = %282, %arg19 = %277, %arg20 = %281) -> (i32, i32, i32, i32)  : i32 {
                      %292 = arith.muli %arg17, %c-766435501_i32 : i32
                      %293 = "ascsimt.bitcast"(%arg17) : (i32) -> ui32
                      %294 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %295 = ascsimt.math_binary %293, %294, "mulhi" : ui32, ui32 -> ui32
                      %296 = "ascsimt.bitcast"(%295) : (ui32) -> i32
                      %297 = arith.muli %arg19, %c-845247145_i32 : i32
                      %298 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                      %299 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %300 = ascsimt.math_binary %298, %299, "mulhi" : ui32, ui32 -> ui32
                      %301 = "ascsimt.bitcast"(%300) : (ui32) -> i32
                      %302 = arith.muli %arg16, %c-1640531527_i32_18 : i32
                      %303 = arith.addi %arg11, %302 : i32
                      %304 = arith.muli %arg16, %c-1150833019_i32_17 : i32
                      %305 = arith.addi %arg12, %304 : i32
                      %306 = arith.xori %301, %arg18 : i32
                      %307 = arith.xori %306, %303 : i32
                      %308 = arith.xori %296, %arg20 : i32
                      %309 = arith.xori %308, %305 : i32
                      scf.yield %307, %297, %309, %292 : i32, i32, i32, i32
                    }
                    %284 = arith.muli %arg15, %c4_i64_14 : i64
                    %285 = arith.index_cast %284 : i64 to index
                    ascsimt.store %arg13[%285], %283#0 : memref<8192xi32, 1>, i32
                    %286 = arith.addi %284, %c1_i64_13 : i64
                    %287 = arith.index_cast %286 : i64 to index
                    ascsimt.store %arg13[%287], %283#1 : memref<8192xi32, 1>, i32
                    %288 = arith.addi %284, %c2_i64_12 : i64
                    %289 = arith.index_cast %288 : i64 to index
                    ascsimt.store %arg13[%289], %283#2 : memref<8192xi32, 1>, i32
                    %290 = arith.addi %284, %c3_i64_11 : i64
                    %291 = arith.index_cast %290 : i64 to index
                    ascsimt.store %arg13[%291], %283#3 : memref<8192xi32, 1>, i32
                  } {cannir.vf_threads = 256 : i64}
                } {threads = 256 : i64}
              }
              %260 = arith.trunci %206 : i64 to i32
              %261 = arith.trunci %66 : i64 to i32
              %262 = arith.trunci %48 : i64 to i32
              %263 = arith.muli %260, %c2_i32 : i32
              %264 = arith.cmpi eq, %262, %c4_i32 : i32
              ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%66, %107, %alloc_8, %alloc_5, %alloc_7, %260, %261, %263, %264, %alloc_6 : i64, memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, memref<1024xui32, 1>, i32, i32, i32, i1, memref<2048xui8, 1>) {
              ^bb0(%arg7: i64, %arg8: memref<32768xbf16, 1>, %arg9: memref<8192xi32, 1>, %arg10: memref<32768xui8, 1>, %arg11: memref<1024xui32, 1>, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i1, %arg16: memref<2048xui8, 1>):
                %c2139095040_i32 = arith.constant 2139095040 : i32
                %c127_i32 = arith.constant 127 : i32
                %c123_i32 = arith.constant 123 : i32
                %c947912704_i32 = arith.constant 947912704 : i32
                %cst = arith.constant 4.2949673E+9 : f32
                %c29360128_i32 = arith.constant 29360128 : i32
                %c128_i32 = arith.constant 128 : i32
                %c65535_i32 = arith.constant 65535 : i32
                %c2_i32_10 = arith.constant 2 : i32
                %c7_i32 = arith.constant 7 : i32
                %c1_i32_11 = arith.constant 1 : i32
                %c4_i32_12 = arith.constant 4 : i32
                %c2139160576_i32 = arith.constant 2139160576 : i32
                %c254_i32 = arith.constant 254 : i32
                %c-15_i32 = arith.constant -15 : i32
                %c15_i32 = arith.constant 15 : i32
                %c255_i32 = arith.constant 255 : i32
                %c2147483647_i32 = arith.constant 2147483647 : i32
                %cst_13 = arith.constant 0.000000e+00 : f32
                %c0_14 = arith.constant 0 : index
                %c64_i32 = arith.constant 64 : i32
                %c32_i32 = arith.constant 32 : i32
                %c0_i32_15 = arith.constant 0 : i32
                %265 = ascvec.create_mask : !ascvec.mask
                %266 = ascvec.arange %c0_i32_15 : i32 -> <ui32>
                %267 = ascvec.cmp_lt_scalar %266, %c32_i32, %265 : <ui32>, i32 -> !ascvec.mask
                %268 = ascvec.cmp_ge_scalar %266, %c32_i32, %265 : <ui32>, i32 -> !ascvec.mask
                scf.for %arg17 = %c0_i32_15 to %arg12 step %c1_i32_11  : i32 {
                  %270 = arith.muli %arg17, %c64_i32 : i32
                  %271 = arith.trunci %arg7 : i64 to i32
                  %272 = arith.subi %271, %270 : i32
                  %mask, %remaining_out = ascvec.update_mask %272 : !ascvec.mask
                  %273 = arith.index_cast %270 : i32 to index
                  %274 = ascvec.loadalign_unpack %arg8[%273] {mode = "b16_to_b32"} : memref<32768xbf16, 1>, index -> <bf16>
                  %275 = ascvec.cast %274, %265 {rounding = 6 : i32} : <bf16> -> <f32>
                  %276 = ascvec.duplicate_scalar %cst_13, %265 : f32 -> <f32>
                  %277 = ascvec.select %275, %276, %mask : <f32>
                  %278 = ascvec.reinterpret %277 : <f32> -> <ui32>
                  %279 = ascvec.duplicate_scalar %c2147483647_i32, %265 : i32 -> <ui32>
                  %280 = ascvec.and %278, %279, %265 : <ui32>, <ui32> -> <ui32>
                  %281 = ascvec.reduce_max %280, %267 : <ui32>
                  %282 = ascvec.duplicate %281, %265 : <ui32> -> <ui32>
                  %283 = ascvec.reduce_max %280, %268 : <ui32>
                  %284 = ascvec.duplicate %283, %265 : <ui32> -> <ui32>
                  %285 = ascvec.select %282, %284, %267 : <ui32>
                  %286 = ascvec.shr %285, 23, %265 : <ui32> -> <ui32>
                  %287 = ascvec.cmp_ge_scalar %286, %c255_i32, %265 : <ui32>, i32 -> !ascvec.mask
                  %288 = ascvec.max_scalar %286, %c15_i32, %265 : <ui32>, i32
                  %289 = ascvec.add_scalar %288, %c-15_i32, %265 : <ui32>, i32
                  %290 = ascvec.duplicate_scalar %c255_i32, %287 merge %289 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %291 = ascvec.duplicate_scalar %c254_i32, %265 : i32 -> <ui32>
                  %292 = ascvec.sub %291, %290, %265 : <ui32>
                  %293 = ascvec.cmp_ne_scalar %290, %c0_i32_15, %265 : <ui32>, i32 -> !ascvec.mask
                  %294 = ascvec.shl %292, 23, %293 : <ui32> -> <ui32>
                  %295 = ascvec.duplicate_scalar %c2139160576_i32, %287 merge %294 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %296 = ascvec.reinterpret %295 : <ui32> -> <f32>
                  %297 = ascvec.mul %277, %296, %265 : <f32>
                  %298 = ascvec.cast %297, %265 : <f32> -> <bf16>
                  %299 = ascvec.cast %298, %265 {rounding = 6 : i32} : <bf16> -> <f32>
                  %300 = arith.remsi %arg17, %c4_i32_12 : i32
                  %301 = arith.cmpi ne, %300, %c0_i32_15 : i32
                  %302 = arith.cmpi slt, %300, %c0_i32_15 : i32
                  %303 = arith.andi %301, %302 : i1
                  %304 = arith.select %303, %c4_i32_12, %c0_i32_15 : i32
                  %305 = arith.addi %300, %304 overflow<nsw> : i32
                  %306 = arith.muli %305, %c64_i32 overflow<nsw> : i32
                  %307 = ascvec.add_scalar %266, %306, %265 : <ui32>, i32
                  %308 = arith.divsi %arg17, %c4_i32_12 : i32
                  %309 = arith.extui %303 : i1 to i32
                  %310 = arith.subi %308, %309 : i32
                  %311 = ascvec.shr %307, 3, %265 : <ui32> -> <ui32>
                  %312 = arith.muli %310, %c32_i32 : i32
                  %313 = ascvec.add_scalar %311, %312, %265 : <ui32>, i32
                  %314 = ascvec.gather %arg9[%c0_14], %313, %265 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                  %315 = ascvec.reinterpret %314 : <i32> -> <ui32>
                  %316 = ascvec.duplicate_scalar %c7_i32, %265 : i32 -> <ui32>
                  %317 = ascvec.and %307, %316, %265 : <ui32>, <ui32> -> <ui32>
                  %318 = ascvec.shl %315, 11, %265 : <ui32> -> <ui32>
                  %319 = ascvec.shr %315, 21, %265 : <ui32> -> <ui32>
                  %320 = ascvec.or %318, %319, %265 : <ui32>, <ui32> -> <ui32>
                  %321 = ascvec.shr %320, 3, %265 : <ui32> -> <ui32>
                  %322 = ascvec.shl %320, 29, %265 : <ui32> -> <ui32>
                  %323 = ascvec.or %321, %322, %265 : <ui32>, <ui32> -> <ui32>
                  %324 = ascvec.duplicate_scalar %c1_i32_11, %265 : i32 -> <ui32>
                  %325 = ascvec.and %317, %324, %265 : <ui32>, <ui32> -> <ui32>
                  %326 = ascvec.cmp_ne_scalar %325, %c0_i32_15, %265 : <ui32>, i32 -> !ascvec.mask
                  %327 = ascvec.select %323, %320, %326 : <ui32>
                  %328 = ascvec.shr %327, 8, %265 : <ui32> -> <ui32>
                  %329 = ascvec.shl %327, 24, %265 : <ui32> -> <ui32>
                  %330 = ascvec.or %328, %329, %265 : <ui32>, <ui32> -> <ui32>
                  %331 = ascvec.duplicate_scalar %c2_i32_10, %265 : i32 -> <ui32>
                  %332 = ascvec.and %317, %331, %265 : <ui32>, <ui32> -> <ui32>
                  %333 = ascvec.cmp_ne_scalar %332, %c0_i32_15, %265 : <ui32>, i32 -> !ascvec.mask
                  %334 = ascvec.select %330, %327, %333 : <ui32>
                  %335 = ascvec.shr %334, 16, %265 : <ui32> -> <ui32>
                  %336 = ascvec.shl %334, 16, %265 : <ui32> -> <ui32>
                  %337 = ascvec.or %335, %336, %265 : <ui32>, <ui32> -> <ui32>
                  %338 = ascvec.duplicate_scalar %c4_i32_12, %265 : i32 -> <ui32>
                  %339 = ascvec.and %317, %338, %265 : <ui32>, <ui32> -> <ui32>
                  %340 = ascvec.cmp_ne_scalar %339, %c0_i32_15, %265 : <ui32>, i32 -> !ascvec.mask
                  %341 = ascvec.select %337, %334, %340 : <ui32>
                  %342 = ascvec.duplicate_scalar %c65535_i32, %265 : i32 -> <ui32>
                  %343 = ascvec.and %341, %342, %265 : <ui32>, <ui32> -> <ui32>
                  %344 = ascvec.reinterpret %299 : <f32> -> <ui32>
                  %345 = ascvec.shr %344, 24, %265 : <ui32> -> <ui32>
                  %346 = ascvec.duplicate_scalar %c128_i32, %265 : i32 -> <ui32>
                  %347 = ascvec.and %345, %346, %265 : <ui32>, <ui32> -> <ui32>
                  %348 = ascvec.and %344, %279, %265 : <ui32>, <ui32> -> <ui32>
                  %349 = ascvec.shr %348, 5, %265 : <ui32> -> <ui32>
                  %350 = ascvec.duplicate_scalar %c29360128_i32, %265 : i32 -> <ui32>
                  %351 = ascvec.sub %349, %350, %265 : <ui32>
                  %352 = ascvec.reinterpret %348 : <ui32> -> <f32>
                  %353 = ascvec.mul_scalar %352, %cst, %265 : <f32>, f32
                  %354 = ascvec.cast %353, %265 {rounding = 4 : i32} : <f32> -> <i32>
                  %355 = ascvec.reinterpret %354 : <i32> -> <ui32>
                  %356 = ascvec.cmp_ge_scalar %348, %c947912704_i32, %265 : <ui32>, i32 -> !ascvec.mask
                  %357 = ascvec.select %351, %355, %356 : <ui32>
                  %358 = ascvec.and %343, %342, %265 : <ui32>, <ui32> -> <ui32>
                  %359 = ascvec.add %357, %358, %265 : <ui32>
                  %360 = ascvec.shr %359, 16, %265 : <ui32> -> <ui32>
                  %361 = ascvec.min_scalar %360, %c123_i32, %265 : <ui32>, i32
                  %362 = ascvec.or %361, %347, %265 : <ui32>, <ui32> -> <ui32>
                  %363 = ascvec.duplicate_scalar %c127_i32, %265 : i32 -> <ui32>
                  %364 = ascvec.cmp_gt_scalar %348, %c2139095040_i32, %265 : <ui32>, i32 -> !ascvec.mask
                  %365 = ascvec.select %363, %362, %364 : <ui32>
                  ascvec.storealign_pack %arg10[%273], %365, %mask {mode = "b32_to_b8"} : memref<32768xui8, 1>, index, <ui32>
                  %366 = arith.muli %arg17, %c2_i32_10 : i32
                  %367 = arith.index_cast %366 : i32 to index
                  ascvec.storealign_1st %arg11[%367], %290 : memref<1024xui32, 1>, index, <ui32>
                  %368 = ascvec.duplicate_scalar %c32_i32, %265 : i32 -> <ui32>
                  %369 = ascvec.gather_reg %290, %368 : <ui32>, <ui32> -> <ui32>
                  %370 = arith.addi %366, %c1_i32_11 : i32
                  %371 = arith.index_cast %370 : i32 to index
                  ascvec.storealign_1st %arg11[%371], %369 : memref<1024xui32, 1>, index, <ui32>
                }
                ascvec.mem_bar vst_vld
                %269 = scf.if %arg15 -> (i32) {
                  %270 = arith.remsi %arg13, %c32_i32 : i32
                  %271 = arith.cmpi ne, %270, %c0_i32_15 : i32
                  %272 = arith.cmpi slt, %270, %c0_i32_15 : i32
                  %273 = arith.andi %271, %272 : i1
                  %274 = arith.divsi %arg13, %c32_i32 : i32
                  %275 = arith.extui %273 : i1 to i32
                  %276 = arith.subi %274, %275 : i32
                  %277 = arith.muli %276, %c2_i32_10 : i32
                  scf.yield %277 : i32
                } else {
                  scf.yield %arg14 : i32
                }
                scf.for %arg17 = %c0_i32_15 to %269 step %c64_i32  : i32 {
                  %270 = arith.subi %269, %arg17 : i32
                  %mask, %remaining_out = ascvec.update_mask %270 : !ascvec.mask
                  %271 = ascvec.add_scalar %266, %arg17, %265 : <ui32>, i32
                  scf.if %arg15 {
                    %272 = ascvec.shr %271, 1, %265 : <ui32> -> <ui32>
                    %273 = ascvec.gather %arg11[%c0_14], %272, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %274 = ascvec.duplicate_scalar %c1_i32_11, %265 : i32 -> <ui32>
                    %275 = ascvec.and %266, %274, %265 : <ui32>, <ui32> -> <ui32>
                    %276 = ascvec.cmp_eq_scalar %275, %c0_i32_15, %265 : <ui32>, i32 -> !ascvec.mask
                    %277 = ascvec.duplicate_scalar %c0_i32_15, %265 : i32 -> <ui32>
                    %278 = ascvec.select %273, %277, %276 : <ui32>
                    %279 = arith.index_cast %arg17 : i32 to index
                    %c0_16 = arith.constant 0 : index
                    %c1_17 = arith.constant 1 : index
                    scf.for %arg18 = %c0_16 to %c1_17 step %c1_17 {
                      ascvec.storealign_pack %arg16[%279], %278, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  } else {
                    %272 = ascvec.gather %arg11[%c0_14], %271, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %273 = arith.index_cast %arg17 : i32 to index
                    %c0_16 = arith.constant 0 : index
                    %c1_17 = arith.constant 1 : index
                    scf.for %arg18 = %c0_16 to %c1_17 step %c1_17 {
                      ascvec.storealign_pack %arg16[%273], %272, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  }
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
            } else {
              %199 = arith.addi %66, %c63_i64 : i64
              %200 = arith.remsi %199, %c64_i64 : i64
              %201 = arith.cmpi ne, %200, %c0_i64 : i64
              %202 = arith.cmpi slt, %200, %c0_i64 : i64
              %203 = arith.andi %201, %202 : i1
              %204 = arith.divsi %199, %c64_i64 : i64
              %205 = arith.extui %203 : i1 to i64
              %206 = arith.subi %204, %205 : i64
              %207 = arith.addi %66, %c1023_i64 : i64
              %208 = arith.remsi %207, %c1024_i64 : i64
              %209 = arith.cmpi ne, %208, %c0_i64 : i64
              %210 = arith.cmpi slt, %208, %c0_i64 : i64
              %211 = arith.andi %209, %210 : i1
              %212 = arith.divsi %207, %c1024_i64 : i64
              %213 = arith.extui %211 : i1 to i64
              %214 = arith.subi %212, %213 : i64
              %215 = arith.trunci %1 : i64 to i32
              %216 = arith.extui %215 : i32 to i64
              %217 = arith.subi %1, %216 : i64
              %218 = arith.remsi %217, %c4294967296_i64 : i64
              %219 = arith.cmpi ne, %218, %c0_i64 : i64
              %220 = arith.cmpi slt, %218, %c0_i64 : i64
              %221 = arith.andi %219, %220 : i1
              %222 = arith.divsi %217, %c4294967296_i64 : i64
              %223 = arith.extui %221 : i1 to i64
              %224 = arith.subi %222, %223 : i64
              %225 = arith.trunci %224 : i64 to i32
              %226 = arith.remsi %108, %c16_i64 : i64
              %227 = arith.cmpi ne, %226, %c0_i64 : i64
              %228 = arith.cmpi slt, %226, %c0_i64 : i64
              %229 = arith.andi %227, %228 : i1
              %230 = arith.divsi %108, %c16_i64 : i64
              %231 = arith.extui %229 : i1 to i64
              %232 = arith.subi %230, %231 : i64
              %233 = arith.remsi %arg3, %c16_i64 : i64
              %234 = arith.cmpi ne, %233, %c0_i64 : i64
              %235 = arith.cmpi slt, %233, %c0_i64 : i64
              %236 = arith.andi %234, %235 : i1
              %237 = arith.divsi %arg3, %c16_i64 : i64
              %238 = arith.extui %236 : i1 to i64
              %239 = arith.subi %237, %238 : i64
              %240 = arith.remsi %232, %239 : i64
              %241 = arith.cmpi ne, %240, %c0_i64 : i64
              %242 = arith.cmpi slt, %240, %c0_i64 : i64
              %243 = arith.cmpi slt, %239, %c0_i64 : i64
              %244 = arith.xori %242, %243 : i1
              %245 = arith.andi %241, %244 : i1
              %246 = arith.divsi %232, %239 : i64
              %247 = arith.extui %245 : i1 to i64
              %248 = arith.subi %246, %247 : i64
              %249 = arith.select %245, %239, %c0_i64 : i64
              %250 = arith.addi %240, %249 : i64
              %251 = arith.trunci %248 : i64 to i32
              %252 = arith.extui %251 : i32 to i64
              %253 = arith.subi %248, %252 : i64
              %254 = arith.remsi %253, %c4294967296_i64 : i64
              %255 = arith.cmpi ne, %254, %c0_i64 : i64
              %256 = arith.cmpi slt, %254, %c0_i64 : i64
              %257 = arith.andi %255, %256 : i1
              %258 = arith.divsi %253, %c4294967296_i64 : i64
              %259 = arith.extui %257 : i1 to i64
              %260 = arith.subi %258, %259 : i64
              %261 = arith.trunci %260 : i64 to i32
              %262 = arith.trunci %250 : i64 to i32
              %263 = arith.extui %262 : i32 to i64
              %264 = arith.subi %250, %263 : i64
              %265 = arith.remsi %264, %c4294967296_i64 : i64
              %266 = arith.cmpi ne, %265, %c0_i64 : i64
              %267 = arith.cmpi slt, %265, %c0_i64 : i64
              %268 = arith.andi %266, %267 : i1
              %269 = arith.divsi %264, %c4294967296_i64 : i64
              %270 = arith.extui %268 : i1 to i64
              %271 = arith.subi %269, %270 : i64
              %272 = arith.trunci %271 : i64 to i32
              %273 = arith.remsi %c64_i64, %239 : i64
              %274 = arith.cmpi ne, %273, %c0_i64 : i64
              %275 = arith.cmpi slt, %273, %c0_i64 : i64
              %276 = arith.xori %275, %243 : i1
              %277 = arith.andi %274, %276 : i1
              %278 = arith.divsi %c64_i64, %239 : i64
              %279 = arith.extui %277 : i1 to i64
              %280 = arith.subi %278, %279 : i64
              %281 = arith.select %277, %239, %c0_i64 : i64
              %282 = arith.addi %273, %281 : i64
              %283 = arith.trunci %214 : i64 to i32
              ascvec.vf(%239, %3, %215, %225, %alloc_8, %282, %280, %283, %251, %261, %262, %272 : i64, i64, i32, i32, memref<8192xi32, 1>, i64, i64, i32, i32, i32, i32, i32) {
              ^bb0(%arg7: i64, %arg8: i64, %arg9: i32, %arg10: i32, %arg11: memref<8192xi32, 1>, %arg12: i64, %arg13: i64, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32):
                %c1_i32_10 = arith.constant 1 : i32
                %c192_i32 = arith.constant 192 : i32
                %c128_i32 = arith.constant 128 : i32
                %c64_i32 = arith.constant 64 : i32
                %c-1767562579_i32 = arith.constant -1767562579 : i32
                %c-1879881855_i32 = arith.constant -1879881855 : i32
                %c-616729560_i32 = arith.constant -616729560 : i32
                %c-239350328_i32 = arith.constant -239350328 : i32
                %c534103459_i32 = arith.constant 534103459 : i32
                %c1401181199_i32 = arith.constant 1401181199 : i32
                %c1684936478_i32 = arith.constant 1684936478 : i32
                %c-1253254570_i32 = arith.constant -1253254570 : i32
                %c-1459197799_i32 = arith.constant -1459197799 : i32
                %c387276957_i32 = arith.constant 387276957 : i32
                %c-308364780_i32 = arith.constant -308364780 : i32
                %c2027808484_i32 = arith.constant 2027808484 : i32
                %c842468239_i32 = arith.constant 842468239 : i32
                %c-626627285_i32 = arith.constant -626627285 : i32
                %c1993301258_i32_11 = arith.constant 1993301258 : i32
                %c1013904242_i32_12 = arith.constant 1013904242 : i32
                %c-1150833019_i32_13 = arith.constant -1150833019 : i32
                %c-1640531527_i32_14 = arith.constant -1640531527 : i32
                %c-845247145_i32 = arith.constant -845247145 : i32
                %c-766435501_i32 = arith.constant -766435501 : i32
                %c16383_i32 = arith.constant 16383 : i32
                %c1073741824_i32_15 = arith.constant 1073741824 : i32
                %c4_i32_16 = arith.constant 4 : i32
                %c0_i64_17 = arith.constant 0 : i64
                %c0_i32_18 = arith.constant 0 : i32
                %c256_i32 = arith.constant 256 : i32
                %c4294967296_i64_19 = arith.constant 4294967296 : i64
                %289:4 = scf.for %arg19 = %c0_i32_18 to %arg14 step %c1_i32_10 iter_args(%arg20 = %arg15, %arg21 = %arg16, %arg22 = %arg17, %arg23 = %arg18) -> (i32, i32, i32, i32)  : i32 {
                  %290 = arith.extui %arg20 : i32 to i64
                  %291 = arith.extui %arg21 : i32 to i64
                  %292 = arith.muli %291, %c4294967296_i64_19 : i64
                  %293 = arith.addi %290, %292 : i64
                  %294 = arith.extui %arg22 : i32 to i64
                  %295 = arith.extui %arg23 : i32 to i64
                  %296 = arith.muli %295, %c4294967296_i64_19 : i64
                  %297 = arith.addi %294, %296 : i64
                  %298 = arith.muli %arg19, %c256_i32 : i32
                  %299 = ascvec.create_mask : !ascvec.mask
                  %300 = ascvec.arange %c0_i32_18 : i32 -> <ui32>
                  %301 = arith.muli %293, %arg7 : i64
                  %302 = arith.addi %301, %297 : i64
                  %303 = arith.trunci %302 : i64 to i32
                  %304 = arith.extui %303 : i32 to i64
                  %305 = arith.subi %302, %304 : i64
                  %306 = arith.remsi %305, %c4294967296_i64_19 : i64
                  %307 = arith.cmpi ne, %306, %c0_i64_17 : i64
                  %308 = arith.cmpi slt, %306, %c0_i64_17 : i64
                  %309 = arith.andi %307, %308 : i1
                  %310 = arith.divsi %305, %c4294967296_i64_19 : i64
                  %311 = arith.extui %309 : i1 to i64
                  %312 = arith.subi %310, %311 : i64
                  %313 = arith.trunci %312 : i64 to i32
                  %314 = ascvec.duplicate_scalar %303, %299 : i32 -> <ui32>
                  %315 = ascvec.duplicate_scalar %313, %299 : i32 -> <ui32>
                  %316 = ascvec.duplicate_scalar %c0_i32_18, %299 : i32 -> <ui32>
                  %carry, %dst = ascvec.addc %314, %300, %299 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %carry_20, %dst_21 = ascvec.addc %315, %316, %carry, %299 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %317 = arith.trunci %arg8 : i64 to i32
                  %318 = arith.extui %317 : i32 to i64
                  %319 = arith.subi %arg8, %318 : i64
                  %320 = arith.remsi %319, %c4294967296_i64_19 : i64
                  %321 = arith.cmpi ne, %320, %c0_i64_17 : i64
                  %322 = arith.cmpi slt, %320, %c0_i64_17 : i64
                  %323 = arith.andi %321, %322 : i1
                  %324 = arith.divsi %319, %c4294967296_i64_19 : i64
                  %325 = arith.extui %323 : i1 to i64
                  %326 = arith.subi %324, %325 : i64
                  %327 = arith.trunci %326 : i64 to i32
                  %328 = arith.divui %317, %c4_i32_16 : i32
                  %329 = arith.remui %327, %c4_i32_16 : i32
                  %330 = arith.muli %329, %c1073741824_i32_15 : i32
                  %331 = arith.addi %328, %330 : i32
                  %332 = arith.divui %327, %c4_i32_16 : i32
                  %333 = ascvec.shr %dst, 14, %299 : <ui32> -> <ui32>
                  %334 = ascvec.shl %dst_21, 18, %299 : <ui32> -> <ui32>
                  %335 = ascvec.or %333, %334, %299 : <ui32>, <ui32> -> <ui32>
                  %336 = ascvec.shr %dst_21, 14, %299 : <ui32> -> <ui32>
                  %337 = ascvec.duplicate_scalar %331, %299 : i32 -> <ui32>
                  %338 = ascvec.duplicate_scalar %332, %299 : i32 -> <ui32>
                  %carry_22, %dst_23 = ascvec.addc %337, %335, %299 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %carry_24, %dst_25 = ascvec.addc %338, %336, %carry_22, %299 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %339 = ascvec.duplicate_scalar %c16383_i32, %299 : i32 -> <ui32>
                  %340 = ascvec.and %dst, %339, %299 : <ui32>, <ui32> -> <ui32>
                  %341 = ascvec.duplicate_scalar %c-766435501_i32, %299 : i32 -> <ui32>
                  %342 = ascvec.duplicate_scalar %c-845247145_i32, %299 : i32 -> <ui32>
                  %dst0, %dst1 = ascvec.mull %316, %341, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_26, %dst1_27 = ascvec.mull %dst_23, %342, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %343 = ascvec.xor %dst1_27, %340, %299 : <ui32>, <ui32> -> <ui32>
                  %344 = ascvec.xor %dst1, %dst_25, %299 : <ui32>, <ui32> -> <ui32>
                  %345 = ascvec.duplicate_scalar %arg9, %299 : i32 -> <ui32>
                  %346 = ascvec.duplicate_scalar %arg10, %299 : i32 -> <ui32>
                  %347 = ascvec.xor %343, %345, %299 : <ui32>, <ui32> -> <ui32>
                  %348 = ascvec.xor %344, %346, %299 : <ui32>, <ui32> -> <ui32>
                  %dst0_28, %dst1_29 = ascvec.mull %347, %341, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_30, %dst1_31 = ascvec.mull %348, %342, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %349 = ascvec.xor %dst1_31, %dst0_26, %299 : <ui32>, <ui32> -> <ui32>
                  %350 = ascvec.xor %dst1_29, %dst0, %299 : <ui32>, <ui32> -> <ui32>
                  %351 = arith.addi %arg9, %c-1640531527_i32_14 : i32
                  %352 = ascvec.duplicate_scalar %351, %299 : i32 -> <ui32>
                  %353 = arith.addi %arg10, %c-1150833019_i32_13 : i32
                  %354 = ascvec.duplicate_scalar %353, %299 : i32 -> <ui32>
                  %355 = ascvec.xor %349, %352, %299 : <ui32>, <ui32> -> <ui32>
                  %356 = ascvec.xor %350, %354, %299 : <ui32>, <ui32> -> <ui32>
                  %dst0_32, %dst1_33 = ascvec.mull %355, %341, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_34, %dst1_35 = ascvec.mull %356, %342, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %357 = ascvec.xor %dst1_35, %dst0_30, %299 : <ui32>, <ui32> -> <ui32>
                  %358 = ascvec.xor %dst1_33, %dst0_28, %299 : <ui32>, <ui32> -> <ui32>
                  %359 = arith.addi %arg9, %c1013904242_i32_12 : i32
                  %360 = ascvec.duplicate_scalar %359, %299 : i32 -> <ui32>
                  %361 = arith.addi %arg10, %c1993301258_i32_11 : i32
                  %362 = ascvec.duplicate_scalar %361, %299 : i32 -> <ui32>
                  %363 = ascvec.xor %357, %360, %299 : <ui32>, <ui32> -> <ui32>
                  %364 = ascvec.xor %358, %362, %299 : <ui32>, <ui32> -> <ui32>
                  %dst0_36, %dst1_37 = ascvec.mull %363, %341, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_38, %dst1_39 = ascvec.mull %364, %342, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %365 = ascvec.xor %dst1_39, %dst0_34, %299 : <ui32>, <ui32> -> <ui32>
                  %366 = ascvec.xor %dst1_37, %dst0_32, %299 : <ui32>, <ui32> -> <ui32>
                  %367 = arith.addi %arg9, %c-626627285_i32 : i32
                  %368 = ascvec.duplicate_scalar %367, %299 : i32 -> <ui32>
                  %369 = arith.addi %arg10, %c842468239_i32 : i32
                  %370 = ascvec.duplicate_scalar %369, %299 : i32 -> <ui32>
                  %371 = ascvec.xor %365, %368, %299 : <ui32>, <ui32> -> <ui32>
                  %372 = ascvec.xor %366, %370, %299 : <ui32>, <ui32> -> <ui32>
                  %dst0_40, %dst1_41 = ascvec.mull %371, %341, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_42, %dst1_43 = ascvec.mull %372, %342, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %373 = ascvec.xor %dst1_43, %dst0_38, %299 : <ui32>, <ui32> -> <ui32>
                  %374 = ascvec.xor %dst1_41, %dst0_36, %299 : <ui32>, <ui32> -> <ui32>
                  %375 = arith.addi %arg9, %c2027808484_i32 : i32
                  %376 = ascvec.duplicate_scalar %375, %299 : i32 -> <ui32>
                  %377 = arith.addi %arg10, %c-308364780_i32 : i32
                  %378 = ascvec.duplicate_scalar %377, %299 : i32 -> <ui32>
                  %379 = ascvec.xor %373, %376, %299 : <ui32>, <ui32> -> <ui32>
                  %380 = ascvec.xor %374, %378, %299 : <ui32>, <ui32> -> <ui32>
                  %dst0_44, %dst1_45 = ascvec.mull %379, %341, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_46, %dst1_47 = ascvec.mull %380, %342, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %381 = ascvec.xor %dst1_47, %dst0_42, %299 : <ui32>, <ui32> -> <ui32>
                  %382 = ascvec.xor %dst1_45, %dst0_40, %299 : <ui32>, <ui32> -> <ui32>
                  %383 = arith.addi %arg9, %c387276957_i32 : i32
                  %384 = ascvec.duplicate_scalar %383, %299 : i32 -> <ui32>
                  %385 = arith.addi %arg10, %c-1459197799_i32 : i32
                  %386 = ascvec.duplicate_scalar %385, %299 : i32 -> <ui32>
                  %387 = ascvec.xor %381, %384, %299 : <ui32>, <ui32> -> <ui32>
                  %388 = ascvec.xor %382, %386, %299 : <ui32>, <ui32> -> <ui32>
                  %dst0_48, %dst1_49 = ascvec.mull %387, %341, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_50, %dst1_51 = ascvec.mull %388, %342, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %389 = ascvec.xor %dst1_51, %dst0_46, %299 : <ui32>, <ui32> -> <ui32>
                  %390 = ascvec.xor %dst1_49, %dst0_44, %299 : <ui32>, <ui32> -> <ui32>
                  %391 = arith.addi %arg9, %c-1253254570_i32 : i32
                  %392 = ascvec.duplicate_scalar %391, %299 : i32 -> <ui32>
                  %393 = arith.addi %arg10, %c1684936478_i32 : i32
                  %394 = ascvec.duplicate_scalar %393, %299 : i32 -> <ui32>
                  %395 = ascvec.xor %389, %392, %299 : <ui32>, <ui32> -> <ui32>
                  %396 = ascvec.xor %390, %394, %299 : <ui32>, <ui32> -> <ui32>
                  %dst0_52, %dst1_53 = ascvec.mull %395, %341, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_54, %dst1_55 = ascvec.mull %396, %342, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %397 = ascvec.xor %dst1_55, %dst0_50, %299 : <ui32>, <ui32> -> <ui32>
                  %398 = ascvec.xor %dst1_53, %dst0_48, %299 : <ui32>, <ui32> -> <ui32>
                  %399 = arith.addi %arg9, %c1401181199_i32 : i32
                  %400 = ascvec.duplicate_scalar %399, %299 : i32 -> <ui32>
                  %401 = arith.addi %arg10, %c534103459_i32 : i32
                  %402 = ascvec.duplicate_scalar %401, %299 : i32 -> <ui32>
                  %403 = ascvec.xor %397, %400, %299 : <ui32>, <ui32> -> <ui32>
                  %404 = ascvec.xor %398, %402, %299 : <ui32>, <ui32> -> <ui32>
                  %dst0_56, %dst1_57 = ascvec.mull %403, %341, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_58, %dst1_59 = ascvec.mull %404, %342, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %405 = ascvec.xor %dst1_59, %dst0_54, %299 : <ui32>, <ui32> -> <ui32>
                  %406 = ascvec.xor %dst1_57, %dst0_52, %299 : <ui32>, <ui32> -> <ui32>
                  %407 = arith.addi %arg9, %c-239350328_i32 : i32
                  %408 = ascvec.duplicate_scalar %407, %299 : i32 -> <ui32>
                  %409 = arith.addi %arg10, %c-616729560_i32 : i32
                  %410 = ascvec.duplicate_scalar %409, %299 : i32 -> <ui32>
                  %411 = ascvec.xor %405, %408, %299 : <ui32>, <ui32> -> <ui32>
                  %412 = ascvec.xor %406, %410, %299 : <ui32>, <ui32> -> <ui32>
                  %dst0_60, %dst1_61 = ascvec.mull %411, %341, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_62, %dst1_63 = ascvec.mull %412, %342, %299 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %413 = ascvec.xor %dst1_63, %dst0_58, %299 : <ui32>, <ui32> -> <ui32>
                  %414 = ascvec.xor %dst1_61, %dst0_56, %299 : <ui32>, <ui32> -> <ui32>
                  %415 = arith.addi %arg9, %c-1879881855_i32 : i32
                  %416 = ascvec.duplicate_scalar %415, %299 : i32 -> <ui32>
                  %417 = arith.addi %arg10, %c-1767562579_i32 : i32
                  %418 = ascvec.duplicate_scalar %417, %299 : i32 -> <ui32>
                  %419 = ascvec.xor %413, %416, %299 : <ui32>, <ui32> -> <ui32>
                  %420 = ascvec.xor %414, %418, %299 : <ui32>, <ui32> -> <ui32>
                  %421 = ascvec.reinterpret %419 : <ui32> -> <i32>
                  %422 = arith.index_cast %298 : i32 to index
                  ascvec.storealign %arg11[%422], %421, %299 : memref<8192xi32, 1>, index, <i32>
                  %423 = arith.addi %298, %c64_i32 : i32
                  %424 = ascvec.reinterpret %dst0_62 : <ui32> -> <i32>
                  %425 = arith.index_cast %423 : i32 to index
                  ascvec.storealign %arg11[%425], %424, %299 : memref<8192xi32, 1>, index, <i32>
                  %426 = arith.addi %298, %c128_i32 : i32
                  %427 = ascvec.reinterpret %420 : <ui32> -> <i32>
                  %428 = arith.index_cast %426 : i32 to index
                  ascvec.storealign %arg11[%428], %427, %299 : memref<8192xi32, 1>, index, <i32>
                  %429 = arith.addi %298, %c192_i32 : i32
                  %430 = ascvec.reinterpret %dst0_60 : <ui32> -> <i32>
                  %431 = arith.index_cast %429 : i32 to index
                  ascvec.storealign %arg11[%431], %430, %299 : memref<8192xi32, 1>, index, <i32>
                  %432 = arith.addi %297, %arg12 : i64
                  %433 = arith.cmpi sge, %432, %arg7 : i64
                  %434 = arith.extui %433 : i1 to i64
                  %435 = arith.addi %293, %arg13 : i64
                  %436 = arith.addi %435, %434 : i64
                  %437 = arith.muli %434, %arg7 : i64
                  %438 = arith.subi %432, %437 : i64
                  %439 = arith.trunci %436 : i64 to i32
                  %440 = arith.extui %439 : i32 to i64
                  %441 = arith.subi %436, %440 : i64
                  %442 = arith.remsi %441, %c4294967296_i64_19 : i64
                  %443 = arith.cmpi ne, %442, %c0_i64_17 : i64
                  %444 = arith.cmpi slt, %442, %c0_i64_17 : i64
                  %445 = arith.andi %443, %444 : i1
                  %446 = arith.divsi %441, %c4294967296_i64_19 : i64
                  %447 = arith.extui %445 : i1 to i64
                  %448 = arith.subi %446, %447 : i64
                  %449 = arith.trunci %448 : i64 to i32
                  %450 = arith.trunci %438 : i64 to i32
                  %451 = arith.extui %450 : i32 to i64
                  %452 = arith.subi %438, %451 : i64
                  %453 = arith.remsi %452, %c4294967296_i64_19 : i64
                  %454 = arith.cmpi ne, %453, %c0_i64_17 : i64
                  %455 = arith.cmpi slt, %453, %c0_i64_17 : i64
                  %456 = arith.andi %454, %455 : i1
                  %457 = arith.divsi %452, %c4294967296_i64_19 : i64
                  %458 = arith.extui %456 : i1 to i64
                  %459 = arith.subi %457, %458 : i64
                  %460 = arith.trunci %459 : i64 to i32
                  scf.yield %439, %449, %450, %460 : i32, i32, i32, i32
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              %284 = arith.trunci %206 : i64 to i32
              %285 = arith.trunci %66 : i64 to i32
              %286 = arith.trunci %48 : i64 to i32
              %287 = arith.muli %284, %c2_i32 : i32
              %288 = arith.cmpi eq, %286, %c4_i32 : i32
              ascvec.get_buf v, %72, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%66, %107, %alloc_8, %alloc_5, %alloc_7, %284, %285, %287, %288, %alloc_6 : i64, memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, memref<1024xui32, 1>, i32, i32, i32, i1, memref<2048xui8, 1>) {
              ^bb0(%arg7: i64, %arg8: memref<32768xbf16, 1>, %arg9: memref<8192xi32, 1>, %arg10: memref<32768xui8, 1>, %arg11: memref<1024xui32, 1>, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i1, %arg16: memref<2048xui8, 1>):
                %c2_i32_10 = arith.constant 2 : i32
                %c2139095040_i32 = arith.constant 2139095040 : i32
                %c127_i32 = arith.constant 127 : i32
                %c123_i32 = arith.constant 123 : i32
                %c947912704_i32 = arith.constant 947912704 : i32
                %cst = arith.constant 4.2949673E+9 : f32
                %c29360128_i32 = arith.constant 29360128 : i32
                %c128_i32 = arith.constant 128 : i32
                %c65535_i32 = arith.constant 65535 : i32
                %c4_i32_11 = arith.constant 4 : i32
                %c256_i32 = arith.constant 256 : i32
                %c1_i32_12 = arith.constant 1 : i32
                %c16_i32 = arith.constant 16 : i32
                %c3_i32 = arith.constant 3 : i32
                %c2139160576_i32 = arith.constant 2139160576 : i32
                %c254_i32 = arith.constant 254 : i32
                %c-15_i32 = arith.constant -15 : i32
                %c15_i32 = arith.constant 15 : i32
                %c255_i32 = arith.constant 255 : i32
                %c2147483647_i32 = arith.constant 2147483647 : i32
                %cst_13 = arith.constant 0.000000e+00 : f32
                %c0_14 = arith.constant 0 : index
                %c64_i32 = arith.constant 64 : i32
                %c32_i32 = arith.constant 32 : i32
                %c0_i32_15 = arith.constant 0 : i32
                %289 = ascvec.create_mask : !ascvec.mask
                %290 = ascvec.arange %c0_i32_15 : i32 -> <ui32>
                %291 = ascvec.cmp_lt_scalar %290, %c32_i32, %289 : <ui32>, i32 -> !ascvec.mask
                %292 = ascvec.cmp_ge_scalar %290, %c32_i32, %289 : <ui32>, i32 -> !ascvec.mask
                scf.for %arg17 = %c0_i32_15 to %arg12 step %c1_i32_12  : i32 {
                  %294 = arith.muli %arg17, %c64_i32 : i32
                  %295 = arith.trunci %arg7 : i64 to i32
                  %296 = arith.subi %295, %294 : i32
                  %mask, %remaining_out = ascvec.update_mask %296 : !ascvec.mask
                  %297 = arith.index_cast %294 : i32 to index
                  %298 = ascvec.loadalign_unpack %arg8[%297] {mode = "b16_to_b32"} : memref<32768xbf16, 1>, index -> <bf16>
                  %299 = ascvec.cast %298, %289 {rounding = 6 : i32} : <bf16> -> <f32>
                  %300 = ascvec.duplicate_scalar %cst_13, %289 : f32 -> <f32>
                  %301 = ascvec.select %299, %300, %mask : <f32>
                  %302 = ascvec.reinterpret %301 : <f32> -> <ui32>
                  %303 = ascvec.duplicate_scalar %c2147483647_i32, %289 : i32 -> <ui32>
                  %304 = ascvec.and %302, %303, %289 : <ui32>, <ui32> -> <ui32>
                  %305 = ascvec.reduce_max %304, %291 : <ui32>
                  %306 = ascvec.duplicate %305, %289 : <ui32> -> <ui32>
                  %307 = ascvec.reduce_max %304, %292 : <ui32>
                  %308 = ascvec.duplicate %307, %289 : <ui32> -> <ui32>
                  %309 = ascvec.select %306, %308, %291 : <ui32>
                  %310 = ascvec.shr %309, 23, %289 : <ui32> -> <ui32>
                  %311 = ascvec.cmp_ge_scalar %310, %c255_i32, %289 : <ui32>, i32 -> !ascvec.mask
                  %312 = ascvec.max_scalar %310, %c15_i32, %289 : <ui32>, i32
                  %313 = ascvec.add_scalar %312, %c-15_i32, %289 : <ui32>, i32
                  %314 = ascvec.duplicate_scalar %c255_i32, %311 merge %313 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %315 = ascvec.duplicate_scalar %c254_i32, %289 : i32 -> <ui32>
                  %316 = ascvec.sub %315, %314, %289 : <ui32>
                  %317 = ascvec.cmp_ne_scalar %314, %c0_i32_15, %289 : <ui32>, i32 -> !ascvec.mask
                  %318 = ascvec.shl %316, 23, %317 : <ui32> -> <ui32>
                  %319 = ascvec.duplicate_scalar %c2139160576_i32, %311 merge %318 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %320 = ascvec.reinterpret %319 : <ui32> -> <f32>
                  %321 = ascvec.mul %301, %320, %289 : <f32>
                  %322 = ascvec.shr %290, 2, %289 : <ui32> -> <ui32>
                  %323 = ascvec.duplicate_scalar %c3_i32, %289 : i32 -> <ui32>
                  %324 = ascvec.and %322, %323, %289 : <ui32>, <ui32> -> <ui32>
                  %325 = ascvec.shl %324, 6, %289 : <ui32> -> <ui32>
                  %326 = ascvec.shr %290, 4, %289 : <ui32> -> <ui32>
                  %327 = ascvec.add %325, %326, %289 : <ui32>
                  %328 = arith.remsi %arg17, %c16_i32 : i32
                  %329 = arith.cmpi ne, %328, %c0_i32_15 : i32
                  %330 = arith.cmpi slt, %328, %c0_i32_15 : i32
                  %331 = arith.andi %329, %330 : i1
                  %332 = arith.divsi %arg17, %c16_i32 : i32
                  %333 = arith.extui %331 : i1 to i32
                  %334 = arith.subi %332, %333 : i32
                  %335 = arith.muli %334, %c256_i32 : i32
                  %336 = arith.select %331, %c16_i32, %c0_i32_15 : i32
                  %337 = arith.addi %328, %336 overflow<nsw> : i32
                  %338 = arith.muli %337, %c4_i32_11 overflow<nsw> : i32
                  %339 = arith.addi %335, %338 : i32
                  %340 = ascvec.add_scalar %327, %339, %289 : <ui32>, i32
                  %341 = ascvec.gather %arg9[%c0_14], %340, %289 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                  %342 = ascvec.reinterpret %341 : <i32> -> <ui32>
                  %343 = ascvec.shl %342, 8, %289 : <ui32> -> <ui32>
                  %344 = ascvec.shr %342, 24, %289 : <ui32> -> <ui32>
                  %345 = ascvec.or %343, %344, %289 : <ui32>, <ui32> -> <ui32>
                  %346 = ascvec.and %290, %323, %289 : <ui32>, <ui32> -> <ui32>
                  %347 = ascvec.duplicate_scalar %c1_i32_12, %289 : i32 -> <ui32>
                  %348 = ascvec.and %346, %347, %289 : <ui32>, <ui32> -> <ui32>
                  %349 = ascvec.cmp_eq_scalar %348, %c0_i32_15, %289 : <ui32>, i32 -> !ascvec.mask
                  %350 = ascvec.select %342, %345, %349 : <ui32>
                  %351 = ascvec.cmp_eq_scalar %346, %c0_i32_15, %289 : <ui32>, i32 -> !ascvec.mask
                  %352 = ascvec.cmp_eq_scalar %346, %c3_i32, %289 : <ui32>, i32 -> !ascvec.mask
                  %353 = ascvec.mask_or %351, %352, %289
                  %354 = ascvec.shr %350, 16, %289 : <ui32> -> <ui32>
                  %355 = ascvec.duplicate_scalar %c65535_i32, %289 : i32 -> <ui32>
                  %356 = ascvec.and %350, %355, %289 : <ui32>, <ui32> -> <ui32>
                  %357 = ascvec.select %354, %356, %353 : <ui32>
                  %358 = ascvec.reinterpret %321 : <f32> -> <ui32>
                  %359 = ascvec.shr %358, 24, %289 : <ui32> -> <ui32>
                  %360 = ascvec.duplicate_scalar %c128_i32, %289 : i32 -> <ui32>
                  %361 = ascvec.and %359, %360, %289 : <ui32>, <ui32> -> <ui32>
                  %362 = ascvec.and %358, %303, %289 : <ui32>, <ui32> -> <ui32>
                  %363 = ascvec.shr %362, 5, %289 : <ui32> -> <ui32>
                  %364 = ascvec.duplicate_scalar %c29360128_i32, %289 : i32 -> <ui32>
                  %365 = ascvec.sub %363, %364, %289 : <ui32>
                  %366 = ascvec.reinterpret %362 : <ui32> -> <f32>
                  %367 = ascvec.mul_scalar %366, %cst, %289 : <f32>, f32
                  %368 = ascvec.cast %367, %289 {rounding = 4 : i32} : <f32> -> <i32>
                  %369 = ascvec.reinterpret %368 : <i32> -> <ui32>
                  %370 = ascvec.cmp_ge_scalar %362, %c947912704_i32, %289 : <ui32>, i32 -> !ascvec.mask
                  %371 = ascvec.select %365, %369, %370 : <ui32>
                  %372 = ascvec.and %357, %355, %289 : <ui32>, <ui32> -> <ui32>
                  %373 = ascvec.add %371, %372, %289 : <ui32>
                  %374 = ascvec.shr %373, 16, %289 : <ui32> -> <ui32>
                  %375 = ascvec.min_scalar %374, %c123_i32, %289 : <ui32>, i32
                  %376 = ascvec.or %375, %361, %289 : <ui32>, <ui32> -> <ui32>
                  %377 = ascvec.duplicate_scalar %c127_i32, %289 : i32 -> <ui32>
                  %378 = ascvec.cmp_gt_scalar %362, %c2139095040_i32, %289 : <ui32>, i32 -> !ascvec.mask
                  %379 = ascvec.select %377, %376, %378 : <ui32>
                  ascvec.storealign_pack %arg10[%297], %379, %mask {mode = "b32_to_b8"} : memref<32768xui8, 1>, index, <ui32>
                  %380 = arith.muli %arg17, %c2_i32_10 : i32
                  %381 = arith.index_cast %380 : i32 to index
                  ascvec.storealign_1st %arg11[%381], %314 : memref<1024xui32, 1>, index, <ui32>
                  %382 = ascvec.duplicate_scalar %c32_i32, %289 : i32 -> <ui32>
                  %383 = ascvec.gather_reg %314, %382 : <ui32>, <ui32> -> <ui32>
                  %384 = arith.addi %380, %c1_i32_12 : i32
                  %385 = arith.index_cast %384 : i32 to index
                  ascvec.storealign_1st %arg11[%385], %383 : memref<1024xui32, 1>, index, <ui32>
                }
                ascvec.mem_bar vst_vld
                %293 = scf.if %arg15 -> (i32) {
                  %294 = arith.remsi %arg13, %c32_i32 : i32
                  %295 = arith.cmpi ne, %294, %c0_i32_15 : i32
                  %296 = arith.cmpi slt, %294, %c0_i32_15 : i32
                  %297 = arith.andi %295, %296 : i1
                  %298 = arith.divsi %arg13, %c32_i32 : i32
                  %299 = arith.extui %297 : i1 to i32
                  %300 = arith.subi %298, %299 : i32
                  %301 = arith.muli %300, %c2_i32_10 : i32
                  scf.yield %301 : i32
                } else {
                  scf.yield %arg14 : i32
                }
                scf.for %arg17 = %c0_i32_15 to %293 step %c64_i32  : i32 {
                  %294 = arith.subi %293, %arg17 : i32
                  %mask, %remaining_out = ascvec.update_mask %294 : !ascvec.mask
                  %295 = ascvec.add_scalar %290, %arg17, %289 : <ui32>, i32
                  scf.if %arg15 {
                    %296 = ascvec.shr %295, 1, %289 : <ui32> -> <ui32>
                    %297 = ascvec.gather %arg11[%c0_14], %296, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %298 = ascvec.duplicate_scalar %c1_i32_12, %289 : i32 -> <ui32>
                    %299 = ascvec.and %290, %298, %289 : <ui32>, <ui32> -> <ui32>
                    %300 = ascvec.cmp_eq_scalar %299, %c0_i32_15, %289 : <ui32>, i32 -> !ascvec.mask
                    %301 = ascvec.duplicate_scalar %c0_i32_15, %289 : i32 -> <ui32>
                    %302 = ascvec.select %297, %301, %300 : <ui32>
                    %303 = arith.index_cast %arg17 : i32 to index
                    %c0_16 = arith.constant 0 : index
                    %c1_17 = arith.constant 1 : index
                    scf.for %arg18 = %c0_16 to %c1_17 step %c1_17 {
                      ascvec.storealign_pack %arg16[%303], %302, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  } else {
                    %296 = ascvec.gather %arg11[%c0_14], %295, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %297 = arith.index_cast %arg17 : i32 to index
                    %c0_16 = arith.constant 0 : index
                    %c1_17 = arith.constant 1 : index
                    scf.for %arg18 = %c0_16 to %c1_17 step %c1_17 {
                      ascvec.storealign_pack %arg16[%297], %296, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  }
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %72, 0 {cannir.buffer.generated_sync}
            }
          }
          %113 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %114 = cannir.extract_shape %113 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
          %115 = cannir.extract_dim %114[0] : !cannir.shape<(-1, -1)>
          %116 = arith.minsi %115, %c0_i64 : i64
          %117 = arith.subi %115, %116 : i64
          %118 = arith.minsi %117, %c1_i64 : i64
          %119 = arith.cmpi sge, %115, %c0_i64 : i64
          %120 = arith.select %119, %118, %c0_i64 : i64
          %121 = arith.cmpi sgt, %120, %c1_i64 : i64
          %122 = arith.select %121, %c1_i64, %120 : i64
          %123 = cannir.extract_dim %114[1] : !cannir.shape<(-1, -1)>
          %124 = arith.divsi %123, %84 : i64
          %125 = arith.minsi %35, %124 : i64
          %126 = arith.muli %125, %34 : i64
          %127 = arith.subi %123, %126 : i64
          %128 = arith.minsi %34, %127 : i64
          %129 = arith.cmpi sle, %35, %124 : i64
          %130 = arith.select %129, %128, %c0_i64 : i64
          %131 = arith.divsi %130, %c32768_i64 : i64
          %132 = arith.minsi %arg5, %131 : i64
          %133 = arith.muli %132, %c32768_i64 : i64
          %134 = arith.subi %130, %133 : i64
          %135 = arith.minsi %134, %c32768_i64 : i64
          %136 = arith.cmpi sle, %arg5, %131 : i64
          %137 = arith.select %136, %135, %c0_i64 : i64
          %138 = arith.cmpi sgt, %137, %c32768_i64 : i64
          %139 = arith.select %138, %c32768_i64, %137 : i64
          %140 = arith.muli %108, %c8_i64 : i64
          %141 = ascscalar.memref_address %alloc_0 : memref<1xui8>
          %142 = arith.divui %140, %c8_i64 : i64
          %143 = arith.addi %141, %142 : i64
          %144 = ascscalar.bind_gm_ptr %143 : memref<1xui8>
          %145 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<32768xui8, 1> -> memref<?xui8, 1>
          %146 = cannir.extract_stride %113 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %147 = cannir.extract_dim %146[0] : !cannir.stride<(-1, 1)>
          %148 = arith.cmpi ugt, %122, %c0_i64 : i64
          %149 = arith.cmpi ugt, %139, %c0_i64 : i64
          %150 = arith.andi %148, %149 : i1
          scf.if %150 {
            ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
            %192 = arith.trunci %122 : i64 to i32
            %193 = arith.trunci %139 : i64 to i32
            ascvec.copy_ub2gm %144[%c0], %145[%c0], %192, %193, %147, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
          }
          %151 = arith.muli %48, %c512_i64 overflow<nsw> : i64
          %152 = cannir.extract_dim %38[0] : !cannir.shape<(-1, -1)>
          %153 = arith.minsi %152, %c0_i64 : i64
          %154 = arith.subi %152, %153 : i64
          %155 = arith.minsi %154, %c1_i64 : i64
          %156 = arith.cmpi sge, %152, %c0_i64 : i64
          %157 = arith.select %156, %155, %c0_i64 : i64
          %158 = arith.cmpi sgt, %157, %c1_i64 : i64
          %159 = arith.select %158, %c1_i64, %157 : i64
          %160 = arith.maxsi %50, %c1_i64 : i64
          %161 = arith.divsi %39, %160 : i64
          %162 = arith.minsi %35, %161 : i64
          %163 = arith.muli %162, %50 : i64
          %164 = arith.subi %39, %163 : i64
          %165 = arith.minsi %50, %164 : i64
          %166 = arith.cmpi sle, %35, %161 : i64
          %167 = arith.select %166, %165, %c0_i64 : i64
          %168 = arith.maxsi %151, %c1_i64 : i64
          %169 = arith.divsi %167, %168 : i64
          %170 = arith.minsi %arg5, %169 : i64
          %171 = arith.muli %170, %151 : i64
          %172 = arith.subi %167, %171 : i64
          %173 = arith.minsi %151, %172 : i64
          %174 = arith.cmpi sle, %arg5, %169 : i64
          %175 = arith.select %174, %173, %c0_i64 : i64
          %176 = arith.cmpi sgt, %175, %c2048_i64 : i64
          %177 = arith.select %176, %c2048_i64, %175 : i64
          %178 = arith.muli %arg5, %151 : i64
          %179 = arith.muli %35, %50 : i64
          %180 = arith.addi %178, %179 : i64
          %181 = arith.muli %180, %c8_i64 : i64
          %182 = ascscalar.memref_address %alloc_1 : memref<1xui8>
          %183 = arith.divui %181, %c8_i64 : i64
          %184 = arith.addi %182, %183 : i64
          %185 = ascscalar.bind_gm_ptr %184 : memref<1xui8>
          %186 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<2048xui8, 1> -> memref<?xui8, 1>
          %187 = cannir.extract_stride %37 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %188 = cannir.extract_dim %187[0] : !cannir.stride<(-1, 1)>
          %189 = arith.cmpi ugt, %159, %c0_i64 : i64
          %190 = arith.cmpi ugt, %177, %c0_i64 : i64
          %191 = arith.andi %189, %190 : i1
          scf.if %191 {
            ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
            %192 = arith.trunci %159 : i64 to i32
            %193 = arith.trunci %177 : i64 to i32
            ascvec.copy_ub2gm %185[%c0], %186[%c0], %192, %193, %188, %c2048_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %74 : i64
        }
      }
    } else {
      %9 = arith.cmpi sgt, %6, %c1_i64 : i64
      %10 = scf.if %9 -> (i1) {
        %11 = cannir.extract_dim %5[1] : !cannir.shape<(-1, -1)>
        %12 = arith.cmpi slt, %11, %c128_i64 : i64
        scf.yield %12 : i1
      } else {
        scf.yield %false : i1
      }
      scf.if %10 {
        %11 = cannir.extract_dim %5[1] : !cannir.shape<(-1, -1)>
        %12 = arith.cmpi sgt, %11, %c64_i64 : i64
        %13 = arith.select %12, %c7_i64, %c6_i64 : i64
        %14 = arith.select %12, %c256_i64, %c512_i64 : i64
        %15 = ascscalar.get_block_num
        %16 = arith.remsi %6, %15 : i64
        %17 = arith.cmpi ne, %16, %c0_i64 : i64
        %18 = arith.cmpi slt, %16, %c0_i64 : i64
        %19 = arith.cmpi slt, %15, %c0_i64 : i64
        %20 = arith.xori %18, %19 : i1
        %21 = arith.andi %17, %20 : i1
        %22 = arith.divsi %6, %15 : i64
        %23 = arith.extui %21 : i1 to i64
        %24 = arith.subi %22, %23 : i64
        %25 = arith.cmpi slt, %24, %c1_i64 : i64
        %26 = arith.select %25, %c1_i64, %24 : i64
        %27 = arith.cmpi sgt, %26, %14 : i64
        %28 = arith.select %27, %14, %26 : i64
        %29 = arith.addi %6, %28 : i64
        %30 = arith.subi %29, %c1_i64 : i64
        %31 = arith.remsi %30, %28 : i64
        %32 = arith.cmpi ne, %31, %c0_i64 : i64
        %33 = arith.cmpi slt, %31, %c0_i64 : i64
        %34 = arith.andi %32, %33 : i1
        %35 = arith.divsi %30, %28 : i64
        %36 = arith.extui %34 : i1 to i64
        %37 = arith.subi %35, %36 : i64
        %38 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
        %39 = cannir.extract_shape %38 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
        %40 = cannir.extract_dim %39[1] : !cannir.shape<(-1, -1)>
        %41 = arith.muli %6, %11 : i64
        %42 = arith.muli %6, %40 : i64
        %43 = ascscalar.get_block_idx
        %44 = scf.for %arg5 = %43 to %37 step %15 iter_args(%arg6 = %c0_i64) -> (i64)  : i64 {
          %45 = arith.muli %arg5, %28 : i64
          %46 = arith.subi %6, %45 : i64
          %47 = arith.cmpi sgt, %46, %28 : i64
          %48 = arith.select %47, %28, %46 : i64
          %49 = arith.cmpi eq, %arg6, %c0_i64 : i64
          %50 = ascscalar.memref_address %alloc_3 : memref<32768xbf16, 1>
          %51 = ascscalar.memref_address %alloc_4 : memref<32768xbf16, 1>
          %52 = arith.select %49, %50, %51 : i64
          %53 = arith.cmpi ne, %arg6, %c0_i64 : i64
          %54 = arith.extui %53 : i1 to i64
          %55 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
          %56 = arith.remui %55, %c2_i64 : i64
          %57 = arith.muli %28, %11 : i64
          %58 = arith.maxsi %57, %c1_i64 : i64
          %59 = arith.divsi %41, %58 : i64
          %60 = arith.minsi %arg5, %59 : i64
          %61 = arith.muli %60, %57 : i64
          %62 = arith.subi %41, %61 : i64
          %63 = arith.minsi %57, %62 : i64
          %64 = arith.cmpi sle, %arg5, %59 : i64
          %65 = arith.select %64, %63, %c0_i64 : i64
          %66 = arith.muli %arg5, %57 : i64
          %67 = arith.muli %66, %c2_i64 : i64
          %68 = arith.muli %65, %c2_i64 : i64
          %69 = arith.cmpi eq, %68, %c0_i64 : i64
          %70 = arith.cmpi ne, %68, %c0_i64 : i64
          %71 = arith.extui %70 : i1 to i64
          %72 = arith.select %69, %c2_i64, %68 : i64
          %73 = ascscalar.bind_local_ptr %52 : memref<1xbf16, 1>
          ascvec.get_buf mte2, %54, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%73, %alloc, %c0_i32, %c0_i32, %c0_i64, %67, %71, %72, %c0_i64, %72, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %54, 0 {cannir.buffer.generated_sync}
          %74 = ascscalar.bind_local_ptr %52 : memref<32768xbf16, 1>
          %75 = arith.muli %45, %11 : i64
          %76 = arith.cmpi eq, %13, %c7_i64 : i64
          %77 = arith.select %76, %c128_i64, %c64_i64 : i64
          %78 = arith.muli %48, %77 : i64
          %79 = arith.addi %78, %c255_i64 : i64
          %80 = arith.remsi %79, %c256_i64 : i64
          %81 = arith.cmpi ne, %80, %c0_i64 : i64
          %82 = arith.cmpi slt, %80, %c0_i64 : i64
          %83 = arith.andi %81, %82 : i1
          %84 = arith.divsi %79, %c256_i64 : i64
          %85 = arith.extui %83 : i1 to i64
          %86 = arith.subi %84, %85 : i64
          %87 = arith.divsi %77, %c32_i64 : i64
          %88 = arith.muli %48, %87 : i64
          %89 = arith.trunci %13 : i64 to i32
          %90 = arith.divsi %77, %c4_i64 : i64
          %91 = arith.subi %90, %c1_i64 overflow<nsw> : i64
          %92 = arith.trunci %91 : i64 to i32
          %93 = arith.remsi %11, %c4_i64 : i64
          %94 = arith.cmpi ne, %93, %c0_i64 : i64
          %95 = arith.cmpi slt, %93, %c0_i64 : i64
          %96 = arith.andi %94, %95 : i1
          %97 = arith.divsi %11, %c4_i64 : i64
          %98 = arith.extui %96 : i1 to i64
          %99 = arith.subi %97, %98 : i64
          %100 = arith.trunci %99 : i64 to i32
          %101 = arith.trunci %48 : i64 to i32
          %102 = arith.muli %48, %11 : i64
          %103 = arith.addi %102, %c1023_i64 : i64
          %104 = arith.remsi %103, %c1024_i64 : i64
          %105 = arith.cmpi ne, %104, %c0_i64 : i64
          %106 = arith.cmpi slt, %104, %c0_i64 : i64
          %107 = arith.andi %105, %106 : i1
          %108 = arith.divsi %103, %c1024_i64 : i64
          %109 = arith.extui %107 : i1 to i64
          %110 = arith.subi %108, %109 : i64
          %111 = arith.trunci %1 : i64 to i32
          %112 = arith.extui %111 : i32 to i64
          %113 = arith.subi %1, %112 : i64
          %114 = arith.remsi %113, %c4294967296_i64 : i64
          %115 = arith.cmpi ne, %114, %c0_i64 : i64
          %116 = arith.cmpi slt, %114, %c0_i64 : i64
          %117 = arith.andi %115, %116 : i1
          %118 = arith.divsi %113, %c4294967296_i64 : i64
          %119 = arith.extui %117 : i1 to i64
          %120 = arith.subi %118, %119 : i64
          %121 = arith.trunci %120 : i64 to i32
          %122 = arith.remsi %75, %c16_i64 : i64
          %123 = arith.cmpi ne, %122, %c0_i64 : i64
          %124 = arith.cmpi slt, %122, %c0_i64 : i64
          %125 = arith.andi %123, %124 : i1
          %126 = arith.divsi %75, %c16_i64 : i64
          %127 = arith.extui %125 : i1 to i64
          %128 = arith.subi %126, %127 : i64
          %129 = arith.remsi %11, %c16_i64 : i64
          %130 = arith.cmpi ne, %129, %c0_i64 : i64
          %131 = arith.cmpi slt, %129, %c0_i64 : i64
          %132 = arith.andi %130, %131 : i1
          %133 = arith.divsi %11, %c16_i64 : i64
          %134 = arith.extui %132 : i1 to i64
          %135 = arith.subi %133, %134 : i64
          %136 = arith.remsi %128, %135 : i64
          %137 = arith.cmpi ne, %136, %c0_i64 : i64
          %138 = arith.cmpi slt, %136, %c0_i64 : i64
          %139 = arith.cmpi slt, %135, %c0_i64 : i64
          %140 = arith.xori %138, %139 : i1
          %141 = arith.andi %137, %140 : i1
          %142 = arith.divsi %128, %135 : i64
          %143 = arith.extui %141 : i1 to i64
          %144 = arith.subi %142, %143 : i64
          %145 = arith.select %141, %135, %c0_i64 : i64
          %146 = arith.addi %136, %145 : i64
          %147 = arith.trunci %144 : i64 to i32
          %148 = arith.extui %147 : i32 to i64
          %149 = arith.subi %144, %148 : i64
          %150 = arith.remsi %149, %c4294967296_i64 : i64
          %151 = arith.cmpi ne, %150, %c0_i64 : i64
          %152 = arith.cmpi slt, %150, %c0_i64 : i64
          %153 = arith.andi %151, %152 : i1
          %154 = arith.divsi %149, %c4294967296_i64 : i64
          %155 = arith.extui %153 : i1 to i64
          %156 = arith.subi %154, %155 : i64
          %157 = arith.trunci %156 : i64 to i32
          %158 = arith.trunci %146 : i64 to i32
          %159 = arith.extui %158 : i32 to i64
          %160 = arith.subi %146, %159 : i64
          %161 = arith.remsi %160, %c4294967296_i64 : i64
          %162 = arith.cmpi ne, %161, %c0_i64 : i64
          %163 = arith.cmpi slt, %161, %c0_i64 : i64
          %164 = arith.andi %162, %163 : i1
          %165 = arith.divsi %160, %c4294967296_i64 : i64
          %166 = arith.extui %164 : i1 to i64
          %167 = arith.subi %165, %166 : i64
          %168 = arith.trunci %167 : i64 to i32
          %169 = arith.remsi %c64_i64, %135 : i64
          %170 = arith.cmpi ne, %169, %c0_i64 : i64
          %171 = arith.cmpi slt, %169, %c0_i64 : i64
          %172 = arith.xori %171, %139 : i1
          %173 = arith.andi %170, %172 : i1
          %174 = arith.divsi %c64_i64, %135 : i64
          %175 = arith.extui %173 : i1 to i64
          %176 = arith.subi %174, %175 : i64
          %177 = arith.select %173, %135, %c0_i64 : i64
          %178 = arith.addi %169, %177 : i64
          %179 = arith.trunci %110 : i64 to i32
          ascvec.vf(%135, %3, %111, %121, %alloc_8, %178, %176, %179, %147, %157, %158, %168 : i64, i64, i32, i32, memref<8192xi32, 1>, i64, i64, i32, i32, i32, i32, i32) {
          ^bb0(%arg7: i64, %arg8: i64, %arg9: i32, %arg10: i32, %arg11: memref<8192xi32, 1>, %arg12: i64, %arg13: i64, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32):
            %c1_i32_10 = arith.constant 1 : i32
            %c128_i32 = arith.constant 128 : i32
            %c-1767562579_i32 = arith.constant -1767562579 : i32
            %c-1879881855_i32 = arith.constant -1879881855 : i32
            %c-616729560_i32 = arith.constant -616729560 : i32
            %c-239350328_i32 = arith.constant -239350328 : i32
            %c534103459_i32 = arith.constant 534103459 : i32
            %c1401181199_i32 = arith.constant 1401181199 : i32
            %c1684936478_i32 = arith.constant 1684936478 : i32
            %c-1253254570_i32 = arith.constant -1253254570 : i32
            %c-1459197799_i32 = arith.constant -1459197799 : i32
            %c387276957_i32 = arith.constant 387276957 : i32
            %c-308364780_i32 = arith.constant -308364780 : i32
            %c2027808484_i32 = arith.constant 2027808484 : i32
            %c842468239_i32 = arith.constant 842468239 : i32
            %c-626627285_i32 = arith.constant -626627285 : i32
            %c1993301258_i32_11 = arith.constant 1993301258 : i32
            %c1013904242_i32_12 = arith.constant 1013904242 : i32
            %c-1150833019_i32_13 = arith.constant -1150833019 : i32
            %c-1640531527_i32_14 = arith.constant -1640531527 : i32
            %c-845247145_i32 = arith.constant -845247145 : i32
            %c-766435501_i32 = arith.constant -766435501 : i32
            %c16383_i32 = arith.constant 16383 : i32
            %c1073741824_i32_15 = arith.constant 1073741824 : i32
            %c4_i32_16 = arith.constant 4 : i32
            %c0_i64_17 = arith.constant 0 : i64
            %c0_i32_18 = arith.constant 0 : i32
            %c256_i32 = arith.constant 256 : i32
            %c4294967296_i64_19 = arith.constant 4294967296 : i64
            %212:4 = scf.for %arg19 = %c0_i32_18 to %arg14 step %c1_i32_10 iter_args(%arg20 = %arg15, %arg21 = %arg16, %arg22 = %arg17, %arg23 = %arg18) -> (i32, i32, i32, i32)  : i32 {
              %213 = arith.extui %arg20 : i32 to i64
              %214 = arith.extui %arg21 : i32 to i64
              %215 = arith.muli %214, %c4294967296_i64_19 : i64
              %216 = arith.addi %213, %215 : i64
              %217 = arith.extui %arg22 : i32 to i64
              %218 = arith.extui %arg23 : i32 to i64
              %219 = arith.muli %218, %c4294967296_i64_19 : i64
              %220 = arith.addi %217, %219 : i64
              %221 = arith.muli %arg19, %c256_i32 : i32
              %222 = ascvec.create_mask : !ascvec.mask
              %223 = ascvec.arange %c0_i32_18 : i32 -> <ui32>
              %224 = arith.muli %216, %arg7 : i64
              %225 = arith.addi %224, %220 : i64
              %226 = arith.trunci %225 : i64 to i32
              %227 = arith.extui %226 : i32 to i64
              %228 = arith.subi %225, %227 : i64
              %229 = arith.remsi %228, %c4294967296_i64_19 : i64
              %230 = arith.cmpi ne, %229, %c0_i64_17 : i64
              %231 = arith.cmpi slt, %229, %c0_i64_17 : i64
              %232 = arith.andi %230, %231 : i1
              %233 = arith.divsi %228, %c4294967296_i64_19 : i64
              %234 = arith.extui %232 : i1 to i64
              %235 = arith.subi %233, %234 : i64
              %236 = arith.trunci %235 : i64 to i32
              %237 = ascvec.duplicate_scalar %226, %222 : i32 -> <ui32>
              %238 = ascvec.duplicate_scalar %236, %222 : i32 -> <ui32>
              %239 = ascvec.duplicate_scalar %c0_i32_18, %222 : i32 -> <ui32>
              %carry, %dst = ascvec.addc %237, %223, %222 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
              %carry_20, %dst_21 = ascvec.addc %238, %239, %carry, %222 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
              %240 = arith.trunci %arg8 : i64 to i32
              %241 = arith.extui %240 : i32 to i64
              %242 = arith.subi %arg8, %241 : i64
              %243 = arith.remsi %242, %c4294967296_i64_19 : i64
              %244 = arith.cmpi ne, %243, %c0_i64_17 : i64
              %245 = arith.cmpi slt, %243, %c0_i64_17 : i64
              %246 = arith.andi %244, %245 : i1
              %247 = arith.divsi %242, %c4294967296_i64_19 : i64
              %248 = arith.extui %246 : i1 to i64
              %249 = arith.subi %247, %248 : i64
              %250 = arith.trunci %249 : i64 to i32
              %251 = arith.divui %240, %c4_i32_16 : i32
              %252 = arith.remui %250, %c4_i32_16 : i32
              %253 = arith.muli %252, %c1073741824_i32_15 : i32
              %254 = arith.addi %251, %253 : i32
              %255 = arith.divui %250, %c4_i32_16 : i32
              %256 = ascvec.shr %dst, 14, %222 : <ui32> -> <ui32>
              %257 = ascvec.shl %dst_21, 18, %222 : <ui32> -> <ui32>
              %258 = ascvec.or %256, %257, %222 : <ui32>, <ui32> -> <ui32>
              %259 = ascvec.shr %dst_21, 14, %222 : <ui32> -> <ui32>
              %260 = ascvec.duplicate_scalar %254, %222 : i32 -> <ui32>
              %261 = ascvec.duplicate_scalar %255, %222 : i32 -> <ui32>
              %carry_22, %dst_23 = ascvec.addc %260, %258, %222 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
              %carry_24, %dst_25 = ascvec.addc %261, %259, %carry_22, %222 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
              %262 = ascvec.duplicate_scalar %c16383_i32, %222 : i32 -> <ui32>
              %263 = ascvec.and %dst, %262, %222 : <ui32>, <ui32> -> <ui32>
              %264 = ascvec.duplicate_scalar %c-766435501_i32, %222 : i32 -> <ui32>
              %265 = ascvec.duplicate_scalar %c-845247145_i32, %222 : i32 -> <ui32>
              %dst0, %dst1 = ascvec.mull %239, %264, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_26, %dst1_27 = ascvec.mull %dst_23, %265, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %266 = ascvec.xor %dst1_27, %263, %222 : <ui32>, <ui32> -> <ui32>
              %267 = ascvec.xor %dst1, %dst_25, %222 : <ui32>, <ui32> -> <ui32>
              %268 = ascvec.duplicate_scalar %arg9, %222 : i32 -> <ui32>
              %269 = ascvec.duplicate_scalar %arg10, %222 : i32 -> <ui32>
              %270 = ascvec.xor %266, %268, %222 : <ui32>, <ui32> -> <ui32>
              %271 = ascvec.xor %267, %269, %222 : <ui32>, <ui32> -> <ui32>
              %dst0_28, %dst1_29 = ascvec.mull %270, %264, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_30, %dst1_31 = ascvec.mull %271, %265, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %272 = ascvec.xor %dst1_31, %dst0_26, %222 : <ui32>, <ui32> -> <ui32>
              %273 = ascvec.xor %dst1_29, %dst0, %222 : <ui32>, <ui32> -> <ui32>
              %274 = arith.addi %arg9, %c-1640531527_i32_14 : i32
              %275 = ascvec.duplicate_scalar %274, %222 : i32 -> <ui32>
              %276 = arith.addi %arg10, %c-1150833019_i32_13 : i32
              %277 = ascvec.duplicate_scalar %276, %222 : i32 -> <ui32>
              %278 = ascvec.xor %272, %275, %222 : <ui32>, <ui32> -> <ui32>
              %279 = ascvec.xor %273, %277, %222 : <ui32>, <ui32> -> <ui32>
              %dst0_32, %dst1_33 = ascvec.mull %278, %264, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_34, %dst1_35 = ascvec.mull %279, %265, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %280 = ascvec.xor %dst1_35, %dst0_30, %222 : <ui32>, <ui32> -> <ui32>
              %281 = ascvec.xor %dst1_33, %dst0_28, %222 : <ui32>, <ui32> -> <ui32>
              %282 = arith.addi %arg9, %c1013904242_i32_12 : i32
              %283 = ascvec.duplicate_scalar %282, %222 : i32 -> <ui32>
              %284 = arith.addi %arg10, %c1993301258_i32_11 : i32
              %285 = ascvec.duplicate_scalar %284, %222 : i32 -> <ui32>
              %286 = ascvec.xor %280, %283, %222 : <ui32>, <ui32> -> <ui32>
              %287 = ascvec.xor %281, %285, %222 : <ui32>, <ui32> -> <ui32>
              %dst0_36, %dst1_37 = ascvec.mull %286, %264, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_38, %dst1_39 = ascvec.mull %287, %265, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %288 = ascvec.xor %dst1_39, %dst0_34, %222 : <ui32>, <ui32> -> <ui32>
              %289 = ascvec.xor %dst1_37, %dst0_32, %222 : <ui32>, <ui32> -> <ui32>
              %290 = arith.addi %arg9, %c-626627285_i32 : i32
              %291 = ascvec.duplicate_scalar %290, %222 : i32 -> <ui32>
              %292 = arith.addi %arg10, %c842468239_i32 : i32
              %293 = ascvec.duplicate_scalar %292, %222 : i32 -> <ui32>
              %294 = ascvec.xor %288, %291, %222 : <ui32>, <ui32> -> <ui32>
              %295 = ascvec.xor %289, %293, %222 : <ui32>, <ui32> -> <ui32>
              %dst0_40, %dst1_41 = ascvec.mull %294, %264, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_42, %dst1_43 = ascvec.mull %295, %265, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %296 = ascvec.xor %dst1_43, %dst0_38, %222 : <ui32>, <ui32> -> <ui32>
              %297 = ascvec.xor %dst1_41, %dst0_36, %222 : <ui32>, <ui32> -> <ui32>
              %298 = arith.addi %arg9, %c2027808484_i32 : i32
              %299 = ascvec.duplicate_scalar %298, %222 : i32 -> <ui32>
              %300 = arith.addi %arg10, %c-308364780_i32 : i32
              %301 = ascvec.duplicate_scalar %300, %222 : i32 -> <ui32>
              %302 = ascvec.xor %296, %299, %222 : <ui32>, <ui32> -> <ui32>
              %303 = ascvec.xor %297, %301, %222 : <ui32>, <ui32> -> <ui32>
              %dst0_44, %dst1_45 = ascvec.mull %302, %264, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_46, %dst1_47 = ascvec.mull %303, %265, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %304 = ascvec.xor %dst1_47, %dst0_42, %222 : <ui32>, <ui32> -> <ui32>
              %305 = ascvec.xor %dst1_45, %dst0_40, %222 : <ui32>, <ui32> -> <ui32>
              %306 = arith.addi %arg9, %c387276957_i32 : i32
              %307 = ascvec.duplicate_scalar %306, %222 : i32 -> <ui32>
              %308 = arith.addi %arg10, %c-1459197799_i32 : i32
              %309 = ascvec.duplicate_scalar %308, %222 : i32 -> <ui32>
              %310 = ascvec.xor %304, %307, %222 : <ui32>, <ui32> -> <ui32>
              %311 = ascvec.xor %305, %309, %222 : <ui32>, <ui32> -> <ui32>
              %dst0_48, %dst1_49 = ascvec.mull %310, %264, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_50, %dst1_51 = ascvec.mull %311, %265, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %312 = ascvec.xor %dst1_51, %dst0_46, %222 : <ui32>, <ui32> -> <ui32>
              %313 = ascvec.xor %dst1_49, %dst0_44, %222 : <ui32>, <ui32> -> <ui32>
              %314 = arith.addi %arg9, %c-1253254570_i32 : i32
              %315 = ascvec.duplicate_scalar %314, %222 : i32 -> <ui32>
              %316 = arith.addi %arg10, %c1684936478_i32 : i32
              %317 = ascvec.duplicate_scalar %316, %222 : i32 -> <ui32>
              %318 = ascvec.xor %312, %315, %222 : <ui32>, <ui32> -> <ui32>
              %319 = ascvec.xor %313, %317, %222 : <ui32>, <ui32> -> <ui32>
              %dst0_52, %dst1_53 = ascvec.mull %318, %264, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_54, %dst1_55 = ascvec.mull %319, %265, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %320 = ascvec.xor %dst1_55, %dst0_50, %222 : <ui32>, <ui32> -> <ui32>
              %321 = ascvec.xor %dst1_53, %dst0_48, %222 : <ui32>, <ui32> -> <ui32>
              %322 = arith.addi %arg9, %c1401181199_i32 : i32
              %323 = ascvec.duplicate_scalar %322, %222 : i32 -> <ui32>
              %324 = arith.addi %arg10, %c534103459_i32 : i32
              %325 = ascvec.duplicate_scalar %324, %222 : i32 -> <ui32>
              %326 = ascvec.xor %320, %323, %222 : <ui32>, <ui32> -> <ui32>
              %327 = ascvec.xor %321, %325, %222 : <ui32>, <ui32> -> <ui32>
              %dst0_56, %dst1_57 = ascvec.mull %326, %264, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_58, %dst1_59 = ascvec.mull %327, %265, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %328 = ascvec.xor %dst1_59, %dst0_54, %222 : <ui32>, <ui32> -> <ui32>
              %329 = ascvec.xor %dst1_57, %dst0_52, %222 : <ui32>, <ui32> -> <ui32>
              %330 = arith.addi %arg9, %c-239350328_i32 : i32
              %331 = ascvec.duplicate_scalar %330, %222 : i32 -> <ui32>
              %332 = arith.addi %arg10, %c-616729560_i32 : i32
              %333 = ascvec.duplicate_scalar %332, %222 : i32 -> <ui32>
              %334 = ascvec.xor %328, %331, %222 : <ui32>, <ui32> -> <ui32>
              %335 = ascvec.xor %329, %333, %222 : <ui32>, <ui32> -> <ui32>
              %dst0_60, %dst1_61 = ascvec.mull %334, %264, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_62, %dst1_63 = ascvec.mull %335, %265, %222 : <ui32>, <ui32> -> <ui32>, <ui32>
              %336 = ascvec.xor %dst1_63, %dst0_58, %222 : <ui32>, <ui32> -> <ui32>
              %337 = ascvec.xor %dst1_61, %dst0_56, %222 : <ui32>, <ui32> -> <ui32>
              %338 = arith.addi %arg9, %c-1879881855_i32 : i32
              %339 = ascvec.duplicate_scalar %338, %222 : i32 -> <ui32>
              %340 = arith.addi %arg10, %c-1767562579_i32 : i32
              %341 = ascvec.duplicate_scalar %340, %222 : i32 -> <ui32>
              %342 = ascvec.xor %336, %339, %222 : <ui32>, <ui32> -> <ui32>
              %343 = ascvec.xor %337, %341, %222 : <ui32>, <ui32> -> <ui32>
              %low, %high = ascvec.intlv %342, %343 : <ui32>, <ui32> -> <ui32>, <ui32>
              %low_64, %high_65 = ascvec.intlv %dst0_62, %dst0_60 : <ui32>, <ui32> -> <ui32>, <ui32>
              %344 = arith.index_cast %221 : i32 to index
              ascvec.storealign_intlv %arg11[%344], %low, %low_64 : memref<8192xi32, 1>, <ui32>, <ui32>
              %345 = arith.addi %221, %c128_i32 : i32
              %346 = arith.index_cast %345 : i32 to index
              ascvec.storealign_intlv %arg11[%346], %high, %high_65 : memref<8192xi32, 1>, <ui32>, <ui32>
              %347 = arith.addi %220, %arg12 : i64
              %348 = arith.cmpi sge, %347, %arg7 : i64
              %349 = arith.extui %348 : i1 to i64
              %350 = arith.addi %216, %arg13 : i64
              %351 = arith.addi %350, %349 : i64
              %352 = arith.muli %349, %arg7 : i64
              %353 = arith.subi %347, %352 : i64
              %354 = arith.trunci %351 : i64 to i32
              %355 = arith.extui %354 : i32 to i64
              %356 = arith.subi %351, %355 : i64
              %357 = arith.remsi %356, %c4294967296_i64_19 : i64
              %358 = arith.cmpi ne, %357, %c0_i64_17 : i64
              %359 = arith.cmpi slt, %357, %c0_i64_17 : i64
              %360 = arith.andi %358, %359 : i1
              %361 = arith.divsi %356, %c4294967296_i64_19 : i64
              %362 = arith.extui %360 : i1 to i64
              %363 = arith.subi %361, %362 : i64
              %364 = arith.trunci %363 : i64 to i32
              %365 = arith.trunci %353 : i64 to i32
              %366 = arith.extui %365 : i32 to i64
              %367 = arith.subi %353, %366 : i64
              %368 = arith.remsi %367, %c4294967296_i64_19 : i64
              %369 = arith.cmpi ne, %368, %c0_i64_17 : i64
              %370 = arith.cmpi slt, %368, %c0_i64_17 : i64
              %371 = arith.andi %369, %370 : i1
              %372 = arith.divsi %367, %c4294967296_i64_19 : i64
              %373 = arith.extui %371 : i1 to i64
              %374 = arith.subi %372, %373 : i64
              %375 = arith.trunci %374 : i64 to i32
              scf.yield %354, %364, %365, %375 : i32, i32, i32, i32
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          %180 = arith.trunci %86 : i64 to i32
          %181 = arith.trunci %88 : i64 to i32
          ascvec.get_buf v, %54, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
          %182 = ascscalar.bitcast_memref %74 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui32, 1>
          %183 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<32768xui8, 1> -> memref<?xui32, 1>
          ascvec.vf(%89, %92, %100, %101, %182, %alloc_8, %183, %alloc_7, %180, %181, %alloc_6 : i32, i32, i32, i32, memref<?xui32, 1>, memref<8192xi32, 1>, memref<?xui32, 1>, memref<1024xui32, 1>, i32, i32, memref<2048xui8, 1>) {
          ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: memref<?xui32, 1>, %arg12: memref<8192xi32, 1>, %arg13: memref<?xui32, 1>, %arg14: memref<1024xui32, 1>, %arg15: i32, %arg16: i32, %arg17: memref<2048xui8, 1>):
            %c2139095040_i32 = arith.constant 2139095040 : i32
            %c127_i32 = arith.constant 127 : i32
            %c123_i32 = arith.constant 123 : i32
            %c947912704_i32 = arith.constant 947912704 : i32
            %cst = arith.constant 4.2949673E+9 : f32
            %c29360128_i32 = arith.constant 29360128 : i32
            %c128_i32 = arith.constant 128 : i32
            %c65535_i32 = arith.constant 65535 : i32
            %c2139160576_i32 = arith.constant 2139160576 : i32
            %c254_i32 = arith.constant 254 : i32
            %c-15_i32 = arith.constant -15 : i32
            %c15_i32 = arith.constant 15 : i32
            %c255_i32 = arith.constant 255 : i32
            %c2147483647_i32 = arith.constant 2147483647 : i32
            %c-65536_i32 = arith.constant -65536 : i32
            %c1_i32_10 = arith.constant 1 : i32
            %c0_11 = arith.constant 0 : index
            %c64_i32 = arith.constant 64 : i32
            %c7_i32 = arith.constant 7 : i32
            %c8_i32 = arith.constant 8 : i32
            %c0_i32_12 = arith.constant 0 : i32
            %212 = ascvec.create_mask : !ascvec.mask
            %213 = ascvec.arange %c0_i32_12 : i32 -> <ui32>
            %mask, %remaining_out = ascvec.update_mask %c8_i32 : !ascvec.mask
            %214 = ascvec.shr %213, 3, %212 : <ui32> -> <ui32>
            %215 = ascvec.duplicate_scalar %c0_i32_12, %212 : i32 -> <ui32>
            %216 = ascvec.duplicate_scalar %arg7, %212 : i32 -> <ui32>
            %217 = ascvec.cmp_eq_scalar %216, %c7_i32, %212 : <ui32>, i32 -> !ascvec.mask
            scf.for %arg18 = %c0_i32_12 to %arg15 step %c1_i32_10  : i32 {
              %218 = arith.muli %arg18, %c64_i32 : i32
              %219 = ascvec.add_scalar %213, %218, %212 : <ui32>, i32
              %220 = ascvec.shr %219, 5, %212 : <ui32> -> <ui32>
              %221 = ascvec.shr %219, 4, %212 : <ui32> -> <ui32>
              %222 = ascvec.select %220, %221, %217 : <ui32>
              %223 = ascvec.duplicate_scalar %arg8, %212 : i32 -> <ui32>
              %224 = ascvec.and %219, %223, %212 : <ui32>, <ui32> -> <ui32>
              %225 = ascvec.mul_scalar %222, %arg9, %212 : <ui32>, i32
              %226 = ascvec.add %225, %224, %212 : <ui32>
              %227 = ascvec.cmp_lt_scalar %222, %arg10, %212 : <ui32>, i32 -> !ascvec.mask
              %228 = ascvec.cmp_lt_scalar %224, %arg9, %212 : <ui32>, i32 -> !ascvec.mask
              %229 = ascvec.mask_and %227, %228, %212
              %230 = ascvec.shl %226, 1, %212 : <ui32> -> <ui32>
              %231 = ascvec.gather %arg11[%c0_11], %230, %229 : memref<?xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
              %232 = ascvec.add_scalar %230, %c1_i32_10, %212 : <ui32>, i32
              %233 = ascvec.gather %arg11[%c0_11], %232, %229 : memref<?xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
              %234 = ascvec.select %231, %215, %229 : <ui32>
              %235 = ascvec.select %233, %215, %229 : <ui32>
              %236 = ascvec.shl %234, 16, %212 : <ui32> -> <ui32>
              %237 = ascvec.reinterpret %236 : <ui32> -> <f32>
              %238 = ascvec.duplicate_scalar %c-65536_i32, %212 : i32 -> <ui32>
              %239 = ascvec.and %234, %238, %212 : <ui32>, <ui32> -> <ui32>
              %240 = ascvec.reinterpret %239 : <ui32> -> <f32>
              %241 = ascvec.shl %235, 16, %212 : <ui32> -> <ui32>
              %242 = ascvec.reinterpret %241 : <ui32> -> <f32>
              %243 = ascvec.and %235, %238, %212 : <ui32>, <ui32> -> <ui32>
              %244 = ascvec.reinterpret %243 : <ui32> -> <f32>
              %245 = ascvec.reinterpret %237 : <f32> -> <ui32>
              %246 = ascvec.duplicate_scalar %c2147483647_i32, %212 : i32 -> <ui32>
              %247 = ascvec.and %245, %246, %212 : <ui32>, <ui32> -> <ui32>
              %248 = ascvec.reinterpret %240 : <f32> -> <ui32>
              %249 = ascvec.and %248, %246, %212 : <ui32>, <ui32> -> <ui32>
              %250 = ascvec.reinterpret %242 : <f32> -> <ui32>
              %251 = ascvec.and %250, %246, %212 : <ui32>, <ui32> -> <ui32>
              %252 = ascvec.reinterpret %244 : <f32> -> <ui32>
              %253 = ascvec.and %252, %246, %212 : <ui32>, <ui32> -> <ui32>
              %254 = ascvec.max %247, %249, %212 : <ui32>
              %255 = ascvec.max %251, %253, %212 : <ui32>
              %256 = ascvec.max %254, %255, %212 : <ui32>
              %257 = ascvec.reduce_max_datablock %256, %212 : <ui32>
              %258 = ascvec.shr %257, 23, %212 : <ui32> -> <ui32>
              %259 = ascvec.cmp_ge_scalar %258, %c255_i32, %212 : <ui32>, i32 -> !ascvec.mask
              %260 = ascvec.max_scalar %258, %c15_i32, %212 : <ui32>, i32
              %261 = ascvec.add_scalar %260, %c-15_i32, %212 : <ui32>, i32
              %262 = ascvec.duplicate_scalar %c255_i32, %259 merge %261 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
              %263 = ascvec.duplicate_scalar %c254_i32, %212 : i32 -> <ui32>
              %264 = ascvec.sub %263, %262, %212 : <ui32>
              %265 = ascvec.cmp_ne_scalar %262, %c0_i32_12, %212 : <ui32>, i32 -> !ascvec.mask
              %266 = ascvec.shl %264, 23, %265 : <ui32> -> <ui32>
              %267 = ascvec.duplicate_scalar %c2139160576_i32, %259 merge %266 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
              %268 = ascvec.reinterpret %267 : <ui32> -> <f32>
              %269 = ascvec.gather_reg %268, %214 : <f32>, <ui32> -> <f32>
              %270 = ascvec.mul %237, %269, %212 : <f32>
              %271 = ascvec.mul %240, %269, %212 : <f32>
              %272 = ascvec.mul %242, %269, %212 : <f32>
              %273 = ascvec.mul %244, %269, %212 : <f32>
              %274 = ascvec.gather %arg12[%c0_11], %226, %229 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
              %275 = ascvec.reinterpret %274 : <i32> -> <ui32>
              %276 = ascvec.shl %275, 8, %229 : <ui32> -> <ui32>
              %277 = ascvec.shr %275, 24, %229 : <ui32> -> <ui32>
              %278 = ascvec.or %276, %277, %229 : <ui32>, <ui32> -> <ui32>
              %279 = ascvec.shr %275, 16, %212 : <ui32> -> <ui32>
              %280 = ascvec.duplicate_scalar %c65535_i32, %212 : i32 -> <ui32>
              %281 = ascvec.and %275, %280, %212 : <ui32>, <ui32> -> <ui32>
              %282 = ascvec.reinterpret %270 : <f32> -> <ui32>
              %283 = ascvec.shr %282, 24, %212 : <ui32> -> <ui32>
              %284 = ascvec.duplicate_scalar %c128_i32, %212 : i32 -> <ui32>
              %285 = ascvec.and %283, %284, %212 : <ui32>, <ui32> -> <ui32>
              %286 = ascvec.and %282, %246, %212 : <ui32>, <ui32> -> <ui32>
              %287 = ascvec.shr %286, 5, %212 : <ui32> -> <ui32>
              %288 = ascvec.duplicate_scalar %c29360128_i32, %212 : i32 -> <ui32>
              %289 = ascvec.sub %287, %288, %212 : <ui32>
              %290 = ascvec.reinterpret %286 : <ui32> -> <f32>
              %291 = ascvec.mul_scalar %290, %cst, %212 : <f32>, f32
              %292 = ascvec.cast %291, %212 {rounding = 4 : i32} : <f32> -> <i32>
              %293 = ascvec.reinterpret %292 : <i32> -> <ui32>
              %294 = ascvec.cmp_ge_scalar %286, %c947912704_i32, %212 : <ui32>, i32 -> !ascvec.mask
              %295 = ascvec.select %289, %293, %294 : <ui32>
              %296 = ascvec.and %279, %280, %212 : <ui32>, <ui32> -> <ui32>
              %297 = ascvec.add %295, %296, %212 : <ui32>
              %298 = ascvec.shr %297, 16, %212 : <ui32> -> <ui32>
              %299 = ascvec.min_scalar %298, %c123_i32, %212 : <ui32>, i32
              %300 = ascvec.or %299, %285, %212 : <ui32>, <ui32> -> <ui32>
              %301 = ascvec.duplicate_scalar %c127_i32, %212 : i32 -> <ui32>
              %302 = ascvec.cmp_gt_scalar %286, %c2139095040_i32, %212 : <ui32>, i32 -> !ascvec.mask
              %303 = ascvec.select %301, %300, %302 : <ui32>
              %304 = ascvec.and %278, %280, %212 : <ui32>, <ui32> -> <ui32>
              %305 = ascvec.reinterpret %271 : <f32> -> <ui32>
              %306 = ascvec.shr %305, 24, %212 : <ui32> -> <ui32>
              %307 = ascvec.and %306, %284, %212 : <ui32>, <ui32> -> <ui32>
              %308 = ascvec.and %305, %246, %212 : <ui32>, <ui32> -> <ui32>
              %309 = ascvec.shr %308, 5, %212 : <ui32> -> <ui32>
              %310 = ascvec.sub %309, %288, %212 : <ui32>
              %311 = ascvec.reinterpret %308 : <ui32> -> <f32>
              %312 = ascvec.mul_scalar %311, %cst, %212 : <f32>, f32
              %313 = ascvec.cast %312, %212 {rounding = 4 : i32} : <f32> -> <i32>
              %314 = ascvec.reinterpret %313 : <i32> -> <ui32>
              %315 = ascvec.cmp_ge_scalar %308, %c947912704_i32, %212 : <ui32>, i32 -> !ascvec.mask
              %316 = ascvec.select %310, %314, %315 : <ui32>
              %317 = ascvec.and %304, %280, %212 : <ui32>, <ui32> -> <ui32>
              %318 = ascvec.add %316, %317, %212 : <ui32>
              %319 = ascvec.shr %318, 16, %212 : <ui32> -> <ui32>
              %320 = ascvec.min_scalar %319, %c123_i32, %212 : <ui32>, i32
              %321 = ascvec.or %320, %307, %212 : <ui32>, <ui32> -> <ui32>
              %322 = ascvec.cmp_gt_scalar %308, %c2139095040_i32, %212 : <ui32>, i32 -> !ascvec.mask
              %323 = ascvec.select %301, %321, %322 : <ui32>
              %324 = ascvec.reinterpret %272 : <f32> -> <ui32>
              %325 = ascvec.shr %324, 24, %212 : <ui32> -> <ui32>
              %326 = ascvec.and %325, %284, %212 : <ui32>, <ui32> -> <ui32>
              %327 = ascvec.and %324, %246, %212 : <ui32>, <ui32> -> <ui32>
              %328 = ascvec.shr %327, 5, %212 : <ui32> -> <ui32>
              %329 = ascvec.sub %328, %288, %212 : <ui32>
              %330 = ascvec.reinterpret %327 : <ui32> -> <f32>
              %331 = ascvec.mul_scalar %330, %cst, %212 : <f32>, f32
              %332 = ascvec.cast %331, %212 {rounding = 4 : i32} : <f32> -> <i32>
              %333 = ascvec.reinterpret %332 : <i32> -> <ui32>
              %334 = ascvec.cmp_ge_scalar %327, %c947912704_i32, %212 : <ui32>, i32 -> !ascvec.mask
              %335 = ascvec.select %329, %333, %334 : <ui32>
              %336 = ascvec.and %281, %280, %212 : <ui32>, <ui32> -> <ui32>
              %337 = ascvec.add %335, %336, %212 : <ui32>
              %338 = ascvec.shr %337, 16, %212 : <ui32> -> <ui32>
              %339 = ascvec.min_scalar %338, %c123_i32, %212 : <ui32>, i32
              %340 = ascvec.or %339, %326, %212 : <ui32>, <ui32> -> <ui32>
              %341 = ascvec.cmp_gt_scalar %327, %c2139095040_i32, %212 : <ui32>, i32 -> !ascvec.mask
              %342 = ascvec.select %301, %340, %341 : <ui32>
              %343 = ascvec.shr %278, 16, %212 : <ui32> -> <ui32>
              %344 = ascvec.reinterpret %273 : <f32> -> <ui32>
              %345 = ascvec.shr %344, 24, %212 : <ui32> -> <ui32>
              %346 = ascvec.and %345, %284, %212 : <ui32>, <ui32> -> <ui32>
              %347 = ascvec.and %344, %246, %212 : <ui32>, <ui32> -> <ui32>
              %348 = ascvec.shr %347, 5, %212 : <ui32> -> <ui32>
              %349 = ascvec.sub %348, %288, %212 : <ui32>
              %350 = ascvec.reinterpret %347 : <ui32> -> <f32>
              %351 = ascvec.mul_scalar %350, %cst, %212 : <f32>, f32
              %352 = ascvec.cast %351, %212 {rounding = 4 : i32} : <f32> -> <i32>
              %353 = ascvec.reinterpret %352 : <i32> -> <ui32>
              %354 = ascvec.cmp_ge_scalar %347, %c947912704_i32, %212 : <ui32>, i32 -> !ascvec.mask
              %355 = ascvec.select %349, %353, %354 : <ui32>
              %356 = ascvec.and %343, %280, %212 : <ui32>, <ui32> -> <ui32>
              %357 = ascvec.add %355, %356, %212 : <ui32>
              %358 = ascvec.shr %357, 16, %212 : <ui32> -> <ui32>
              %359 = ascvec.min_scalar %358, %c123_i32, %212 : <ui32>, i32
              %360 = ascvec.or %359, %346, %212 : <ui32>, <ui32> -> <ui32>
              %361 = ascvec.cmp_gt_scalar %347, %c2139095040_i32, %212 : <ui32>, i32 -> !ascvec.mask
              %362 = ascvec.select %301, %360, %361 : <ui32>
              %363 = ascvec.shl %323, 8, %212 : <ui32> -> <ui32>
              %364 = ascvec.or %303, %363, %212 : <ui32>, <ui32> -> <ui32>
              %365 = ascvec.shl %342, 16, %212 : <ui32> -> <ui32>
              %366 = ascvec.shl %362, 24, %212 : <ui32> -> <ui32>
              %367 = ascvec.or %365, %366, %212 : <ui32>, <ui32> -> <ui32>
              %368 = ascvec.or %364, %367, %212 : <ui32>, <ui32> -> <ui32>
              ascvec.scatter %arg13[%c0_11], %368, %226, %229 : memref<?xui32, 1>, <ui32>, <ui32>, !ascvec.mask
              %369 = arith.muli %arg18, %c8_i32 : i32
              %370 = arith.index_cast %369 : i32 to index
              ascvec.storealign %arg14[%370], %262, %mask : memref<1024xui32, 1>, index, <ui32>
            }
            ascvec.mem_bar vst_vld
            scf.for %arg18 = %c0_i32_12 to %arg16 step %c64_i32  : i32 {
              %218 = arith.index_cast %arg18 : i32 to index
              %219 = ascvec.loadalign %arg14[%218] : memref<1024xui32, 1>, index -> <ui32>
              %220 = arith.subi %arg16, %arg18 : i32
              %mask_13, %remaining_out_14 = ascvec.update_mask %220 : !ascvec.mask
              ascvec.storealign_pack %arg17[%218], %219, %mask_13 {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %54, 0 {cannir.buffer.generated_sync}
          %184 = arith.cmpi sgt, %65, %c32768_i64 : i64
          %185 = arith.select %184, %c32768_i64, %65 : i64
          %186 = arith.muli %66, %c8_i64 : i64
          %187 = ascscalar.memref_address %alloc_0 : memref<1xui8>
          %188 = arith.divui %186, %c8_i64 : i64
          %189 = arith.addi %187, %188 : i64
          %190 = ascscalar.bind_gm_ptr %189 : memref<1xui8>
          %191 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<32768xui8, 1> -> memref<?xui8, 1>
          %192 = arith.cmpi ugt, %185, %c0_i64 : i64
          scf.if %192 {
            ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
            %212 = arith.trunci %185 : i64 to i32
            ascvec.copy_ub2gm %190[%c0], %191[%c0], %c1_i32, %212, %41, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
          }
          %193 = arith.muli %28, %40 : i64
          %194 = arith.maxsi %193, %c1_i64 : i64
          %195 = arith.divsi %42, %194 : i64
          %196 = arith.minsi %arg5, %195 : i64
          %197 = arith.muli %196, %193 : i64
          %198 = arith.subi %42, %197 : i64
          %199 = arith.minsi %193, %198 : i64
          %200 = arith.cmpi sle, %arg5, %195 : i64
          %201 = arith.select %200, %199, %c0_i64 : i64
          %202 = arith.cmpi sgt, %201, %c2048_i64 : i64
          %203 = arith.select %202, %c2048_i64, %201 : i64
          %204 = arith.muli %arg5, %193 : i64
          %205 = arith.muli %204, %c8_i64 : i64
          %206 = ascscalar.memref_address %alloc_1 : memref<1xui8>
          %207 = arith.divui %205, %c8_i64 : i64
          %208 = arith.addi %206, %207 : i64
          %209 = ascscalar.bind_gm_ptr %208 : memref<1xui8>
          %210 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<2048xui8, 1> -> memref<?xui8, 1>
          %211 = arith.cmpi ugt, %203, %c0_i64 : i64
          scf.if %211 {
            ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
            %212 = arith.trunci %203 : i64 to i32
            ascvec.copy_ub2gm %209[%c0], %210[%c0], %c1_i32, %212, %42, %c2048_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %56 : i64
        }
      } else {
        %11 = cannir.extract_dim %5[1] : !cannir.shape<(-1, -1)>
        %12 = arith.addi %11, %c63_i64 : i64
        %13 = arith.remsi %12, %c64_i64 : i64
        %14 = arith.cmpi ne, %13, %c0_i64 : i64
        %15 = arith.cmpi slt, %13, %c0_i64 : i64
        %16 = arith.andi %14, %15 : i1
        %17 = arith.divsi %12, %c64_i64 : i64
        %18 = arith.extui %16 : i1 to i64
        %19 = arith.subi %17, %18 : i64
        %20 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
        %21 = cannir.extract_shape %20 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
        %22 = cannir.extract_dim %21[1] : !cannir.shape<(-1, -1)>
        %23 = arith.remsi %11, %c16_i64 : i64
        %24 = arith.cmpi ne, %23, %c0_i64 : i64
        %25 = arith.cmpi slt, %23, %c0_i64 : i64
        %26 = arith.andi %24, %25 : i1
        %27 = arith.divsi %11, %c16_i64 : i64
        %28 = arith.extui %26 : i1 to i64
        %29 = arith.subi %27, %28 : i64
        %30 = arith.cmpi eq, %22, %29 : i64
        %31 = arith.select %30, %c4_i64, %c2_i64 : i64
        %32 = arith.muli %6, %19 : i64
        %33 = ascscalar.get_block_num
        %34 = arith.addi %32, %33 : i64
        %35 = arith.subi %34, %c1_i64 : i64
        %36 = arith.remsi %35, %33 : i64
        %37 = arith.cmpi ne, %36, %c0_i64 : i64
        %38 = arith.cmpi slt, %36, %c0_i64 : i64
        %39 = arith.cmpi slt, %33, %c0_i64 : i64
        %40 = arith.xori %38, %39 : i1
        %41 = arith.andi %37, %40 : i1
        %42 = arith.divsi %35, %33 : i64
        %43 = arith.extui %41 : i1 to i64
        %44 = arith.subi %42, %43 : i64
        %45 = arith.addi %44, %c511_i64 : i64
        %46 = arith.remsi %45, %c512_i64 : i64
        %47 = arith.cmpi ne, %46, %c0_i64 : i64
        %48 = arith.cmpi slt, %46, %c0_i64 : i64
        %49 = arith.andi %47, %48 : i1
        %50 = arith.divsi %45, %c512_i64 : i64
        %51 = arith.extui %49 : i1 to i64
        %52 = arith.subi %50, %51 : i64
        %53 = arith.addi %44, %52 : i64
        %54 = arith.subi %53, %c1_i64 : i64
        %55 = arith.remsi %54, %52 : i64
        %56 = arith.cmpi ne, %55, %c0_i64 : i64
        %57 = arith.cmpi slt, %55, %c0_i64 : i64
        %58 = arith.cmpi slt, %52, %c0_i64 : i64
        %59 = arith.xori %57, %58 : i1
        %60 = arith.andi %56, %59 : i1
        %61 = arith.divsi %54, %52 : i64
        %62 = arith.extui %60 : i1 to i64
        %63 = arith.subi %61, %62 : i64
        %64 = arith.cmpi sge, %63, %c4_i64 : i64
        %65 = scf.if %64 -> (i64) {
          %84 = arith.addi %63, %c3_i64 : i64
          %85 = arith.remsi %84, %c4_i64 : i64
          %86 = arith.cmpi ne, %85, %c0_i64 : i64
          %87 = arith.cmpi slt, %85, %c0_i64 : i64
          %88 = arith.andi %86, %87 : i1
          %89 = arith.divsi %84, %c4_i64 : i64
          %90 = arith.extui %88 : i1 to i64
          %91 = arith.subi %89, %90 : i64
          %92 = arith.muli %91, %c4_i64 : i64
          scf.yield %92 : i64
        } else {
          scf.yield %63 : i64
        }
        %66 = arith.muli %33, %c64_i64 : i64
        %67 = arith.cmpi sge, %32, %66 : i64
        %68 = scf.if %67 -> (i64) {
          %84 = arith.addi %65, %c15_i64 : i64
          %85 = arith.remsi %84, %c16_i64 : i64
          %86 = arith.cmpi ne, %85, %c0_i64 : i64
          %87 = arith.cmpi slt, %85, %c0_i64 : i64
          %88 = arith.andi %86, %87 : i1
          %89 = arith.divsi %84, %c16_i64 : i64
          %90 = arith.extui %88 : i1 to i64
          %91 = arith.subi %89, %90 : i64
          %92 = arith.muli %91, %c16_i64 : i64
          scf.yield %92 : i64
        } else {
          scf.yield %65 : i64
        }
        %69 = arith.muli %68, %c64_i64 : i64
        %70 = arith.addi %19, %68 : i64
        %71 = arith.subi %70, %c1_i64 : i64
        %72 = arith.remsi %71, %68 : i64
        %73 = arith.cmpi ne, %72, %c0_i64 : i64
        %74 = arith.cmpi slt, %72, %c0_i64 : i64
        %75 = arith.cmpi slt, %68, %c0_i64 : i64
        %76 = arith.xori %74, %75 : i1
        %77 = arith.andi %73, %76 : i1
        %78 = arith.divsi %71, %68 : i64
        %79 = arith.extui %77 : i1 to i64
        %80 = arith.subi %78, %79 : i64
        %81 = arith.muli %6, %80 : i64
        %82 = ascscalar.get_block_idx
        %83 = scf.for %arg5 = %82 to %81 step %33 iter_args(%arg6 = %c0_i64) -> (i64)  : i64 {
          %84 = arith.remsi %arg5, %80 : i64
          %85 = arith.cmpi ne, %84, %c0_i64 : i64
          %86 = arith.cmpi slt, %84, %c0_i64 : i64
          %87 = arith.cmpi slt, %80, %c0_i64 : i64
          %88 = arith.xori %86, %87 : i1
          %89 = arith.andi %85, %88 : i1
          %90 = arith.divsi %arg5, %80 : i64
          %91 = arith.extui %89 : i1 to i64
          %92 = arith.subi %90, %91 : i64
          %93 = arith.select %89, %80, %c0_i64 : i64
          %94 = arith.addi %84, %93 : i64
          %95 = arith.muli %94, %69 : i64
          %96 = arith.subi %11, %95 : i64
          %97 = arith.cmpi sgt, %96, %69 : i64
          %98 = arith.select %97, %69, %96 : i64
          %99 = arith.cmpi eq, %arg6, %c0_i64 : i64
          %100 = ascscalar.memref_address %alloc_3 : memref<32768xbf16, 1>
          %101 = ascscalar.memref_address %alloc_4 : memref<32768xbf16, 1>
          %102 = arith.select %99, %100, %101 : i64
          %103 = arith.cmpi ne, %arg6, %c0_i64 : i64
          %104 = arith.extui %103 : i1 to i64
          %105 = arith.addi %arg6, %c1_i64 overflow<nsw> : i64
          %106 = arith.remui %105, %c2_i64 : i64
          %107 = arith.minsi %92, %6 : i64
          %108 = arith.subi %6, %107 : i64
          %109 = arith.minsi %108, %c1_i64 : i64
          %110 = arith.cmpi sle, %92, %6 : i64
          %111 = arith.select %110, %109, %c0_i64 : i64
          %112 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %113 = cannir.extract_stride %112 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %114 = cannir.extract_dim %113[0] : !cannir.stride<(-1, 1)>
          %115 = arith.muli %114, %c2_i64 : i64
          %116 = arith.maxsi %69, %c1_i64 : i64
          %117 = arith.divsi %11, %116 : i64
          %118 = arith.minsi %94, %117 : i64
          %119 = arith.muli %118, %69 : i64
          %120 = arith.subi %11, %119 : i64
          %121 = arith.minsi %69, %120 : i64
          %122 = arith.cmpi sle, %94, %117 : i64
          %123 = arith.select %122, %121, %c0_i64 : i64
          %124 = arith.muli %92, %114 : i64
          %125 = arith.muli %124, %c2_i64 : i64
          %126 = arith.muli %95, %c2_i64 : i64
          %127 = arith.addi %125, %126 : i64
          %128 = arith.muli %123, %c2_i64 : i64
          %129 = arith.cmpi eq, %128, %c0_i64 : i64
          %130 = arith.select %129, %c0_i64, %111 : i64
          %131 = arith.select %129, %c2_i64, %128 : i64
          %132 = ascscalar.bind_local_ptr %102 : memref<1xbf16, 1>
          ascvec.get_buf mte2, %104, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%132, %alloc, %c0_i32, %c0_i32, %c0_i64, %127, %130, %131, %115, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %104, 0 {cannir.buffer.generated_sync}
          %133 = ascscalar.bind_local_ptr %102 : memref<32768xbf16, 1>
          %134 = arith.cmpi sge, %98, %c256_i64 : i64
          %135 = arith.remsi %98, %c256_i64 : i64
          %136 = arith.cmpi eq, %135, %c0_i64 : i64
          %137 = arith.select %134, %136, %false : i1
          scf.if %137 {
            %210 = arith.muli %92, %11 : i64
            %211 = arith.addi %210, %95 : i64
            %212 = arith.remsi %arg3, %c256_i64 : i64
            %213 = arith.cmpi ne, %212, %c0_i64 : i64
            %214 = arith.cmpi slt, %212, %c0_i64 : i64
            %215 = arith.andi %213, %214 : i1
            %216 = arith.select %215, %c256_i64, %c0_i64 : i64
            %217 = arith.addi %212, %216 overflow<nsw> : i64
            %218 = arith.cmpi eq, %217, %c0_i64 : i64
            scf.if %218 {
              %219 = arith.cmpi sge, %98, %c1024_i64 : i64
              %220 = arith.remsi %98, %c1024_i64 : i64
              %221 = arith.cmpi eq, %220, %c0_i64 : i64
              %222 = arith.select %219, %221, %false : i1
              scf.if %222 {
                %223 = arith.divsi %98, %c256_i64 : i64
                %224 = arith.addi %98, %c1023_i64 : i64
                %225 = arith.remsi %224, %c1024_i64 : i64
                %226 = arith.cmpi ne, %225, %c0_i64 : i64
                %227 = arith.cmpi slt, %225, %c0_i64 : i64
                %228 = arith.andi %226, %227 : i1
                %229 = arith.divsi %224, %c1024_i64 : i64
                %230 = arith.extui %228 : i1 to i64
                %231 = arith.subi %229, %230 : i64
                %232 = arith.trunci %1 : i64 to i32
                %233 = arith.extui %232 : i32 to i64
                %234 = arith.subi %1, %233 : i64
                %235 = arith.remsi %234, %c4294967296_i64 : i64
                %236 = arith.cmpi ne, %235, %c0_i64 : i64
                %237 = arith.cmpi slt, %235, %c0_i64 : i64
                %238 = arith.andi %236, %237 : i1
                %239 = arith.divsi %234, %c4294967296_i64 : i64
                %240 = arith.extui %238 : i1 to i64
                %241 = arith.subi %239, %240 : i64
                %242 = arith.trunci %241 : i64 to i32
                %243 = arith.divui %211, %c32_i64 : i64
                %244 = arith.trunci %243 : i64 to i32
                %245 = arith.extui %244 : i32 to i64
                %246 = arith.subi %243, %245 : i64
                %247 = arith.remsi %246, %c4294967296_i64 : i64
                %248 = arith.cmpi ne, %247, %c0_i64 : i64
                %249 = arith.cmpi slt, %247, %c0_i64 : i64
                %250 = arith.andi %248, %249 : i1
                %251 = arith.divsi %246, %c4294967296_i64 : i64
                %252 = arith.extui %250 : i1 to i64
                %253 = arith.subi %251, %252 : i64
                %254 = arith.trunci %253 : i64 to i32
                %255 = arith.trunci %3 : i64 to i32
                %256 = arith.extui %255 : i32 to i64
                %257 = arith.subi %3, %256 : i64
                %258 = arith.remsi %257, %c4294967296_i64 : i64
                %259 = arith.cmpi ne, %258, %c0_i64 : i64
                %260 = arith.cmpi slt, %258, %c0_i64 : i64
                %261 = arith.andi %259, %260 : i1
                %262 = arith.divsi %257, %c4294967296_i64 : i64
                %263 = arith.extui %261 : i1 to i64
                %264 = arith.subi %262, %263 : i64
                %265 = arith.trunci %264 : i64 to i32
                %266 = arith.divui %255, %c4_i32 : i32
                %267 = arith.remui %265, %c4_i32 : i32
                %268 = arith.muli %267, %c1073741824_i32 : i32
                %269 = arith.addi %266, %268 : i32
                %270 = arith.divui %265, %c4_i32 : i32
                %271 = arith.muli %231, %c32_i64 : i64
                %272 = arith.divui %243, %c16384_i64 : i64
                %273 = arith.addi %243, %271 : i64
                %274 = arith.subi %273, %c1_i64 : i64
                %275 = arith.divui %274, %c16384_i64 : i64
                %276 = arith.cmpi eq, %272, %275 : i64
                scf.if %276 {
                  %286 = arith.trunci %272 : i64 to i32
                  %287 = arith.extui %286 : i32 to i64
                  %288 = arith.subi %272, %287 : i64
                  %289 = arith.remsi %288, %c4294967296_i64 : i64
                  %290 = arith.cmpi ne, %289, %c0_i64 : i64
                  %291 = arith.cmpi slt, %289, %c0_i64 : i64
                  %292 = arith.andi %290, %291 : i1
                  %293 = arith.divsi %288, %c4294967296_i64 : i64
                  %294 = arith.extui %292 : i1 to i64
                  %295 = arith.subi %293, %294 : i64
                  %296 = arith.trunci %295 : i64 to i32
                  %297 = arith.addi %269, %286 : i32
                  %298 = arith.addi %270, %296 : i32
                  %299 = arith.cmpi ult, %297, %269 : i32
                  %300 = arith.extui %299 : i1 to i32
                  %301 = arith.addi %298, %300 : i32
                  %302 = arith.extui %297 : i32 to i64
                  %303 = arith.muli %302, %c3449720151_i64 : i64
                  %304 = arith.trunci %303 : i64 to i32
                  %305 = arith.extui %304 : i32 to i64
                  %306 = arith.subi %303, %305 : i64
                  %307 = arith.remsi %306, %c4294967296_i64 : i64
                  %308 = arith.cmpi ne, %307, %c0_i64 : i64
                  %309 = arith.cmpi slt, %307, %c0_i64 : i64
                  %310 = arith.andi %308, %309 : i1
                  %311 = arith.divsi %306, %c4294967296_i64 : i64
                  %312 = arith.extui %310 : i1 to i64
                  %313 = arith.subi %311, %312 : i64
                  %314 = arith.trunci %313 : i64 to i32
                  %315 = arith.xori %301, %242 : i32
                  %316 = arith.extui %315 : i32 to i64
                  %317 = arith.muli %316, %c3449720151_i64 : i64
                  %318 = arith.trunci %317 : i64 to i32
                  %319 = arith.extui %318 : i32 to i64
                  %320 = arith.subi %317, %319 : i64
                  %321 = arith.remsi %320, %c4294967296_i64 : i64
                  %322 = arith.cmpi ne, %321, %c0_i64 : i64
                  %323 = arith.cmpi slt, %321, %c0_i64 : i64
                  %324 = arith.andi %322, %323 : i1
                  %325 = arith.divsi %320, %c4294967296_i64 : i64
                  %326 = arith.extui %324 : i1 to i64
                  %327 = arith.subi %325, %326 : i64
                  %328 = arith.trunci %327 : i64 to i32
                  %329 = arith.addi %232, %c-1640531527_i32 : i32
                  %330 = arith.addi %242, %c-1150833019_i32 : i32
                  %331 = arith.xori %328, %304 : i32
                  %332 = arith.xori %331, %329 : i32
                  %333 = arith.extui %332 : i32 to i64
                  %334 = arith.muli %333, %c3528531795_i64 : i64
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
                  %346 = arith.xori %314, %232 : i32
                  %347 = arith.addi %232, %c1013904242_i32 : i32
                  %348 = arith.xori %318, %347 : i32
                  %349 = arith.addi %242, %c1993301258_i32 : i32
                  %350 = arith.xori %345, %349 : i32
                  ascsimt.vf(%244, %346, %330, %348, %350, %232, %242, %335, %alloc_8, %271 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
                    %c256_i64_10 = arith.constant 256 : i64
                    %c3_i64_11 = arith.constant 3 : i64
                    %c2_i64_12 = arith.constant 2 : i64
                    %c1_i64_13 = arith.constant 1 : i64
                    %c4_i64_14 = arith.constant 4 : i64
                    %c1_i32_15 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c3_i32 = arith.constant 3 : i32
                    %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %351 = ascsimt.thread_idx {ascsimt.scope = 31 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %352 = arith.extsi %351 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg17 = %352 to %arg16 step %c256_i64_10  : i64 {
                      %353 = arith.trunci %arg17 : i64 to i32
                      %354 = arith.addi %arg7, %353 : i32
                      %355 = arith.remui %354, %c16384_i32 : i32
                      %356 = arith.xori %355, %arg8 : i32
                      %357 = arith.muli %356, %c-766435501_i32 : i32
                      %358 = "ascsimt.bitcast"(%356) : (i32) -> ui32
                      %359 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %360 = ascsimt.math_binary %358, %359, "mulhi" : ui32, ui32 -> ui32
                      %361 = "ascsimt.bitcast"(%360) : (ui32) -> i32
                      %362 = arith.xori %361, %arg9 : i32
                      %363 = arith.muli %362, %c-845247145_i32 : i32
                      %364 = "ascsimt.bitcast"(%362) : (i32) -> ui32
                      %365 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %366 = ascsimt.math_binary %364, %365, "mulhi" : ui32, ui32 -> ui32
                      %367 = "ascsimt.bitcast"(%366) : (ui32) -> i32
                      %368 = arith.xori %367, %arg10 : i32
                      %369 = arith.xori %357, %arg11 : i32
                      %370:4 = scf.for %arg18 = %c3_i32 to %c10_i32 step %c1_i32_15 iter_args(%arg19 = %368, %arg20 = %363, %arg21 = %369, %arg22 = %arg14) -> (i32, i32, i32, i32)  : i32 {
                        %379 = arith.muli %arg19, %c-766435501_i32 : i32
                        %380 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %381 = ascsimt.math_binary %380, %359, "mulhi" : ui32, ui32 -> ui32
                        %382 = "ascsimt.bitcast"(%381) : (ui32) -> i32
                        %383 = arith.muli %arg21, %c-845247145_i32 : i32
                        %384 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %385 = ascsimt.math_binary %384, %365, "mulhi" : ui32, ui32 -> ui32
                        %386 = "ascsimt.bitcast"(%385) : (ui32) -> i32
                        %387 = arith.muli %arg18, %c-1640531527_i32_17 : i32
                        %388 = arith.addi %arg12, %387 : i32
                        %389 = arith.muli %arg18, %c-1150833019_i32_16 : i32
                        %390 = arith.addi %arg13, %389 : i32
                        %391 = arith.xori %386, %arg20 : i32
                        %392 = arith.xori %391, %388 : i32
                        %393 = arith.xori %382, %arg22 : i32
                        %394 = arith.xori %393, %390 : i32
                        scf.yield %392, %383, %394, %379 : i32, i32, i32, i32
                      }
                      %371 = arith.muli %arg17, %c4_i64_14 : i64
                      %372 = arith.index_cast %371 : i64 to index
                      ascsimt.store %arg15[%372], %370#0 : memref<8192xi32, 1>, i32
                      %373 = arith.addi %371, %c1_i64_13 : i64
                      %374 = arith.index_cast %373 : i64 to index
                      ascsimt.store %arg15[%374], %370#1 : memref<8192xi32, 1>, i32
                      %375 = arith.addi %371, %c2_i64_12 : i64
                      %376 = arith.index_cast %375 : i64 to index
                      ascsimt.store %arg15[%376], %370#2 : memref<8192xi32, 1>, i32
                      %377 = arith.addi %371, %c3_i64_11 : i64
                      %378 = arith.index_cast %377 : i64 to index
                      ascsimt.store %arg15[%378], %370#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%244, %254, %269, %270, %232, %242, %alloc_8, %271 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: memref<8192xi32, 1>, %arg14: i64):
                    %c256_i64_10 = arith.constant 256 : i64
                    %c3_i64_11 = arith.constant 3 : i64
                    %c2_i64_12 = arith.constant 2 : i64
                    %c1_i64_13 = arith.constant 1 : i64
                    %c4_i64_14 = arith.constant 4 : i64
                    %c1_i32_15 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c0_i32_16 = arith.constant 0 : i32
                    %c-1150833019_i32_17 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_18 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c262144_i32 = arith.constant 262144 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %286 = ascsimt.thread_idx {ascsimt.scope = 32 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %287 = arith.extsi %286 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg15 = %287 to %arg14 step %c256_i64_10  : i64 {
                      %288 = arith.trunci %arg15 : i64 to i32
                      %289 = arith.addi %arg7, %288 : i32
                      %290 = arith.cmpi ult, %289, %arg7 : i32
                      %291 = arith.extui %290 : i1 to i32
                      %292 = arith.addi %arg8, %291 : i32
                      %293 = arith.divui %289, %c16384_i32 : i32
                      %294 = arith.remui %292, %c16384_i32 : i32
                      %295 = arith.muli %294, %c262144_i32 : i32
                      %296 = arith.addi %293, %295 : i32
                      %297 = arith.divui %292, %c16384_i32 : i32
                      %298 = arith.addi %arg9, %296 : i32
                      %299 = arith.addi %arg10, %297 : i32
                      %300 = arith.cmpi ult, %298, %arg9 : i32
                      %301 = arith.extui %300 : i1 to i32
                      %302 = arith.addi %299, %301 : i32
                      %303 = arith.remui %289, %c16384_i32 : i32
                      %304:4 = scf.for %arg16 = %c0_i32_16 to %c10_i32 step %c1_i32_15 iter_args(%arg17 = %c0_i32_16, %arg18 = %303, %arg19 = %298, %arg20 = %302) -> (i32, i32, i32, i32)  : i32 {
                        %313 = arith.muli %arg17, %c-766435501_i32 : i32
                        %314 = "ascsimt.bitcast"(%arg17) : (i32) -> ui32
                        %315 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %316 = ascsimt.math_binary %314, %315, "mulhi" : ui32, ui32 -> ui32
                        %317 = "ascsimt.bitcast"(%316) : (ui32) -> i32
                        %318 = arith.muli %arg19, %c-845247145_i32 : i32
                        %319 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %320 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %321 = ascsimt.math_binary %319, %320, "mulhi" : ui32, ui32 -> ui32
                        %322 = "ascsimt.bitcast"(%321) : (ui32) -> i32
                        %323 = arith.muli %arg16, %c-1640531527_i32_18 : i32
                        %324 = arith.addi %arg11, %323 : i32
                        %325 = arith.muli %arg16, %c-1150833019_i32_17 : i32
                        %326 = arith.addi %arg12, %325 : i32
                        %327 = arith.xori %322, %arg18 : i32
                        %328 = arith.xori %327, %324 : i32
                        %329 = arith.xori %317, %arg20 : i32
                        %330 = arith.xori %329, %326 : i32
                        scf.yield %328, %318, %330, %313 : i32, i32, i32, i32
                      }
                      %305 = arith.muli %arg15, %c4_i64_14 : i64
                      %306 = arith.index_cast %305 : i64 to index
                      ascsimt.store %arg13[%306], %304#0 : memref<8192xi32, 1>, i32
                      %307 = arith.addi %305, %c1_i64_13 : i64
                      %308 = arith.index_cast %307 : i64 to index
                      ascsimt.store %arg13[%308], %304#1 : memref<8192xi32, 1>, i32
                      %309 = arith.addi %305, %c2_i64_12 : i64
                      %310 = arith.index_cast %309 : i64 to index
                      ascsimt.store %arg13[%310], %304#2 : memref<8192xi32, 1>, i32
                      %311 = arith.addi %305, %c3_i64_11 : i64
                      %312 = arith.index_cast %311 : i64 to index
                      ascsimt.store %arg13[%312], %304#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %277 = arith.trunci %31 : i64 to i32
                %278 = arith.divsi %277, %c2_i32 : i32
                %279 = arith.trunci %278 : i32 to i8
                %280 = arith.trunci %223 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %281 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%279, %alloc_6, %133, %281, %alloc_9, %280 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, memref<2048xui16, 1>, i32) {
                ^bb0(%arg7: i8, %arg8: memref<2048xui8, 1>, %arg9: memref<32768xbf16, 1>, %arg10: memref<?xui16, 1>, %arg11: memref<2048xui16, 1>, %arg12: i32):
                  %c16_10 = arith.constant 16 : index
                  %c4_i32_11 = arith.constant 4 : i32
                  %c0_i32_12 = arith.constant 0 : i32
                  %c3_i32 = arith.constant 3 : i32
                  %c2_i32_13 = arith.constant 2 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c1_i32_14 = arith.constant 1 : i32
                  %c0_15 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %286 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %287 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %288 = ascvec.duplicate_scalar %c32767_i16, %286 : i16 -> <ui16>
                  %289 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %290 = ascvec.shr %289, 4, %286 : <ui16> -> <ui16>
                  %291 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %292 = ascvec.duplicate_scalar %c1_i8, %287 : i8 -> <ui8>
                  %293 = ascvec.and %291, %292, %287 : <ui8>, <ui8> -> <ui8>
                  %294 = ascvec.cmp_lt_scalar %291, %c16_i8, %287 : <ui8>, i8 -> !ascvec.mask
                  %295 = ascvec.cmp_lt_scalar %293, %arg7, %287 : <ui8>, i8 -> !ascvec.mask
                  %296 = ascvec.mask_and %294, %295, %287
                  ascvec.store_unalign_begin %arg8 : memref<2048xui8, 1>
                  %mask_16, %remaining_out_17 = ascvec.update_mask %c1_i32_14 elem_bits 16 : !ascvec.mask
                  %297 = ascvec.duplicate_scalar %c-1_i16, %mask_16 : i16 -> <ui16>
                  %298 = scf.for %arg13 = %c0_i32_12 to %arg12 step %c4_i32_11 iter_args(%arg14 = %297) -> (!ascvec.vreg<ui16>)  : i32 {
                    %299 = ascvec.duplicate_scalar %c-1_i16, %286 : i16 -> <ui16>
                    %300 = ascvec.duplicate_scalar %c0_i16, %286 : i16 -> <ui16>
                    %301 = arith.muli %arg13, %c256_i32 : i32
                    %302 = arith.index_cast %301 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg9[%302] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %303 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %304 = ascvec.and %303, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %305 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %306 = ascvec.and %305, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %307 = ascvec.max %304, %306, %286 : <ui16>
                    %308 = ascvec.reduce_max_datablock %307, %286 : <ui16>
                    %309 = ascvec.shr %308, 7, %286 : <ui16> -> <ui16>
                    %310 = ascvec.cmp_ge_scalar %309, %c255_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %311 = ascvec.max_scalar %309, %c15_i16, %286 : <ui16>, i16
                    %312 = ascvec.add_scalar %311, %c-15_i16, %286 : <ui16>, i16
                    %313 = ascvec.duplicate_scalar %c255_i16, %310 merge %312 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %314 = ascvec.duplicate_scalar %c254_i16, %286 : i16 -> <ui16>
                    %315 = ascvec.sub %314, %313, %286 : <ui16>
                    %316 = ascvec.cmp_ne_scalar %313, %c0_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %317 = ascvec.shl %315, 7, %316 : <ui16> -> <ui16>
                    %318 = ascvec.max %300, %313, %mask : <ui16>
                    %319 = ascvec.gather_reg %317, %290 : <ui16>, <ui16> -> <ui16>
                    %320 = ascvec.reinterpret %319 : <ui16> -> <bf16>
                    %321 = ascvec.mul %result0, %320, %286 : <bf16>
                    %322 = ascvec.mul %result1, %320, %286 : <bf16>
                    ascvec.storealign_intlv %arg9[%302], %321, %322 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %323 = ascvec.reinterpret %321 : <bf16> -> <ui16>
                    %324 = ascvec.and %323, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %325 = ascvec.reinterpret %322 : <bf16> -> <ui16>
                    %326 = ascvec.and %325, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %327 = ascvec.min %324, %326, %286 : <ui16>
                    %328 = ascvec.min %299, %327, %286 : <ui16>
                    %329 = arith.muli %arg13, %c16_i32 : i32
                    %330 = arith.index_cast %329 : i32 to index
                    ascvec.storealign %arg10[%330], %313, %mask : memref<?xui16, 1>, index, <ui16>
                    %331 = ascvec.reinterpret %313 : <ui16> -> <ui8>
                    %332 = ascvec.squeeze_store_reg %331, %296 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_15], %332 : memref<2048xui8, 1>, <ui8>
                    %333 = arith.addi %arg13, %c1_i32_14 : i32
                    %334 = arith.muli %333, %c256_i32 : i32
                    %335 = arith.index_cast %334 : i32 to index
                    %result0_20, %result1_21 = ascvec.loadalign_deintlv %arg9[%335] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %336 = ascvec.reinterpret %result0_20 : <bf16> -> <ui16>
                    %337 = ascvec.and %336, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %338 = ascvec.reinterpret %result1_21 : <bf16> -> <ui16>
                    %339 = ascvec.and %338, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %340 = ascvec.max %337, %339, %286 : <ui16>
                    %341 = ascvec.reduce_max_datablock %340, %286 : <ui16>
                    %342 = ascvec.shr %341, 7, %286 : <ui16> -> <ui16>
                    %343 = ascvec.cmp_ge_scalar %342, %c255_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %344 = ascvec.max_scalar %342, %c15_i16, %286 : <ui16>, i16
                    %345 = ascvec.add_scalar %344, %c-15_i16, %286 : <ui16>, i16
                    %346 = ascvec.duplicate_scalar %c255_i16, %343 merge %345 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %347 = ascvec.sub %314, %346, %286 : <ui16>
                    %348 = ascvec.cmp_ne_scalar %346, %c0_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %349 = ascvec.shl %347, 7, %348 : <ui16> -> <ui16>
                    %350 = ascvec.max %318, %346, %mask : <ui16>
                    %351 = ascvec.gather_reg %349, %290 : <ui16>, <ui16> -> <ui16>
                    %352 = ascvec.reinterpret %351 : <ui16> -> <bf16>
                    %353 = ascvec.mul %result0_20, %352, %286 : <bf16>
                    %354 = ascvec.mul %result1_21, %352, %286 : <bf16>
                    ascvec.storealign_intlv %arg9[%335], %353, %354 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %355 = ascvec.reinterpret %353 : <bf16> -> <ui16>
                    %356 = ascvec.and %355, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %357 = ascvec.reinterpret %354 : <bf16> -> <ui16>
                    %358 = ascvec.and %357, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %359 = ascvec.min %356, %358, %286 : <ui16>
                    %360 = ascvec.min %328, %359, %286 : <ui16>
                    %361 = arith.muli %333, %c16_i32 : i32
                    %362 = arith.index_cast %361 : i32 to index
                    ascvec.storealign %arg10[%362], %346, %mask : memref<?xui16, 1>, index, <ui16>
                    %363 = ascvec.reinterpret %346 : <ui16> -> <ui8>
                    %364 = ascvec.squeeze_store_reg %363, %296 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_15], %364 : memref<2048xui8, 1>, <ui8>
                    %365 = arith.addi %arg13, %c2_i32_13 : i32
                    %366 = arith.muli %365, %c256_i32 : i32
                    %367 = arith.index_cast %366 : i32 to index
                    %result0_22, %result1_23 = ascvec.loadalign_deintlv %arg9[%367] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %368 = ascvec.reinterpret %result0_22 : <bf16> -> <ui16>
                    %369 = ascvec.and %368, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %370 = ascvec.reinterpret %result1_23 : <bf16> -> <ui16>
                    %371 = ascvec.and %370, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %372 = ascvec.max %369, %371, %286 : <ui16>
                    %373 = ascvec.reduce_max_datablock %372, %286 : <ui16>
                    %374 = ascvec.shr %373, 7, %286 : <ui16> -> <ui16>
                    %375 = ascvec.cmp_ge_scalar %374, %c255_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %376 = ascvec.max_scalar %374, %c15_i16, %286 : <ui16>, i16
                    %377 = ascvec.add_scalar %376, %c-15_i16, %286 : <ui16>, i16
                    %378 = ascvec.duplicate_scalar %c255_i16, %375 merge %377 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %379 = ascvec.sub %314, %378, %286 : <ui16>
                    %380 = ascvec.cmp_ne_scalar %378, %c0_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %381 = ascvec.shl %379, 7, %380 : <ui16> -> <ui16>
                    %382 = ascvec.max %350, %378, %mask : <ui16>
                    %383 = ascvec.gather_reg %381, %290 : <ui16>, <ui16> -> <ui16>
                    %384 = ascvec.reinterpret %383 : <ui16> -> <bf16>
                    %385 = ascvec.mul %result0_22, %384, %286 : <bf16>
                    %386 = ascvec.mul %result1_23, %384, %286 : <bf16>
                    ascvec.storealign_intlv %arg9[%367], %385, %386 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %387 = ascvec.reinterpret %385 : <bf16> -> <ui16>
                    %388 = ascvec.and %387, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %389 = ascvec.reinterpret %386 : <bf16> -> <ui16>
                    %390 = ascvec.and %389, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %391 = ascvec.min %388, %390, %286 : <ui16>
                    %392 = ascvec.min %360, %391, %286 : <ui16>
                    %393 = arith.muli %365, %c16_i32 : i32
                    %394 = arith.index_cast %393 : i32 to index
                    ascvec.storealign %arg10[%394], %378, %mask : memref<?xui16, 1>, index, <ui16>
                    %395 = ascvec.reinterpret %378 : <ui16> -> <ui8>
                    %396 = ascvec.squeeze_store_reg %395, %296 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_15], %396 : memref<2048xui8, 1>, <ui8>
                    %397 = arith.addi %arg13, %c3_i32 : i32
                    %398 = arith.muli %397, %c256_i32 : i32
                    %399 = arith.index_cast %398 : i32 to index
                    %result0_24, %result1_25 = ascvec.loadalign_deintlv %arg9[%399] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %400 = ascvec.reinterpret %result0_24 : <bf16> -> <ui16>
                    %401 = ascvec.and %400, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %402 = ascvec.reinterpret %result1_25 : <bf16> -> <ui16>
                    %403 = ascvec.and %402, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %404 = ascvec.max %401, %403, %286 : <ui16>
                    %405 = ascvec.reduce_max_datablock %404, %286 : <ui16>
                    %406 = ascvec.shr %405, 7, %286 : <ui16> -> <ui16>
                    %407 = ascvec.cmp_ge_scalar %406, %c255_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %408 = ascvec.max_scalar %406, %c15_i16, %286 : <ui16>, i16
                    %409 = ascvec.add_scalar %408, %c-15_i16, %286 : <ui16>, i16
                    %410 = ascvec.duplicate_scalar %c255_i16, %407 merge %409 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %411 = ascvec.sub %314, %410, %286 : <ui16>
                    %412 = ascvec.cmp_ne_scalar %410, %c0_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %413 = ascvec.shl %411, 7, %412 : <ui16> -> <ui16>
                    %414 = ascvec.max %382, %410, %mask : <ui16>
                    %415 = ascvec.gather_reg %413, %290 : <ui16>, <ui16> -> <ui16>
                    %416 = ascvec.reinterpret %415 : <ui16> -> <bf16>
                    %417 = ascvec.mul %result0_24, %416, %286 : <bf16>
                    %418 = ascvec.mul %result1_25, %416, %286 : <bf16>
                    ascvec.storealign_intlv %arg9[%399], %417, %418 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %419 = ascvec.reinterpret %417 : <bf16> -> <ui16>
                    %420 = ascvec.and %419, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %421 = ascvec.reinterpret %418 : <bf16> -> <ui16>
                    %422 = ascvec.and %421, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %423 = ascvec.min %420, %422, %286 : <ui16>
                    %424 = ascvec.min %392, %423, %286 : <ui16>
                    %425 = arith.muli %397, %c16_i32 : i32
                    %426 = arith.index_cast %425 : i32 to index
                    ascvec.storealign %arg10[%426], %410, %mask : memref<?xui16, 1>, index, <ui16>
                    %427 = ascvec.reinterpret %410 : <ui16> -> <ui8>
                    %428 = ascvec.squeeze_store_reg %427, %296 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_15], %428 : memref<2048xui8, 1>, <ui8>
                    %429 = ascvec.reduce_min %424, %286 : <ui16>
                    %430 = ascvec.reduce_max %414, %mask : <ui16>
                    %431 = ascvec.cmp_eq_scalar %430, %c255_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %432 = ascvec.duplicate_scalar %c0_i16, %431 merge %429 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.storealign_1st %arg11[%330], %432 : memref<2048xui16, 1>, index, <ui16>
                    %433 = ascvec.min %arg14, %432, %mask_16 : <ui16>
                    scf.yield %433 : !ascvec.vreg<ui16>
                  }
                  %c0_18 = arith.constant 0 : index
                  %c1_19 = arith.constant 1 : index
                  scf.for %arg13 = %c0_18 to %c1_19 step %c1_19 {
                    ascvec.store_unalign_post %arg8[%c0_15] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg11[%c16_10], %298 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %282 = memref.load %alloc_9[%c16] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %283 = cannir.scalar_storage_cast %282 : ui16 -> i16
                %284 = arith.extui %283 : i16 to i64
                %285 = arith.cmpi sge, %284, %c14464_i64 : i64
                scf.if %285 {
                  ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %286 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                  %287 = ascscalar.bitcast_memref %133 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%133, %286, %287, %alloc_5, %280 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg7: memref<32768xbf16, 1>, %arg8: memref<?xui8, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<32768xui8, 1>, %arg11: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_10 = arith.constant 1 : i32
                    %c0_i32_11 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c128_i32 = arith.constant 128 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c31_i16 = arith.constant 31 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %288 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %289 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %290 = ascvec.duplicate_scalar %c255_i16, %288 : i16 -> <ui16>
                    %291 = ascvec.duplicate_scalar %c-128_i8, %289 : i8 -> <ui8>
                    %292 = ascvec.duplicate_scalar %c31_i16, %288 : i16 -> <ui16>
                    scf.for %arg12 = %c0_i32_11 to %arg11 step %c1_i32_10  : i32 {
                      %293 = arith.muli %arg12, %c256_i32 : i32
                      %294 = arith.index_cast %293 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg7[%294] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %295 = arith.muli %arg12, %c128_i32 : i32
                      %296 = arith.index_cast %295 : i32 to index
                      %297 = ascvec.loadalign %arg8[%296] : memref<?xui8, 1>, index -> <ui8>
                      %298 = ascvec.unpack %297 {part = "lower"} : <ui8> -> <ui16>
                      %299 = ascvec.and %298, %292, %288 : <ui16>, <ui16> -> <ui16>
                      %300 = ascvec.shr %298, 3, %288 : <ui16> -> <ui16>
                      %301 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %302 = ascvec.add %301, %299, %288 : <ui16>
                      %303 = ascvec.shr %302, 5, %288 : <ui16> -> <ui16>
                      %304 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %305 = ascvec.add %304, %300, %288 : <ui16>
                      %306 = ascvec.shr %305, 5, %288 : <ui16> -> <ui16>
                      %307 = ascvec.and %303, %290, %288 : <ui16>, <ui16> -> <ui16>
                      %308 = ascvec.shl %306, 8, %288 : <ui16> -> <ui16>
                      %309 = ascvec.or %307, %308, %288 : <ui16>, <ui16> -> <ui16>
                      %310 = ascvec.reinterpret %309 : <ui16> -> <ui8>
                      %311 = ascvec.add_scalar %310, %c64_i8, %289 : <ui8>, i8
                      %312 = ascvec.min_scalar %311, %c123_i8, %289 : <ui8>, i8
                      %313 = arith.muli %arg12, %c512_i32 : i32
                      %314 = arith.index_cast %313 : i32 to index
                      %result0_12, %result1_13 = ascvec.loadalign_deintlv %arg9[%314] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %315 = ascvec.and %result1_13, %291, %289 : <ui8>, <ui8> -> <ui8>
                      %316 = ascvec.or %312, %315, %289 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg10[%294], %316, %289 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
                } else {
                  scf.for %arg7 = %c0_i64 to %223 step %c16_i64  : i64 {
                    %286 = arith.addi %arg7, %c16_i64 : i64
                    %287 = arith.cmpi sgt, %286, %223 : i64
                    %288 = arith.select %287, %223, %286 : i64
                    %289 = arith.muli %arg7, %c16_i64 : i64
                    ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %290 = arith.index_cast %289 : i64 to index
                    %291 = memref.load %alloc_9[%290] : memref<2048xui16, 1>
                    ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %292 = cannir.scalar_storage_cast %291 : ui16 -> i16
                    %293 = arith.extui %292 : i16 to i64
                    %294 = arith.addi %arg7, %c4_i64 : i64
                    %295 = scf.for %arg8 = %294 to %288 step %c4_i64 iter_args(%arg9 = %293) -> (i64)  : i64 {
                      %297 = arith.muli %arg8, %c16_i64 : i64
                      ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %298 = arith.index_cast %297 : i64 to index
                      %299 = memref.load %alloc_9[%298] : memref<2048xui16, 1>
                      ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %300 = cannir.scalar_storage_cast %299 : ui16 -> i16
                      %301 = arith.extui %300 : i16 to i64
                      %302 = arith.cmpi slt, %301, %arg9 : i64
                      %303 = arith.select %302, %301, %arg9 : i64
                      scf.yield %303 : i64
                    }
                    %296 = arith.cmpi sge, %295, %c14464_i64 : i64
                    scf.if %296 {
                      %297 = arith.trunci %arg7 : i64 to i32
                      %298 = arith.trunci %288 : i64 to i32
                      ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %299 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                      %300 = ascscalar.bitcast_memref %133 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                      ascvec.vf(%133, %299, %300, %alloc_5, %297, %298 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                      ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32, %arg13: i32):
                        %c512_i32 = arith.constant 512 : i32
                        %c1_i32_10 = arith.constant 1 : i32
                        %c123_i8 = arith.constant 123 : i8
                        %c64_i8 = arith.constant 64 : i8
                        %c128_i32 = arith.constant 128 : i32
                        %c256_i32 = arith.constant 256 : i32
                        %c31_i16 = arith.constant 31 : i16
                        %c-128_i8 = arith.constant -128 : i8
                        %c255_i16 = arith.constant 255 : i16
                        %301 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                        %302 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                        %303 = ascvec.duplicate_scalar %c255_i16, %301 : i16 -> <ui16>
                        %304 = ascvec.duplicate_scalar %c-128_i8, %302 : i8 -> <ui8>
                        %305 = ascvec.duplicate_scalar %c31_i16, %301 : i16 -> <ui16>
                        scf.for %arg14 = %arg12 to %arg13 step %c1_i32_10  : i32 {
                          %306 = arith.muli %arg14, %c256_i32 : i32
                          %307 = arith.index_cast %306 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg8[%307] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                          %308 = arith.muli %arg14, %c128_i32 : i32
                          %309 = arith.index_cast %308 : i32 to index
                          %310 = ascvec.loadalign %arg9[%309] : memref<?xui8, 1>, index -> <ui8>
                          %311 = ascvec.unpack %310 {part = "lower"} : <ui8> -> <ui16>
                          %312 = ascvec.and %311, %305, %301 : <ui16>, <ui16> -> <ui16>
                          %313 = ascvec.shr %311, 3, %301 : <ui16> -> <ui16>
                          %314 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                          %315 = ascvec.add %314, %312, %301 : <ui16>
                          %316 = ascvec.shr %315, 5, %301 : <ui16> -> <ui16>
                          %317 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                          %318 = ascvec.add %317, %313, %301 : <ui16>
                          %319 = ascvec.shr %318, 5, %301 : <ui16> -> <ui16>
                          %320 = ascvec.and %316, %303, %301 : <ui16>, <ui16> -> <ui16>
                          %321 = ascvec.shl %319, 8, %301 : <ui16> -> <ui16>
                          %322 = ascvec.or %320, %321, %301 : <ui16>, <ui16> -> <ui16>
                          %323 = ascvec.reinterpret %322 : <ui16> -> <ui8>
                          %324 = ascvec.add_scalar %323, %c64_i8, %302 : <ui8>, i8
                          %325 = ascvec.min_scalar %324, %c123_i8, %302 : <ui8>, i8
                          %326 = arith.muli %arg14, %c512_i32 : i32
                          %327 = arith.index_cast %326 : i32 to index
                          %result0_11, %result1_12 = ascvec.loadalign_deintlv %arg10[%327] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                          %328 = ascvec.and %result1_12, %304, %302 : <ui8>, <ui8> -> <ui8>
                          %329 = ascvec.or %325, %328, %302 : <ui8>, <ui8> -> <ui8>
                          ascvec.storealign %arg11[%307], %329, %302 : memref<32768xui8, 1>, index, <ui8>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
                    } else {
                      %297 = arith.cmpi sge, %295, %c13568_i64 : i64
                      scf.if %297 {
                        %298 = arith.trunci %arg7 : i64 to i32
                        %299 = arith.trunci %288 : i64 to i32
                        ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %300 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                        %301 = ascscalar.bitcast_memref %133 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%133, %300, %301, %alloc_5, %298, %299 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32, %arg13: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_10 = arith.constant 1 : i32
                          %c123_i8 = arith.constant 123 : i8
                          %c64_i8 = arith.constant 64 : i8
                          %c448_i16 = arith.constant 448 : i16
                          %c14464_i16 = arith.constant 14464 : i16
                          %c128_i32 = arith.constant 128 : i32
                          %c256_i32 = arith.constant 256 : i32
                          %cst = arith.constant 2.684350e+08 : bf16
                          %c32767_i16 = arith.constant 32767 : i16
                          %c4095_i16 = arith.constant 4095 : i16
                          %c3_i8 = arith.constant 3 : i8
                          %c1_i8 = arith.constant 1 : i8
                          %c0_i8 = arith.constant 0 : i8
                          %c-128_i8 = arith.constant -128 : i8
                          %c255_i16 = arith.constant 255 : i16
                          %302 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %303 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %304 = ascvec.duplicate_scalar %c255_i16, %302 : i16 -> <ui16>
                          %305 = ascvec.duplicate_scalar %c-128_i8, %303 : i8 -> <ui8>
                          %306 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %307 = ascvec.shr %306, 1, %303 : <ui8> -> <ui8>
                          %308 = ascvec.shr %307, 2, %303 : <ui8> -> <ui8>
                          %309 = ascvec.shl %308, 2, %303 : <ui8> -> <ui8>
                          %310 = ascvec.duplicate_scalar %c1_i8, %303 : i8 -> <ui8>
                          %311 = ascvec.and %306, %310, %303 : <ui8>, <ui8> -> <ui8>
                          %312 = ascvec.add_scalar %307, %c3_i8, %303 : <ui8>, i8
                          %313 = ascvec.add %312, %311, %303 : <ui8>
                          %314 = ascvec.duplicate_scalar %c3_i8, %303 : i8 -> <ui8>
                          %315 = ascvec.and %313, %314, %303 : <ui8>, <ui8> -> <ui8>
                          %316 = ascvec.or %309, %315, %303 : <ui8>, <ui8> -> <ui8>
                          %317 = ascvec.duplicate_scalar %c4095_i16, %302 : i16 -> <ui16>
                          %318 = ascvec.duplicate_scalar %c32767_i16, %302 : i16 -> <ui16>
                          %319 = ascvec.duplicate_scalar %cst, %302 : bf16 -> <bf16>
                          scf.for %arg14 = %arg12 to %arg13 step %c1_i32_10  : i32 {
                            %320 = arith.muli %arg14, %c256_i32 : i32
                            %321 = arith.index_cast %320 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg8[%321] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %322 = arith.muli %arg14, %c128_i32 : i32
                            %323 = arith.index_cast %322 : i32 to index
                            %324 = ascvec.loadalign %arg9[%323] : memref<?xui8, 1>, index -> <ui8>
                            %325 = ascvec.gather_reg %324, %316 : <ui8>, <ui8> -> <ui8>
                            %326 = ascvec.reinterpret %325 : <ui8> -> <ui16>
                            %327 = ascvec.shr %326, 1, %302 : <ui16> -> <ui16>
                            %328 = ascvec.and %327, %317, %302 : <ui16>, <ui16> -> <ui16>
                            %329 = ascvec.shr %326, 4, %302 : <ui16> -> <ui16>
                            %330 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %331 = ascvec.and %330, %318, %302 : <ui16>, <ui16> -> <ui16>
                            %332 = ascvec.cmp_lt_scalar %331, %c14464_i16, %302 : <ui16>, i16 -> !ascvec.mask
                            %333 = ascvec.shr %328, 7, %302 : <ui16> -> <ui16>
                            %334 = ascvec.add %331, %333, %302 : <ui16>
                            %335 = ascvec.shr %334, 5, %302 : <ui16> -> <ui16>
                            %336 = ascvec.reinterpret %331 : <ui16> -> <bf16>
                            %337 = ascvec.mul %336, %319, %332 : <bf16>
                            %338 = ascvec.cast %337, %302 {rounding = 4 : i32} : <bf16> -> <f16>
                            %339 = ascvec.cast %338, %302 {rounding = 4 : i32} : <f16> -> <i16>
                            %340 = ascvec.reinterpret %339 : <i16> -> <ui16>
                            %341 = ascvec.add %340, %328, %302 : <ui16>
                            %342 = ascvec.shr %341, 12, %302 : <ui16> -> <ui16>
                            %343 = ascvec.add_scalar %342, %c448_i16, %302 : <ui16>, i16
                            %344 = ascvec.select %343, %335, %332 : <ui16>
                            %345 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %346 = ascvec.and %345, %318, %302 : <ui16>, <ui16> -> <ui16>
                            %347 = ascvec.cmp_lt_scalar %346, %c14464_i16, %302 : <ui16>, i16 -> !ascvec.mask
                            %348 = ascvec.shr %329, 7, %302 : <ui16> -> <ui16>
                            %349 = ascvec.add %346, %348, %302 : <ui16>
                            %350 = ascvec.shr %349, 5, %302 : <ui16> -> <ui16>
                            %351 = ascvec.reinterpret %346 : <ui16> -> <bf16>
                            %352 = ascvec.mul %351, %319, %347 : <bf16>
                            %353 = ascvec.cast %352, %302 {rounding = 4 : i32} : <bf16> -> <f16>
                            %354 = ascvec.cast %353, %302 {rounding = 4 : i32} : <f16> -> <i16>
                            %355 = ascvec.reinterpret %354 : <i16> -> <ui16>
                            %356 = ascvec.add %355, %329, %302 : <ui16>
                            %357 = ascvec.shr %356, 12, %302 : <ui16> -> <ui16>
                            %358 = ascvec.add_scalar %357, %c448_i16, %302 : <ui16>, i16
                            %359 = ascvec.select %358, %350, %347 : <ui16>
                            %360 = ascvec.and %344, %304, %302 : <ui16>, <ui16> -> <ui16>
                            %361 = ascvec.shl %359, 8, %302 : <ui16> -> <ui16>
                            %362 = ascvec.or %360, %361, %302 : <ui16>, <ui16> -> <ui16>
                            %363 = ascvec.reinterpret %362 : <ui16> -> <ui8>
                            %364 = ascvec.add_scalar %363, %c64_i8, %303 : <ui8>, i8
                            %365 = ascvec.min_scalar %364, %c123_i8, %303 : <ui8>, i8
                            %366 = arith.muli %arg14, %c512_i32 : i32
                            %367 = arith.index_cast %366 : i32 to index
                            %result0_11, %result1_12 = ascvec.loadalign_deintlv %arg10[%367] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %368 = ascvec.and %result1_12, %305, %303 : <ui8>, <ui8> -> <ui8>
                            %369 = ascvec.or %365, %368, %303 : <ui8>, <ui8> -> <ui8>
                            ascvec.storealign %arg11[%321], %369, %303 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
                      } else {
                        %298 = arith.trunci %arg7 : i64 to i32
                        %299 = arith.trunci %288 : i64 to i32
                        ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %300 = ascscalar.bitcast_memref %133 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%133, %alloc_8, %300, %281, %alloc_5, %298, %299 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<8192xi32, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<?xui16, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32, %arg14: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c127_i8 = arith.constant 127 : i8
                          %c16_i32 = arith.constant 16 : i32
                          %c0_i16 = arith.constant 0 : i16
                          %c123_i8 = arith.constant 123 : i8
                          %c-1_i8 = arith.constant -1 : i8
                          %c60_i8 = arith.constant 60 : i8
                          %cst = arith.constant 6.10351526E-5 : f32
                          %c3_i32 = arith.constant 3 : i32
                          %c65535_i32 = arith.constant 65535 : i32
                          %c4_i32_10 = arith.constant 4 : i32
                          %c2_i32_11 = arith.constant 2 : i32
                          %c1_i32_12 = arith.constant 1 : i32
                          %c7_i32 = arith.constant 7 : i32
                          %c32_i32 = arith.constant 32 : i32
                          %c0_i32_13 = arith.constant 0 : i32
                          %c32767_i16 = arith.constant 32767 : i16
                          %c0_14 = arith.constant 0 : index
                          %c256_i32 = arith.constant 256 : i32
                          %c3_i8 = arith.constant 3 : i8
                          %c-128_i8 = arith.constant -128 : i8
                          %c0_i8 = arith.constant 0 : i8
                          %c8_i32 = arith.constant 8 : i32
                          %301 = ascvec.create_mask : !ascvec.mask
                          %302 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %303 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                          %304 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %305 = ascvec.duplicate_scalar %c-128_i8, %303 : i8 -> <ui8>
                          %306 = ascvec.shr %304, 5, %303 : <ui8> -> <ui8>
                          %307 = ascvec.duplicate_scalar %c3_i8, %303 : i8 -> <ui8>
                          %308 = ascvec.and %304, %307, %303 : <ui8>, <ui8> -> <ui8>
                          %309 = ascvec.shl %308, 6, %303 : <ui8> -> <ui8>
                          %310 = ascvec.shr %304, 2, %303 : <ui8> -> <ui8>
                          %311 = ascvec.or %309, %310, %303 : <ui8>, <ui8> -> <ui8>
                          scf.for %arg15 = %arg13 to %arg14 step %c1_i32_12  : i32 {
                            %312 = arith.muli %arg15, %c256_i32 : i32
                            %313 = arith.index_cast %312 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg8[%313] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %314 = ascvec.duplicate_scalar %c32767_i16, %302 : i16 -> <ui16>
                            %315 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %316 = ascvec.and %315, %314, %302 : <ui16>, <ui16> -> <ui16>
                            %317 = ascvec.reinterpret %316 : <ui16> -> <bf16>
                            %318 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %319 = ascvec.and %318, %314, %302 : <ui16>, <ui16> -> <ui16>
                            %320 = ascvec.reinterpret %319 : <ui16> -> <bf16>
                            %321 = ascvec.cast %317, %302 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %322 = ascvec.cast %317, %302 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %323 = ascvec.cast %320, %302 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %324 = ascvec.cast %320, %302 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %325 = ascvec.arange %c0_i32_13 : i32 -> <ui32>
                            %326 = ascvec.shl %325, 2, %301 : <ui32> -> <ui32>
                            %327 = ascvec.add_scalar %326, %c0_i32_13, %301 : <ui32>, i32
                            %328 = ascvec.shr %327, 3, %301 : <ui32> -> <ui32>
                            %329 = arith.muli %arg15, %c32_i32 : i32
                            %330 = ascvec.add_scalar %328, %329, %301 : <ui32>, i32
                            %331 = ascvec.gather %arg9[%c0_14], %330, %301 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %332 = ascvec.reinterpret %331 : <i32> -> <ui32>
                            %333 = ascvec.duplicate_scalar %c7_i32, %301 : i32 -> <ui32>
                            %334 = ascvec.and %327, %333, %301 : <ui32>, <ui32> -> <ui32>
                            %335 = ascvec.shl %332, 11, %301 : <ui32> -> <ui32>
                            %336 = ascvec.shr %332, 21, %301 : <ui32> -> <ui32>
                            %337 = ascvec.or %335, %336, %301 : <ui32>, <ui32> -> <ui32>
                            %338 = ascvec.shr %337, 3, %301 : <ui32> -> <ui32>
                            %339 = ascvec.shl %337, 29, %301 : <ui32> -> <ui32>
                            %340 = ascvec.or %338, %339, %301 : <ui32>, <ui32> -> <ui32>
                            %341 = ascvec.duplicate_scalar %c1_i32_12, %301 : i32 -> <ui32>
                            %342 = ascvec.and %334, %341, %301 : <ui32>, <ui32> -> <ui32>
                            %343 = ascvec.cmp_ne_scalar %342, %c0_i32_13, %301 : <ui32>, i32 -> !ascvec.mask
                            %344 = ascvec.select %340, %337, %343 : <ui32>
                            %345 = ascvec.shr %344, 8, %301 : <ui32> -> <ui32>
                            %346 = ascvec.shl %344, 24, %301 : <ui32> -> <ui32>
                            %347 = ascvec.or %345, %346, %301 : <ui32>, <ui32> -> <ui32>
                            %348 = ascvec.duplicate_scalar %c2_i32_11, %301 : i32 -> <ui32>
                            %349 = ascvec.and %334, %348, %301 : <ui32>, <ui32> -> <ui32>
                            %350 = ascvec.cmp_ne_scalar %349, %c0_i32_13, %301 : <ui32>, i32 -> !ascvec.mask
                            %351 = ascvec.select %347, %344, %350 : <ui32>
                            %352 = ascvec.shr %351, 16, %301 : <ui32> -> <ui32>
                            %353 = ascvec.shl %351, 16, %301 : <ui32> -> <ui32>
                            %354 = ascvec.or %352, %353, %301 : <ui32>, <ui32> -> <ui32>
                            %355 = ascvec.duplicate_scalar %c4_i32_10, %301 : i32 -> <ui32>
                            %356 = ascvec.and %334, %355, %301 : <ui32>, <ui32> -> <ui32>
                            %357 = ascvec.cmp_ne_scalar %356, %c0_i32_13, %301 : <ui32>, i32 -> !ascvec.mask
                            %358 = ascvec.select %354, %351, %357 : <ui32>
                            %359 = ascvec.duplicate_scalar %c65535_i32, %301 : i32 -> <ui32>
                            %360 = ascvec.and %358, %359, %301 : <ui32>, <ui32> -> <ui32>
                            %361 = ascvec.add_scalar %326, %c1_i32_12, %301 : <ui32>, i32
                            %362 = ascvec.shr %361, 3, %301 : <ui32> -> <ui32>
                            %363 = ascvec.add_scalar %362, %329, %301 : <ui32>, i32
                            %364 = ascvec.gather %arg9[%c0_14], %363, %301 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %365 = ascvec.reinterpret %364 : <i32> -> <ui32>
                            %366 = ascvec.and %361, %333, %301 : <ui32>, <ui32> -> <ui32>
                            %367 = ascvec.shl %365, 11, %301 : <ui32> -> <ui32>
                            %368 = ascvec.shr %365, 21, %301 : <ui32> -> <ui32>
                            %369 = ascvec.or %367, %368, %301 : <ui32>, <ui32> -> <ui32>
                            %370 = ascvec.shr %369, 3, %301 : <ui32> -> <ui32>
                            %371 = ascvec.shl %369, 29, %301 : <ui32> -> <ui32>
                            %372 = ascvec.or %370, %371, %301 : <ui32>, <ui32> -> <ui32>
                            %373 = ascvec.and %366, %341, %301 : <ui32>, <ui32> -> <ui32>
                            %374 = ascvec.cmp_ne_scalar %373, %c0_i32_13, %301 : <ui32>, i32 -> !ascvec.mask
                            %375 = ascvec.select %372, %369, %374 : <ui32>
                            %376 = ascvec.shr %375, 8, %301 : <ui32> -> <ui32>
                            %377 = ascvec.shl %375, 24, %301 : <ui32> -> <ui32>
                            %378 = ascvec.or %376, %377, %301 : <ui32>, <ui32> -> <ui32>
                            %379 = ascvec.and %366, %348, %301 : <ui32>, <ui32> -> <ui32>
                            %380 = ascvec.cmp_ne_scalar %379, %c0_i32_13, %301 : <ui32>, i32 -> !ascvec.mask
                            %381 = ascvec.select %378, %375, %380 : <ui32>
                            %382 = ascvec.shr %381, 16, %301 : <ui32> -> <ui32>
                            %383 = ascvec.shl %381, 16, %301 : <ui32> -> <ui32>
                            %384 = ascvec.or %382, %383, %301 : <ui32>, <ui32> -> <ui32>
                            %385 = ascvec.and %366, %355, %301 : <ui32>, <ui32> -> <ui32>
                            %386 = ascvec.cmp_ne_scalar %385, %c0_i32_13, %301 : <ui32>, i32 -> !ascvec.mask
                            %387 = ascvec.select %384, %381, %386 : <ui32>
                            %388 = ascvec.and %387, %359, %301 : <ui32>, <ui32> -> <ui32>
                            %389 = ascvec.add_scalar %326, %c2_i32_11, %301 : <ui32>, i32
                            %390 = ascvec.shr %389, 3, %301 : <ui32> -> <ui32>
                            %391 = ascvec.add_scalar %390, %329, %301 : <ui32>, i32
                            %392 = ascvec.gather %arg9[%c0_14], %391, %301 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %393 = ascvec.reinterpret %392 : <i32> -> <ui32>
                            %394 = ascvec.and %389, %333, %301 : <ui32>, <ui32> -> <ui32>
                            %395 = ascvec.shl %393, 11, %301 : <ui32> -> <ui32>
                            %396 = ascvec.shr %393, 21, %301 : <ui32> -> <ui32>
                            %397 = ascvec.or %395, %396, %301 : <ui32>, <ui32> -> <ui32>
                            %398 = ascvec.shr %397, 3, %301 : <ui32> -> <ui32>
                            %399 = ascvec.shl %397, 29, %301 : <ui32> -> <ui32>
                            %400 = ascvec.or %398, %399, %301 : <ui32>, <ui32> -> <ui32>
                            %401 = ascvec.and %394, %341, %301 : <ui32>, <ui32> -> <ui32>
                            %402 = ascvec.cmp_ne_scalar %401, %c0_i32_13, %301 : <ui32>, i32 -> !ascvec.mask
                            %403 = ascvec.select %400, %397, %402 : <ui32>
                            %404 = ascvec.shr %403, 8, %301 : <ui32> -> <ui32>
                            %405 = ascvec.shl %403, 24, %301 : <ui32> -> <ui32>
                            %406 = ascvec.or %404, %405, %301 : <ui32>, <ui32> -> <ui32>
                            %407 = ascvec.and %394, %348, %301 : <ui32>, <ui32> -> <ui32>
                            %408 = ascvec.cmp_ne_scalar %407, %c0_i32_13, %301 : <ui32>, i32 -> !ascvec.mask
                            %409 = ascvec.select %406, %403, %408 : <ui32>
                            %410 = ascvec.shr %409, 16, %301 : <ui32> -> <ui32>
                            %411 = ascvec.shl %409, 16, %301 : <ui32> -> <ui32>
                            %412 = ascvec.or %410, %411, %301 : <ui32>, <ui32> -> <ui32>
                            %413 = ascvec.and %394, %355, %301 : <ui32>, <ui32> -> <ui32>
                            %414 = ascvec.cmp_ne_scalar %413, %c0_i32_13, %301 : <ui32>, i32 -> !ascvec.mask
                            %415 = ascvec.select %412, %409, %414 : <ui32>
                            %416 = ascvec.and %415, %359, %301 : <ui32>, <ui32> -> <ui32>
                            %417 = ascvec.add_scalar %326, %c3_i32, %301 : <ui32>, i32
                            %418 = ascvec.shr %417, 3, %301 : <ui32> -> <ui32>
                            %419 = ascvec.add_scalar %418, %329, %301 : <ui32>, i32
                            %420 = ascvec.gather %arg9[%c0_14], %419, %301 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %421 = ascvec.reinterpret %420 : <i32> -> <ui32>
                            %422 = ascvec.and %417, %333, %301 : <ui32>, <ui32> -> <ui32>
                            %423 = ascvec.shl %421, 11, %301 : <ui32> -> <ui32>
                            %424 = ascvec.shr %421, 21, %301 : <ui32> -> <ui32>
                            %425 = ascvec.or %423, %424, %301 : <ui32>, <ui32> -> <ui32>
                            %426 = ascvec.shr %425, 3, %301 : <ui32> -> <ui32>
                            %427 = ascvec.shl %425, 29, %301 : <ui32> -> <ui32>
                            %428 = ascvec.or %426, %427, %301 : <ui32>, <ui32> -> <ui32>
                            %429 = ascvec.and %422, %341, %301 : <ui32>, <ui32> -> <ui32>
                            %430 = ascvec.cmp_ne_scalar %429, %c0_i32_13, %301 : <ui32>, i32 -> !ascvec.mask
                            %431 = ascvec.select %428, %425, %430 : <ui32>
                            %432 = ascvec.shr %431, 8, %301 : <ui32> -> <ui32>
                            %433 = ascvec.shl %431, 24, %301 : <ui32> -> <ui32>
                            %434 = ascvec.or %432, %433, %301 : <ui32>, <ui32> -> <ui32>
                            %435 = ascvec.and %422, %348, %301 : <ui32>, <ui32> -> <ui32>
                            %436 = ascvec.cmp_ne_scalar %435, %c0_i32_13, %301 : <ui32>, i32 -> !ascvec.mask
                            %437 = ascvec.select %434, %431, %436 : <ui32>
                            %438 = ascvec.shr %437, 16, %301 : <ui32> -> <ui32>
                            %439 = ascvec.shl %437, 16, %301 : <ui32> -> <ui32>
                            %440 = ascvec.or %438, %439, %301 : <ui32>, <ui32> -> <ui32>
                            %441 = ascvec.and %422, %355, %301 : <ui32>, <ui32> -> <ui32>
                            %442 = ascvec.cmp_ne_scalar %441, %c0_i32_13, %301 : <ui32>, i32 -> !ascvec.mask
                            %443 = ascvec.select %440, %437, %442 : <ui32>
                            %444 = ascvec.and %443, %359, %301 : <ui32>, <ui32> -> <ui32>
                            %445 = ascvec.reinterpret %321 : <f32> -> <ui32>
                            %446 = ascvec.reinterpret %445 : <ui32> -> <f32>
                            %447 = ascvec.max_scalar %446, %cst, %301 : <f32>, f32
                            %448 = ascvec.add %446, %447, %301 : <f32>
                            %449 = ascvec.reinterpret %448 : <f32> -> <ui32>
                            %450 = ascvec.shr %449, 5, %301 : <ui32> -> <ui32>
                            %451 = ascvec.add %450, %360, %301 : <ui32>
                            %452 = ascvec.reinterpret %323 : <f32> -> <ui32>
                            %453 = ascvec.reinterpret %452 : <ui32> -> <f32>
                            %454 = ascvec.max_scalar %453, %cst, %301 : <f32>, f32
                            %455 = ascvec.add %453, %454, %301 : <f32>
                            %456 = ascvec.reinterpret %455 : <f32> -> <ui32>
                            %457 = ascvec.shr %456, 5, %301 : <ui32> -> <ui32>
                            %458 = ascvec.add %457, %388, %301 : <ui32>
                            %459 = ascvec.reinterpret %322 : <f32> -> <ui32>
                            %460 = ascvec.reinterpret %459 : <ui32> -> <f32>
                            %461 = ascvec.max_scalar %460, %cst, %301 : <f32>, f32
                            %462 = ascvec.add %460, %461, %301 : <f32>
                            %463 = ascvec.reinterpret %462 : <f32> -> <ui32>
                            %464 = ascvec.shr %463, 5, %301 : <ui32> -> <ui32>
                            %465 = ascvec.add %464, %416, %301 : <ui32>
                            %466 = ascvec.reinterpret %324 : <f32> -> <ui32>
                            %467 = ascvec.reinterpret %466 : <ui32> -> <f32>
                            %468 = ascvec.max_scalar %467, %cst, %301 : <f32>, f32
                            %469 = ascvec.add %467, %468, %301 : <f32>
                            %470 = ascvec.reinterpret %469 : <f32> -> <ui32>
                            %471 = ascvec.shr %470, 5, %301 : <ui32> -> <ui32>
                            %472 = ascvec.add %471, %444, %301 : <ui32>
                            %473 = ascvec.reinterpret %451 : <ui32> -> <ui16>
                            %474 = ascvec.reinterpret %458 : <ui32> -> <ui16>
                            %result0_15, %result1_16 = ascvec.deintlv %473, %474 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %475 = ascvec.reinterpret %465 : <ui32> -> <ui16>
                            %476 = ascvec.reinterpret %472 : <ui32> -> <ui16>
                            %result0_17, %result1_18 = ascvec.deintlv %475, %476 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %477 = ascvec.reinterpret %result1_16 : <ui16> -> <ui8>
                            %478 = ascvec.reinterpret %result1_18 : <ui16> -> <ui8>
                            %result0_19, %result1_20 = ascvec.deintlv %477, %478 : <ui8>, <ui8> -> <ui8>, <ui8>
                            %479 = ascvec.gather_reg %result0_19, %311 : <ui8>, <ui8> -> <ui8>
                            %480 = ascvec.add_scalar %479, %c60_i8, %303 : <ui8>, i8
                            %481 = ascvec.cmp_ne_scalar %480, %c-1_i8, %303 : <ui8>, i8 -> !ascvec.mask
                            %482 = ascvec.min_scalar %480, %c123_i8, %481 : <ui8>, i8
                            %483 = arith.muli %arg15, %c512_i32 : i32
                            %484 = arith.index_cast %483 : i32 to index
                            %result0_21, %result1_22 = ascvec.loadalign_deintlv %arg10[%484] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %485 = ascvec.and %result1_22, %305, %303 : <ui8>, <ui8> -> <ui8>
                            %486 = ascvec.or %482, %485, %303 : <ui8>, <ui8> -> <ui8>
                            %487 = ascvec.arange %c0_i16 : i16 -> <ui16>
                            %488 = arith.muli %arg15, %c16_i32 : i32
                            %489 = arith.trunci %488 : i32 to i16
                            %490 = ascvec.add_scalar %487, %489, %302 : <ui16>, i16
                            %491 = ascvec.gather %arg11[%c0_14], %490, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                            %492 = ascvec.pack %491 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                            %493 = ascvec.gather_reg %492, %306 : <ui8>, <ui8> -> <ui8>
                            %494 = ascvec.cmp_eq_scalar %493, %c-1_i8, %303 : <ui8>, i8 -> !ascvec.mask
                            %495 = ascvec.duplicate_scalar %c127_i8, %494 merge %486 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                            ascvec.storealign %arg12[%313], %495, %303 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
                      }
                    }
                  }
                }
              } else {
                %223 = arith.divsi %98, %c256_i64 : i64
                %224 = arith.addi %98, %c1023_i64 : i64
                %225 = arith.remsi %224, %c1024_i64 : i64
                %226 = arith.cmpi ne, %225, %c0_i64 : i64
                %227 = arith.cmpi slt, %225, %c0_i64 : i64
                %228 = arith.andi %226, %227 : i1
                %229 = arith.divsi %224, %c1024_i64 : i64
                %230 = arith.extui %228 : i1 to i64
                %231 = arith.subi %229, %230 : i64
                %232 = arith.trunci %1 : i64 to i32
                %233 = arith.extui %232 : i32 to i64
                %234 = arith.subi %1, %233 : i64
                %235 = arith.remsi %234, %c4294967296_i64 : i64
                %236 = arith.cmpi ne, %235, %c0_i64 : i64
                %237 = arith.cmpi slt, %235, %c0_i64 : i64
                %238 = arith.andi %236, %237 : i1
                %239 = arith.divsi %234, %c4294967296_i64 : i64
                %240 = arith.extui %238 : i1 to i64
                %241 = arith.subi %239, %240 : i64
                %242 = arith.trunci %241 : i64 to i32
                %243 = arith.divui %211, %c32_i64 : i64
                %244 = arith.trunci %243 : i64 to i32
                %245 = arith.extui %244 : i32 to i64
                %246 = arith.subi %243, %245 : i64
                %247 = arith.remsi %246, %c4294967296_i64 : i64
                %248 = arith.cmpi ne, %247, %c0_i64 : i64
                %249 = arith.cmpi slt, %247, %c0_i64 : i64
                %250 = arith.andi %248, %249 : i1
                %251 = arith.divsi %246, %c4294967296_i64 : i64
                %252 = arith.extui %250 : i1 to i64
                %253 = arith.subi %251, %252 : i64
                %254 = arith.trunci %253 : i64 to i32
                %255 = arith.trunci %3 : i64 to i32
                %256 = arith.extui %255 : i32 to i64
                %257 = arith.subi %3, %256 : i64
                %258 = arith.remsi %257, %c4294967296_i64 : i64
                %259 = arith.cmpi ne, %258, %c0_i64 : i64
                %260 = arith.cmpi slt, %258, %c0_i64 : i64
                %261 = arith.andi %259, %260 : i1
                %262 = arith.divsi %257, %c4294967296_i64 : i64
                %263 = arith.extui %261 : i1 to i64
                %264 = arith.subi %262, %263 : i64
                %265 = arith.trunci %264 : i64 to i32
                %266 = arith.divui %255, %c4_i32 : i32
                %267 = arith.remui %265, %c4_i32 : i32
                %268 = arith.muli %267, %c1073741824_i32 : i32
                %269 = arith.addi %266, %268 : i32
                %270 = arith.divui %265, %c4_i32 : i32
                %271 = arith.muli %231, %c32_i64 : i64
                %272 = arith.divui %243, %c16384_i64 : i64
                %273 = arith.addi %243, %271 : i64
                %274 = arith.subi %273, %c1_i64 : i64
                %275 = arith.divui %274, %c16384_i64 : i64
                %276 = arith.cmpi eq, %272, %275 : i64
                scf.if %276 {
                  %286 = arith.trunci %272 : i64 to i32
                  %287 = arith.extui %286 : i32 to i64
                  %288 = arith.subi %272, %287 : i64
                  %289 = arith.remsi %288, %c4294967296_i64 : i64
                  %290 = arith.cmpi ne, %289, %c0_i64 : i64
                  %291 = arith.cmpi slt, %289, %c0_i64 : i64
                  %292 = arith.andi %290, %291 : i1
                  %293 = arith.divsi %288, %c4294967296_i64 : i64
                  %294 = arith.extui %292 : i1 to i64
                  %295 = arith.subi %293, %294 : i64
                  %296 = arith.trunci %295 : i64 to i32
                  %297 = arith.addi %269, %286 : i32
                  %298 = arith.addi %270, %296 : i32
                  %299 = arith.cmpi ult, %297, %269 : i32
                  %300 = arith.extui %299 : i1 to i32
                  %301 = arith.addi %298, %300 : i32
                  %302 = arith.extui %297 : i32 to i64
                  %303 = arith.muli %302, %c3449720151_i64 : i64
                  %304 = arith.trunci %303 : i64 to i32
                  %305 = arith.extui %304 : i32 to i64
                  %306 = arith.subi %303, %305 : i64
                  %307 = arith.remsi %306, %c4294967296_i64 : i64
                  %308 = arith.cmpi ne, %307, %c0_i64 : i64
                  %309 = arith.cmpi slt, %307, %c0_i64 : i64
                  %310 = arith.andi %308, %309 : i1
                  %311 = arith.divsi %306, %c4294967296_i64 : i64
                  %312 = arith.extui %310 : i1 to i64
                  %313 = arith.subi %311, %312 : i64
                  %314 = arith.trunci %313 : i64 to i32
                  %315 = arith.xori %301, %242 : i32
                  %316 = arith.extui %315 : i32 to i64
                  %317 = arith.muli %316, %c3449720151_i64 : i64
                  %318 = arith.trunci %317 : i64 to i32
                  %319 = arith.extui %318 : i32 to i64
                  %320 = arith.subi %317, %319 : i64
                  %321 = arith.remsi %320, %c4294967296_i64 : i64
                  %322 = arith.cmpi ne, %321, %c0_i64 : i64
                  %323 = arith.cmpi slt, %321, %c0_i64 : i64
                  %324 = arith.andi %322, %323 : i1
                  %325 = arith.divsi %320, %c4294967296_i64 : i64
                  %326 = arith.extui %324 : i1 to i64
                  %327 = arith.subi %325, %326 : i64
                  %328 = arith.trunci %327 : i64 to i32
                  %329 = arith.addi %232, %c-1640531527_i32 : i32
                  %330 = arith.addi %242, %c-1150833019_i32 : i32
                  %331 = arith.xori %328, %304 : i32
                  %332 = arith.xori %331, %329 : i32
                  %333 = arith.extui %332 : i32 to i64
                  %334 = arith.muli %333, %c3528531795_i64 : i64
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
                  %346 = arith.xori %314, %232 : i32
                  %347 = arith.addi %232, %c1013904242_i32 : i32
                  %348 = arith.xori %318, %347 : i32
                  %349 = arith.addi %242, %c1993301258_i32 : i32
                  %350 = arith.xori %345, %349 : i32
                  ascsimt.vf(%244, %346, %330, %348, %350, %232, %242, %335, %alloc_8, %271 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
                    %c256_i64_10 = arith.constant 256 : i64
                    %c3_i64_11 = arith.constant 3 : i64
                    %c2_i64_12 = arith.constant 2 : i64
                    %c1_i64_13 = arith.constant 1 : i64
                    %c4_i64_14 = arith.constant 4 : i64
                    %c1_i32_15 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c3_i32 = arith.constant 3 : i32
                    %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %351 = ascsimt.thread_idx {ascsimt.scope = 38 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %352 = arith.extsi %351 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg17 = %352 to %arg16 step %c256_i64_10  : i64 {
                      %353 = arith.trunci %arg17 : i64 to i32
                      %354 = arith.addi %arg7, %353 : i32
                      %355 = arith.remui %354, %c16384_i32 : i32
                      %356 = arith.xori %355, %arg8 : i32
                      %357 = arith.muli %356, %c-766435501_i32 : i32
                      %358 = "ascsimt.bitcast"(%356) : (i32) -> ui32
                      %359 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %360 = ascsimt.math_binary %358, %359, "mulhi" : ui32, ui32 -> ui32
                      %361 = "ascsimt.bitcast"(%360) : (ui32) -> i32
                      %362 = arith.xori %361, %arg9 : i32
                      %363 = arith.muli %362, %c-845247145_i32 : i32
                      %364 = "ascsimt.bitcast"(%362) : (i32) -> ui32
                      %365 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %366 = ascsimt.math_binary %364, %365, "mulhi" : ui32, ui32 -> ui32
                      %367 = "ascsimt.bitcast"(%366) : (ui32) -> i32
                      %368 = arith.xori %367, %arg10 : i32
                      %369 = arith.xori %357, %arg11 : i32
                      %370:4 = scf.for %arg18 = %c3_i32 to %c10_i32 step %c1_i32_15 iter_args(%arg19 = %368, %arg20 = %363, %arg21 = %369, %arg22 = %arg14) -> (i32, i32, i32, i32)  : i32 {
                        %379 = arith.muli %arg19, %c-766435501_i32 : i32
                        %380 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %381 = ascsimt.math_binary %380, %359, "mulhi" : ui32, ui32 -> ui32
                        %382 = "ascsimt.bitcast"(%381) : (ui32) -> i32
                        %383 = arith.muli %arg21, %c-845247145_i32 : i32
                        %384 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %385 = ascsimt.math_binary %384, %365, "mulhi" : ui32, ui32 -> ui32
                        %386 = "ascsimt.bitcast"(%385) : (ui32) -> i32
                        %387 = arith.muli %arg18, %c-1640531527_i32_17 : i32
                        %388 = arith.addi %arg12, %387 : i32
                        %389 = arith.muli %arg18, %c-1150833019_i32_16 : i32
                        %390 = arith.addi %arg13, %389 : i32
                        %391 = arith.xori %386, %arg20 : i32
                        %392 = arith.xori %391, %388 : i32
                        %393 = arith.xori %382, %arg22 : i32
                        %394 = arith.xori %393, %390 : i32
                        scf.yield %392, %383, %394, %379 : i32, i32, i32, i32
                      }
                      %371 = arith.muli %arg17, %c4_i64_14 : i64
                      %372 = arith.index_cast %371 : i64 to index
                      ascsimt.store %arg15[%372], %370#0 : memref<8192xi32, 1>, i32
                      %373 = arith.addi %371, %c1_i64_13 : i64
                      %374 = arith.index_cast %373 : i64 to index
                      ascsimt.store %arg15[%374], %370#1 : memref<8192xi32, 1>, i32
                      %375 = arith.addi %371, %c2_i64_12 : i64
                      %376 = arith.index_cast %375 : i64 to index
                      ascsimt.store %arg15[%376], %370#2 : memref<8192xi32, 1>, i32
                      %377 = arith.addi %371, %c3_i64_11 : i64
                      %378 = arith.index_cast %377 : i64 to index
                      ascsimt.store %arg15[%378], %370#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%244, %254, %269, %270, %232, %242, %alloc_8, %271 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: memref<8192xi32, 1>, %arg14: i64):
                    %c256_i64_10 = arith.constant 256 : i64
                    %c3_i64_11 = arith.constant 3 : i64
                    %c2_i64_12 = arith.constant 2 : i64
                    %c1_i64_13 = arith.constant 1 : i64
                    %c4_i64_14 = arith.constant 4 : i64
                    %c1_i32_15 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c0_i32_16 = arith.constant 0 : i32
                    %c-1150833019_i32_17 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_18 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c262144_i32 = arith.constant 262144 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %286 = ascsimt.thread_idx {ascsimt.scope = 39 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %287 = arith.extsi %286 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg15 = %287 to %arg14 step %c256_i64_10  : i64 {
                      %288 = arith.trunci %arg15 : i64 to i32
                      %289 = arith.addi %arg7, %288 : i32
                      %290 = arith.cmpi ult, %289, %arg7 : i32
                      %291 = arith.extui %290 : i1 to i32
                      %292 = arith.addi %arg8, %291 : i32
                      %293 = arith.divui %289, %c16384_i32 : i32
                      %294 = arith.remui %292, %c16384_i32 : i32
                      %295 = arith.muli %294, %c262144_i32 : i32
                      %296 = arith.addi %293, %295 : i32
                      %297 = arith.divui %292, %c16384_i32 : i32
                      %298 = arith.addi %arg9, %296 : i32
                      %299 = arith.addi %arg10, %297 : i32
                      %300 = arith.cmpi ult, %298, %arg9 : i32
                      %301 = arith.extui %300 : i1 to i32
                      %302 = arith.addi %299, %301 : i32
                      %303 = arith.remui %289, %c16384_i32 : i32
                      %304:4 = scf.for %arg16 = %c0_i32_16 to %c10_i32 step %c1_i32_15 iter_args(%arg17 = %c0_i32_16, %arg18 = %303, %arg19 = %298, %arg20 = %302) -> (i32, i32, i32, i32)  : i32 {
                        %313 = arith.muli %arg17, %c-766435501_i32 : i32
                        %314 = "ascsimt.bitcast"(%arg17) : (i32) -> ui32
                        %315 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %316 = ascsimt.math_binary %314, %315, "mulhi" : ui32, ui32 -> ui32
                        %317 = "ascsimt.bitcast"(%316) : (ui32) -> i32
                        %318 = arith.muli %arg19, %c-845247145_i32 : i32
                        %319 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %320 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %321 = ascsimt.math_binary %319, %320, "mulhi" : ui32, ui32 -> ui32
                        %322 = "ascsimt.bitcast"(%321) : (ui32) -> i32
                        %323 = arith.muli %arg16, %c-1640531527_i32_18 : i32
                        %324 = arith.addi %arg11, %323 : i32
                        %325 = arith.muli %arg16, %c-1150833019_i32_17 : i32
                        %326 = arith.addi %arg12, %325 : i32
                        %327 = arith.xori %322, %arg18 : i32
                        %328 = arith.xori %327, %324 : i32
                        %329 = arith.xori %317, %arg20 : i32
                        %330 = arith.xori %329, %326 : i32
                        scf.yield %328, %318, %330, %313 : i32, i32, i32, i32
                      }
                      %305 = arith.muli %arg15, %c4_i64_14 : i64
                      %306 = arith.index_cast %305 : i64 to index
                      ascsimt.store %arg13[%306], %304#0 : memref<8192xi32, 1>, i32
                      %307 = arith.addi %305, %c1_i64_13 : i64
                      %308 = arith.index_cast %307 : i64 to index
                      ascsimt.store %arg13[%308], %304#1 : memref<8192xi32, 1>, i32
                      %309 = arith.addi %305, %c2_i64_12 : i64
                      %310 = arith.index_cast %309 : i64 to index
                      ascsimt.store %arg13[%310], %304#2 : memref<8192xi32, 1>, i32
                      %311 = arith.addi %305, %c3_i64_11 : i64
                      %312 = arith.index_cast %311 : i64 to index
                      ascsimt.store %arg13[%312], %304#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %277 = arith.trunci %31 : i64 to i32
                %278 = arith.divsi %277, %c2_i32 : i32
                %279 = arith.trunci %278 : i32 to i8
                %280 = arith.trunci %223 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %281 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%279, %alloc_6, %133, %281, %280, %alloc_9 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, i32, memref<2048xui16, 1>) {
                ^bb0(%arg7: i8, %arg8: memref<2048xui8, 1>, %arg9: memref<32768xbf16, 1>, %arg10: memref<?xui16, 1>, %arg11: i32, %arg12: memref<2048xui16, 1>):
                  %c1_i32_10 = arith.constant 1 : i32
                  %c0_i32_11 = arith.constant 0 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c0_12 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %286 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %287 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %288 = ascvec.duplicate_scalar %c32767_i16, %286 : i16 -> <ui16>
                  %289 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %290 = ascvec.shr %289, 4, %286 : <ui16> -> <ui16>
                  %291 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %292 = ascvec.duplicate_scalar %c1_i8, %287 : i8 -> <ui8>
                  %293 = ascvec.and %291, %292, %287 : <ui8>, <ui8> -> <ui8>
                  %294 = ascvec.cmp_lt_scalar %291, %c16_i8, %287 : <ui8>, i8 -> !ascvec.mask
                  %295 = ascvec.cmp_lt_scalar %293, %arg7, %287 : <ui8>, i8 -> !ascvec.mask
                  %296 = ascvec.mask_and %294, %295, %287
                  ascvec.store_unalign_begin %arg8 : memref<2048xui8, 1>
                  %297 = ascvec.duplicate_scalar %c-1_i16, %286 : i16 -> <ui16>
                  %298 = ascvec.duplicate_scalar %c0_i16, %286 : i16 -> <ui16>
                  %299:2 = scf.for %arg13 = %c0_i32_11 to %arg11 step %c1_i32_10 iter_args(%arg14 = %298, %arg15 = %297) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i32 {
                    %300 = arith.muli %arg13, %c256_i32 : i32
                    %301 = arith.index_cast %300 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg9[%301] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %302 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %303 = ascvec.and %302, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %304 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %305 = ascvec.and %304, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %306 = ascvec.max %303, %305, %286 : <ui16>
                    %307 = ascvec.reduce_max_datablock %306, %286 : <ui16>
                    %308 = ascvec.shr %307, 7, %286 : <ui16> -> <ui16>
                    %309 = ascvec.cmp_ge_scalar %308, %c255_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %310 = ascvec.max_scalar %308, %c15_i16, %286 : <ui16>, i16
                    %311 = ascvec.add_scalar %310, %c-15_i16, %286 : <ui16>, i16
                    %312 = ascvec.duplicate_scalar %c255_i16, %309 merge %311 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %313 = ascvec.duplicate_scalar %c254_i16, %286 : i16 -> <ui16>
                    %314 = ascvec.sub %313, %312, %286 : <ui16>
                    %315 = ascvec.cmp_ne_scalar %312, %c0_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %316 = ascvec.shl %314, 7, %315 : <ui16> -> <ui16>
                    %317 = ascvec.max %arg14, %312, %mask : <ui16>
                    %318 = ascvec.gather_reg %316, %290 : <ui16>, <ui16> -> <ui16>
                    %319 = ascvec.reinterpret %318 : <ui16> -> <bf16>
                    %320 = ascvec.mul %result0, %319, %286 : <bf16>
                    %321 = ascvec.mul %result1, %319, %286 : <bf16>
                    ascvec.storealign_intlv %arg9[%301], %320, %321 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %322 = ascvec.reinterpret %320 : <bf16> -> <ui16>
                    %323 = ascvec.and %322, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %324 = ascvec.reinterpret %321 : <bf16> -> <ui16>
                    %325 = ascvec.and %324, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %326 = ascvec.min %323, %325, %286 : <ui16>
                    %327 = ascvec.min %arg15, %326, %286 : <ui16>
                    %328 = arith.muli %arg13, %c16_i32 : i32
                    %329 = arith.index_cast %328 : i32 to index
                    ascvec.storealign %arg10[%329], %312, %mask : memref<?xui16, 1>, index, <ui16>
                    %330 = ascvec.reinterpret %312 : <ui16> -> <ui8>
                    %331 = ascvec.squeeze_store_reg %330, %296 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_12], %331 : memref<2048xui8, 1>, <ui8>
                    scf.yield %317, %327 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                  }
                  %c0_13 = arith.constant 0 : index
                  %c1_14 = arith.constant 1 : index
                  scf.for %arg13 = %c0_13 to %c1_14 step %c1_14 {
                    %300 = ascvec.reduce_min %299#1, %286 : <ui16>
                    %301 = ascvec.reduce_max %299#0, %mask : <ui16>
                    %302 = ascvec.cmp_eq_scalar %301, %c255_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %303 = ascvec.duplicate_scalar %c0_i16, %302 merge %300 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.store_unalign_post %arg8[%c0_12] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg12[%c0_12], %303 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %282 = memref.load %alloc_9[%c0] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %283 = cannir.scalar_storage_cast %282 : ui16 -> i16
                %284 = arith.extui %283 : i16 to i64
                %285 = arith.cmpi sge, %284, %c14464_i64 : i64
                scf.if %285 {
                  ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %286 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                  %287 = ascscalar.bitcast_memref %133 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%133, %286, %287, %alloc_5, %280 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg7: memref<32768xbf16, 1>, %arg8: memref<?xui8, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<32768xui8, 1>, %arg11: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_10 = arith.constant 1 : i32
                    %c0_i32_11 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c128_i32 = arith.constant 128 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c31_i16 = arith.constant 31 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %288 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %289 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %290 = ascvec.duplicate_scalar %c255_i16, %288 : i16 -> <ui16>
                    %291 = ascvec.duplicate_scalar %c-128_i8, %289 : i8 -> <ui8>
                    %292 = ascvec.duplicate_scalar %c31_i16, %288 : i16 -> <ui16>
                    scf.for %arg12 = %c0_i32_11 to %arg11 step %c1_i32_10  : i32 {
                      %293 = arith.muli %arg12, %c256_i32 : i32
                      %294 = arith.index_cast %293 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg7[%294] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %295 = arith.muli %arg12, %c128_i32 : i32
                      %296 = arith.index_cast %295 : i32 to index
                      %297 = ascvec.loadalign %arg8[%296] : memref<?xui8, 1>, index -> <ui8>
                      %298 = ascvec.unpack %297 {part = "lower"} : <ui8> -> <ui16>
                      %299 = ascvec.and %298, %292, %288 : <ui16>, <ui16> -> <ui16>
                      %300 = ascvec.shr %298, 3, %288 : <ui16> -> <ui16>
                      %301 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %302 = ascvec.add %301, %299, %288 : <ui16>
                      %303 = ascvec.shr %302, 5, %288 : <ui16> -> <ui16>
                      %304 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %305 = ascvec.add %304, %300, %288 : <ui16>
                      %306 = ascvec.shr %305, 5, %288 : <ui16> -> <ui16>
                      %307 = ascvec.and %303, %290, %288 : <ui16>, <ui16> -> <ui16>
                      %308 = ascvec.shl %306, 8, %288 : <ui16> -> <ui16>
                      %309 = ascvec.or %307, %308, %288 : <ui16>, <ui16> -> <ui16>
                      %310 = ascvec.reinterpret %309 : <ui16> -> <ui8>
                      %311 = ascvec.add_scalar %310, %c64_i8, %289 : <ui8>, i8
                      %312 = ascvec.min_scalar %311, %c123_i8, %289 : <ui8>, i8
                      %313 = arith.muli %arg12, %c512_i32 : i32
                      %314 = arith.index_cast %313 : i32 to index
                      %result0_12, %result1_13 = ascvec.loadalign_deintlv %arg9[%314] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %315 = ascvec.and %result1_13, %291, %289 : <ui8>, <ui8> -> <ui8>
                      %316 = ascvec.or %312, %315, %289 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg10[%294], %316, %289 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
                } else {
                  %286 = arith.cmpi sge, %284, %c13568_i64 : i64
                  scf.if %286 {
                    ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %287 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                    %288 = ascscalar.bitcast_memref %133 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                    ascvec.vf(%133, %287, %288, %alloc_5, %280 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                    ^bb0(%arg7: memref<32768xbf16, 1>, %arg8: memref<?xui8, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<32768xui8, 1>, %arg11: i32):
                      %c512_i32 = arith.constant 512 : i32
                      %c1_i32_10 = arith.constant 1 : i32
                      %c0_i32_11 = arith.constant 0 : i32
                      %c123_i8 = arith.constant 123 : i8
                      %c64_i8 = arith.constant 64 : i8
                      %c448_i16 = arith.constant 448 : i16
                      %c14464_i16 = arith.constant 14464 : i16
                      %c128_i32 = arith.constant 128 : i32
                      %c256_i32 = arith.constant 256 : i32
                      %cst = arith.constant 2.684350e+08 : bf16
                      %c32767_i16 = arith.constant 32767 : i16
                      %c4095_i16 = arith.constant 4095 : i16
                      %c3_i8 = arith.constant 3 : i8
                      %c1_i8 = arith.constant 1 : i8
                      %c0_i8 = arith.constant 0 : i8
                      %c-128_i8 = arith.constant -128 : i8
                      %c255_i16 = arith.constant 255 : i16
                      %289 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                      %290 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                      %291 = ascvec.duplicate_scalar %c255_i16, %289 : i16 -> <ui16>
                      %292 = ascvec.duplicate_scalar %c-128_i8, %290 : i8 -> <ui8>
                      %293 = ascvec.arange %c0_i8 : i8 -> <ui8>
                      %294 = ascvec.shr %293, 1, %290 : <ui8> -> <ui8>
                      %295 = ascvec.shr %294, 2, %290 : <ui8> -> <ui8>
                      %296 = ascvec.shl %295, 2, %290 : <ui8> -> <ui8>
                      %297 = ascvec.duplicate_scalar %c1_i8, %290 : i8 -> <ui8>
                      %298 = ascvec.and %293, %297, %290 : <ui8>, <ui8> -> <ui8>
                      %299 = ascvec.add_scalar %294, %c3_i8, %290 : <ui8>, i8
                      %300 = ascvec.add %299, %298, %290 : <ui8>
                      %301 = ascvec.duplicate_scalar %c3_i8, %290 : i8 -> <ui8>
                      %302 = ascvec.and %300, %301, %290 : <ui8>, <ui8> -> <ui8>
                      %303 = ascvec.or %296, %302, %290 : <ui8>, <ui8> -> <ui8>
                      %304 = ascvec.duplicate_scalar %c4095_i16, %289 : i16 -> <ui16>
                      %305 = ascvec.duplicate_scalar %c32767_i16, %289 : i16 -> <ui16>
                      %306 = ascvec.duplicate_scalar %cst, %289 : bf16 -> <bf16>
                      scf.for %arg12 = %c0_i32_11 to %arg11 step %c1_i32_10  : i32 {
                        %307 = arith.muli %arg12, %c256_i32 : i32
                        %308 = arith.index_cast %307 : i32 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg7[%308] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                        %309 = arith.muli %arg12, %c128_i32 : i32
                        %310 = arith.index_cast %309 : i32 to index
                        %311 = ascvec.loadalign %arg8[%310] : memref<?xui8, 1>, index -> <ui8>
                        %312 = ascvec.gather_reg %311, %303 : <ui8>, <ui8> -> <ui8>
                        %313 = ascvec.reinterpret %312 : <ui8> -> <ui16>
                        %314 = ascvec.shr %313, 1, %289 : <ui16> -> <ui16>
                        %315 = ascvec.and %314, %304, %289 : <ui16>, <ui16> -> <ui16>
                        %316 = ascvec.shr %313, 4, %289 : <ui16> -> <ui16>
                        %317 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                        %318 = ascvec.and %317, %305, %289 : <ui16>, <ui16> -> <ui16>
                        %319 = ascvec.cmp_lt_scalar %318, %c14464_i16, %289 : <ui16>, i16 -> !ascvec.mask
                        %320 = ascvec.shr %315, 7, %289 : <ui16> -> <ui16>
                        %321 = ascvec.add %318, %320, %289 : <ui16>
                        %322 = ascvec.shr %321, 5, %289 : <ui16> -> <ui16>
                        %323 = ascvec.reinterpret %318 : <ui16> -> <bf16>
                        %324 = ascvec.mul %323, %306, %319 : <bf16>
                        %325 = ascvec.cast %324, %289 {rounding = 4 : i32} : <bf16> -> <f16>
                        %326 = ascvec.cast %325, %289 {rounding = 4 : i32} : <f16> -> <i16>
                        %327 = ascvec.reinterpret %326 : <i16> -> <ui16>
                        %328 = ascvec.add %327, %315, %289 : <ui16>
                        %329 = ascvec.shr %328, 12, %289 : <ui16> -> <ui16>
                        %330 = ascvec.add_scalar %329, %c448_i16, %289 : <ui16>, i16
                        %331 = ascvec.select %330, %322, %319 : <ui16>
                        %332 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                        %333 = ascvec.and %332, %305, %289 : <ui16>, <ui16> -> <ui16>
                        %334 = ascvec.cmp_lt_scalar %333, %c14464_i16, %289 : <ui16>, i16 -> !ascvec.mask
                        %335 = ascvec.shr %316, 7, %289 : <ui16> -> <ui16>
                        %336 = ascvec.add %333, %335, %289 : <ui16>
                        %337 = ascvec.shr %336, 5, %289 : <ui16> -> <ui16>
                        %338 = ascvec.reinterpret %333 : <ui16> -> <bf16>
                        %339 = ascvec.mul %338, %306, %334 : <bf16>
                        %340 = ascvec.cast %339, %289 {rounding = 4 : i32} : <bf16> -> <f16>
                        %341 = ascvec.cast %340, %289 {rounding = 4 : i32} : <f16> -> <i16>
                        %342 = ascvec.reinterpret %341 : <i16> -> <ui16>
                        %343 = ascvec.add %342, %316, %289 : <ui16>
                        %344 = ascvec.shr %343, 12, %289 : <ui16> -> <ui16>
                        %345 = ascvec.add_scalar %344, %c448_i16, %289 : <ui16>, i16
                        %346 = ascvec.select %345, %337, %334 : <ui16>
                        %347 = ascvec.and %331, %291, %289 : <ui16>, <ui16> -> <ui16>
                        %348 = ascvec.shl %346, 8, %289 : <ui16> -> <ui16>
                        %349 = ascvec.or %347, %348, %289 : <ui16>, <ui16> -> <ui16>
                        %350 = ascvec.reinterpret %349 : <ui16> -> <ui8>
                        %351 = ascvec.add_scalar %350, %c64_i8, %290 : <ui8>, i8
                        %352 = ascvec.min_scalar %351, %c123_i8, %290 : <ui8>, i8
                        %353 = arith.muli %arg12, %c512_i32 : i32
                        %354 = arith.index_cast %353 : i32 to index
                        %result0_12, %result1_13 = ascvec.loadalign_deintlv %arg9[%354] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                        %355 = ascvec.and %result1_13, %292, %290 : <ui8>, <ui8> -> <ui8>
                        %356 = ascvec.or %352, %355, %290 : <ui8>, <ui8> -> <ui8>
                        ascvec.storealign %arg10[%308], %356, %290 : memref<32768xui8, 1>, index, <ui8>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
                  } else {
                    ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %287 = ascscalar.bitcast_memref %133 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                    ascvec.vf(%133, %alloc_8, %287, %281, %alloc_5, %280 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32) {
                    ^bb0(%arg7: memref<32768xbf16, 1>, %arg8: memref<8192xi32, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<?xui16, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32):
                      %c512_i32 = arith.constant 512 : i32
                      %c127_i8 = arith.constant 127 : i8
                      %c16_i32 = arith.constant 16 : i32
                      %c0_i16 = arith.constant 0 : i16
                      %c123_i8 = arith.constant 123 : i8
                      %c-1_i8 = arith.constant -1 : i8
                      %c60_i8 = arith.constant 60 : i8
                      %cst = arith.constant 6.10351526E-5 : f32
                      %c3_i32 = arith.constant 3 : i32
                      %c65535_i32 = arith.constant 65535 : i32
                      %c4_i32_10 = arith.constant 4 : i32
                      %c2_i32_11 = arith.constant 2 : i32
                      %c1_i32_12 = arith.constant 1 : i32
                      %c7_i32 = arith.constant 7 : i32
                      %c32_i32 = arith.constant 32 : i32
                      %c0_i32_13 = arith.constant 0 : i32
                      %c32767_i16 = arith.constant 32767 : i16
                      %c0_14 = arith.constant 0 : index
                      %c256_i32 = arith.constant 256 : i32
                      %c3_i8 = arith.constant 3 : i8
                      %c-128_i8 = arith.constant -128 : i8
                      %c0_i8 = arith.constant 0 : i8
                      %c8_i32 = arith.constant 8 : i32
                      %288 = ascvec.create_mask : !ascvec.mask
                      %289 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                      %290 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                      %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                      %291 = ascvec.arange %c0_i8 : i8 -> <ui8>
                      %292 = ascvec.duplicate_scalar %c-128_i8, %290 : i8 -> <ui8>
                      %293 = ascvec.shr %291, 5, %290 : <ui8> -> <ui8>
                      %294 = ascvec.duplicate_scalar %c3_i8, %290 : i8 -> <ui8>
                      %295 = ascvec.and %291, %294, %290 : <ui8>, <ui8> -> <ui8>
                      %296 = ascvec.shl %295, 6, %290 : <ui8> -> <ui8>
                      %297 = ascvec.shr %291, 2, %290 : <ui8> -> <ui8>
                      %298 = ascvec.or %296, %297, %290 : <ui8>, <ui8> -> <ui8>
                      scf.for %arg13 = %c0_i32_13 to %arg12 step %c1_i32_12  : i32 {
                        %299 = arith.muli %arg13, %c256_i32 : i32
                        %300 = arith.index_cast %299 : i32 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg7[%300] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                        %301 = ascvec.duplicate_scalar %c32767_i16, %289 : i16 -> <ui16>
                        %302 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                        %303 = ascvec.and %302, %301, %289 : <ui16>, <ui16> -> <ui16>
                        %304 = ascvec.reinterpret %303 : <ui16> -> <bf16>
                        %305 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                        %306 = ascvec.and %305, %301, %289 : <ui16>, <ui16> -> <ui16>
                        %307 = ascvec.reinterpret %306 : <ui16> -> <bf16>
                        %308 = ascvec.cast %304, %289 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %309 = ascvec.cast %304, %289 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %310 = ascvec.cast %307, %289 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %311 = ascvec.cast %307, %289 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %312 = ascvec.arange %c0_i32_13 : i32 -> <ui32>
                        %313 = ascvec.shl %312, 2, %288 : <ui32> -> <ui32>
                        %314 = ascvec.add_scalar %313, %c0_i32_13, %288 : <ui32>, i32
                        %315 = ascvec.shr %314, 3, %288 : <ui32> -> <ui32>
                        %316 = arith.muli %arg13, %c32_i32 : i32
                        %317 = ascvec.add_scalar %315, %316, %288 : <ui32>, i32
                        %318 = ascvec.gather %arg8[%c0_14], %317, %288 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %319 = ascvec.reinterpret %318 : <i32> -> <ui32>
                        %320 = ascvec.duplicate_scalar %c7_i32, %288 : i32 -> <ui32>
                        %321 = ascvec.and %314, %320, %288 : <ui32>, <ui32> -> <ui32>
                        %322 = ascvec.shl %319, 11, %288 : <ui32> -> <ui32>
                        %323 = ascvec.shr %319, 21, %288 : <ui32> -> <ui32>
                        %324 = ascvec.or %322, %323, %288 : <ui32>, <ui32> -> <ui32>
                        %325 = ascvec.shr %324, 3, %288 : <ui32> -> <ui32>
                        %326 = ascvec.shl %324, 29, %288 : <ui32> -> <ui32>
                        %327 = ascvec.or %325, %326, %288 : <ui32>, <ui32> -> <ui32>
                        %328 = ascvec.duplicate_scalar %c1_i32_12, %288 : i32 -> <ui32>
                        %329 = ascvec.and %321, %328, %288 : <ui32>, <ui32> -> <ui32>
                        %330 = ascvec.cmp_ne_scalar %329, %c0_i32_13, %288 : <ui32>, i32 -> !ascvec.mask
                        %331 = ascvec.select %327, %324, %330 : <ui32>
                        %332 = ascvec.shr %331, 8, %288 : <ui32> -> <ui32>
                        %333 = ascvec.shl %331, 24, %288 : <ui32> -> <ui32>
                        %334 = ascvec.or %332, %333, %288 : <ui32>, <ui32> -> <ui32>
                        %335 = ascvec.duplicate_scalar %c2_i32_11, %288 : i32 -> <ui32>
                        %336 = ascvec.and %321, %335, %288 : <ui32>, <ui32> -> <ui32>
                        %337 = ascvec.cmp_ne_scalar %336, %c0_i32_13, %288 : <ui32>, i32 -> !ascvec.mask
                        %338 = ascvec.select %334, %331, %337 : <ui32>
                        %339 = ascvec.shr %338, 16, %288 : <ui32> -> <ui32>
                        %340 = ascvec.shl %338, 16, %288 : <ui32> -> <ui32>
                        %341 = ascvec.or %339, %340, %288 : <ui32>, <ui32> -> <ui32>
                        %342 = ascvec.duplicate_scalar %c4_i32_10, %288 : i32 -> <ui32>
                        %343 = ascvec.and %321, %342, %288 : <ui32>, <ui32> -> <ui32>
                        %344 = ascvec.cmp_ne_scalar %343, %c0_i32_13, %288 : <ui32>, i32 -> !ascvec.mask
                        %345 = ascvec.select %341, %338, %344 : <ui32>
                        %346 = ascvec.duplicate_scalar %c65535_i32, %288 : i32 -> <ui32>
                        %347 = ascvec.and %345, %346, %288 : <ui32>, <ui32> -> <ui32>
                        %348 = ascvec.add_scalar %313, %c1_i32_12, %288 : <ui32>, i32
                        %349 = ascvec.shr %348, 3, %288 : <ui32> -> <ui32>
                        %350 = ascvec.add_scalar %349, %316, %288 : <ui32>, i32
                        %351 = ascvec.gather %arg8[%c0_14], %350, %288 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %352 = ascvec.reinterpret %351 : <i32> -> <ui32>
                        %353 = ascvec.and %348, %320, %288 : <ui32>, <ui32> -> <ui32>
                        %354 = ascvec.shl %352, 11, %288 : <ui32> -> <ui32>
                        %355 = ascvec.shr %352, 21, %288 : <ui32> -> <ui32>
                        %356 = ascvec.or %354, %355, %288 : <ui32>, <ui32> -> <ui32>
                        %357 = ascvec.shr %356, 3, %288 : <ui32> -> <ui32>
                        %358 = ascvec.shl %356, 29, %288 : <ui32> -> <ui32>
                        %359 = ascvec.or %357, %358, %288 : <ui32>, <ui32> -> <ui32>
                        %360 = ascvec.and %353, %328, %288 : <ui32>, <ui32> -> <ui32>
                        %361 = ascvec.cmp_ne_scalar %360, %c0_i32_13, %288 : <ui32>, i32 -> !ascvec.mask
                        %362 = ascvec.select %359, %356, %361 : <ui32>
                        %363 = ascvec.shr %362, 8, %288 : <ui32> -> <ui32>
                        %364 = ascvec.shl %362, 24, %288 : <ui32> -> <ui32>
                        %365 = ascvec.or %363, %364, %288 : <ui32>, <ui32> -> <ui32>
                        %366 = ascvec.and %353, %335, %288 : <ui32>, <ui32> -> <ui32>
                        %367 = ascvec.cmp_ne_scalar %366, %c0_i32_13, %288 : <ui32>, i32 -> !ascvec.mask
                        %368 = ascvec.select %365, %362, %367 : <ui32>
                        %369 = ascvec.shr %368, 16, %288 : <ui32> -> <ui32>
                        %370 = ascvec.shl %368, 16, %288 : <ui32> -> <ui32>
                        %371 = ascvec.or %369, %370, %288 : <ui32>, <ui32> -> <ui32>
                        %372 = ascvec.and %353, %342, %288 : <ui32>, <ui32> -> <ui32>
                        %373 = ascvec.cmp_ne_scalar %372, %c0_i32_13, %288 : <ui32>, i32 -> !ascvec.mask
                        %374 = ascvec.select %371, %368, %373 : <ui32>
                        %375 = ascvec.and %374, %346, %288 : <ui32>, <ui32> -> <ui32>
                        %376 = ascvec.add_scalar %313, %c2_i32_11, %288 : <ui32>, i32
                        %377 = ascvec.shr %376, 3, %288 : <ui32> -> <ui32>
                        %378 = ascvec.add_scalar %377, %316, %288 : <ui32>, i32
                        %379 = ascvec.gather %arg8[%c0_14], %378, %288 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %380 = ascvec.reinterpret %379 : <i32> -> <ui32>
                        %381 = ascvec.and %376, %320, %288 : <ui32>, <ui32> -> <ui32>
                        %382 = ascvec.shl %380, 11, %288 : <ui32> -> <ui32>
                        %383 = ascvec.shr %380, 21, %288 : <ui32> -> <ui32>
                        %384 = ascvec.or %382, %383, %288 : <ui32>, <ui32> -> <ui32>
                        %385 = ascvec.shr %384, 3, %288 : <ui32> -> <ui32>
                        %386 = ascvec.shl %384, 29, %288 : <ui32> -> <ui32>
                        %387 = ascvec.or %385, %386, %288 : <ui32>, <ui32> -> <ui32>
                        %388 = ascvec.and %381, %328, %288 : <ui32>, <ui32> -> <ui32>
                        %389 = ascvec.cmp_ne_scalar %388, %c0_i32_13, %288 : <ui32>, i32 -> !ascvec.mask
                        %390 = ascvec.select %387, %384, %389 : <ui32>
                        %391 = ascvec.shr %390, 8, %288 : <ui32> -> <ui32>
                        %392 = ascvec.shl %390, 24, %288 : <ui32> -> <ui32>
                        %393 = ascvec.or %391, %392, %288 : <ui32>, <ui32> -> <ui32>
                        %394 = ascvec.and %381, %335, %288 : <ui32>, <ui32> -> <ui32>
                        %395 = ascvec.cmp_ne_scalar %394, %c0_i32_13, %288 : <ui32>, i32 -> !ascvec.mask
                        %396 = ascvec.select %393, %390, %395 : <ui32>
                        %397 = ascvec.shr %396, 16, %288 : <ui32> -> <ui32>
                        %398 = ascvec.shl %396, 16, %288 : <ui32> -> <ui32>
                        %399 = ascvec.or %397, %398, %288 : <ui32>, <ui32> -> <ui32>
                        %400 = ascvec.and %381, %342, %288 : <ui32>, <ui32> -> <ui32>
                        %401 = ascvec.cmp_ne_scalar %400, %c0_i32_13, %288 : <ui32>, i32 -> !ascvec.mask
                        %402 = ascvec.select %399, %396, %401 : <ui32>
                        %403 = ascvec.and %402, %346, %288 : <ui32>, <ui32> -> <ui32>
                        %404 = ascvec.add_scalar %313, %c3_i32, %288 : <ui32>, i32
                        %405 = ascvec.shr %404, 3, %288 : <ui32> -> <ui32>
                        %406 = ascvec.add_scalar %405, %316, %288 : <ui32>, i32
                        %407 = ascvec.gather %arg8[%c0_14], %406, %288 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %408 = ascvec.reinterpret %407 : <i32> -> <ui32>
                        %409 = ascvec.and %404, %320, %288 : <ui32>, <ui32> -> <ui32>
                        %410 = ascvec.shl %408, 11, %288 : <ui32> -> <ui32>
                        %411 = ascvec.shr %408, 21, %288 : <ui32> -> <ui32>
                        %412 = ascvec.or %410, %411, %288 : <ui32>, <ui32> -> <ui32>
                        %413 = ascvec.shr %412, 3, %288 : <ui32> -> <ui32>
                        %414 = ascvec.shl %412, 29, %288 : <ui32> -> <ui32>
                        %415 = ascvec.or %413, %414, %288 : <ui32>, <ui32> -> <ui32>
                        %416 = ascvec.and %409, %328, %288 : <ui32>, <ui32> -> <ui32>
                        %417 = ascvec.cmp_ne_scalar %416, %c0_i32_13, %288 : <ui32>, i32 -> !ascvec.mask
                        %418 = ascvec.select %415, %412, %417 : <ui32>
                        %419 = ascvec.shr %418, 8, %288 : <ui32> -> <ui32>
                        %420 = ascvec.shl %418, 24, %288 : <ui32> -> <ui32>
                        %421 = ascvec.or %419, %420, %288 : <ui32>, <ui32> -> <ui32>
                        %422 = ascvec.and %409, %335, %288 : <ui32>, <ui32> -> <ui32>
                        %423 = ascvec.cmp_ne_scalar %422, %c0_i32_13, %288 : <ui32>, i32 -> !ascvec.mask
                        %424 = ascvec.select %421, %418, %423 : <ui32>
                        %425 = ascvec.shr %424, 16, %288 : <ui32> -> <ui32>
                        %426 = ascvec.shl %424, 16, %288 : <ui32> -> <ui32>
                        %427 = ascvec.or %425, %426, %288 : <ui32>, <ui32> -> <ui32>
                        %428 = ascvec.and %409, %342, %288 : <ui32>, <ui32> -> <ui32>
                        %429 = ascvec.cmp_ne_scalar %428, %c0_i32_13, %288 : <ui32>, i32 -> !ascvec.mask
                        %430 = ascvec.select %427, %424, %429 : <ui32>
                        %431 = ascvec.and %430, %346, %288 : <ui32>, <ui32> -> <ui32>
                        %432 = ascvec.reinterpret %308 : <f32> -> <ui32>
                        %433 = ascvec.reinterpret %432 : <ui32> -> <f32>
                        %434 = ascvec.max_scalar %433, %cst, %288 : <f32>, f32
                        %435 = ascvec.add %433, %434, %288 : <f32>
                        %436 = ascvec.reinterpret %435 : <f32> -> <ui32>
                        %437 = ascvec.shr %436, 5, %288 : <ui32> -> <ui32>
                        %438 = ascvec.add %437, %347, %288 : <ui32>
                        %439 = ascvec.reinterpret %310 : <f32> -> <ui32>
                        %440 = ascvec.reinterpret %439 : <ui32> -> <f32>
                        %441 = ascvec.max_scalar %440, %cst, %288 : <f32>, f32
                        %442 = ascvec.add %440, %441, %288 : <f32>
                        %443 = ascvec.reinterpret %442 : <f32> -> <ui32>
                        %444 = ascvec.shr %443, 5, %288 : <ui32> -> <ui32>
                        %445 = ascvec.add %444, %375, %288 : <ui32>
                        %446 = ascvec.reinterpret %309 : <f32> -> <ui32>
                        %447 = ascvec.reinterpret %446 : <ui32> -> <f32>
                        %448 = ascvec.max_scalar %447, %cst, %288 : <f32>, f32
                        %449 = ascvec.add %447, %448, %288 : <f32>
                        %450 = ascvec.reinterpret %449 : <f32> -> <ui32>
                        %451 = ascvec.shr %450, 5, %288 : <ui32> -> <ui32>
                        %452 = ascvec.add %451, %403, %288 : <ui32>
                        %453 = ascvec.reinterpret %311 : <f32> -> <ui32>
                        %454 = ascvec.reinterpret %453 : <ui32> -> <f32>
                        %455 = ascvec.max_scalar %454, %cst, %288 : <f32>, f32
                        %456 = ascvec.add %454, %455, %288 : <f32>
                        %457 = ascvec.reinterpret %456 : <f32> -> <ui32>
                        %458 = ascvec.shr %457, 5, %288 : <ui32> -> <ui32>
                        %459 = ascvec.add %458, %431, %288 : <ui32>
                        %460 = ascvec.reinterpret %438 : <ui32> -> <ui16>
                        %461 = ascvec.reinterpret %445 : <ui32> -> <ui16>
                        %result0_15, %result1_16 = ascvec.deintlv %460, %461 : <ui16>, <ui16> -> <ui16>, <ui16>
                        %462 = ascvec.reinterpret %452 : <ui32> -> <ui16>
                        %463 = ascvec.reinterpret %459 : <ui32> -> <ui16>
                        %result0_17, %result1_18 = ascvec.deintlv %462, %463 : <ui16>, <ui16> -> <ui16>, <ui16>
                        %464 = ascvec.reinterpret %result1_16 : <ui16> -> <ui8>
                        %465 = ascvec.reinterpret %result1_18 : <ui16> -> <ui8>
                        %result0_19, %result1_20 = ascvec.deintlv %464, %465 : <ui8>, <ui8> -> <ui8>, <ui8>
                        %466 = ascvec.gather_reg %result0_19, %298 : <ui8>, <ui8> -> <ui8>
                        %467 = ascvec.add_scalar %466, %c60_i8, %290 : <ui8>, i8
                        %468 = ascvec.cmp_ne_scalar %467, %c-1_i8, %290 : <ui8>, i8 -> !ascvec.mask
                        %469 = ascvec.min_scalar %467, %c123_i8, %468 : <ui8>, i8
                        %470 = arith.muli %arg13, %c512_i32 : i32
                        %471 = arith.index_cast %470 : i32 to index
                        %result0_21, %result1_22 = ascvec.loadalign_deintlv %arg9[%471] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                        %472 = ascvec.and %result1_22, %292, %290 : <ui8>, <ui8> -> <ui8>
                        %473 = ascvec.or %469, %472, %290 : <ui8>, <ui8> -> <ui8>
                        %474 = ascvec.arange %c0_i16 : i16 -> <ui16>
                        %475 = arith.muli %arg13, %c16_i32 : i32
                        %476 = arith.trunci %475 : i32 to i16
                        %477 = ascvec.add_scalar %474, %476, %289 : <ui16>, i16
                        %478 = ascvec.gather %arg10[%c0_14], %477, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %479 = ascvec.pack %478 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                        %480 = ascvec.gather_reg %479, %293 : <ui8>, <ui8> -> <ui8>
                        %481 = ascvec.cmp_eq_scalar %480, %c-1_i8, %290 : <ui8>, i8 -> !ascvec.mask
                        %482 = ascvec.duplicate_scalar %c127_i8, %481 merge %473 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                        ascvec.storealign %arg11[%300], %482, %290 : memref<32768xui8, 1>, index, <ui8>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
                  }
                }
              }
            } else {
              %219 = arith.cmpi sge, %98, %c1024_i64 : i64
              %220 = arith.remsi %98, %c1024_i64 : i64
              %221 = arith.cmpi eq, %220, %c0_i64 : i64
              %222 = arith.select %219, %221, %false : i1
              scf.if %222 {
                %223 = arith.divsi %98, %c256_i64 : i64
                %224 = arith.addi %98, %c1023_i64 : i64
                %225 = arith.remsi %224, %c1024_i64 : i64
                %226 = arith.cmpi ne, %225, %c0_i64 : i64
                %227 = arith.cmpi slt, %225, %c0_i64 : i64
                %228 = arith.andi %226, %227 : i1
                %229 = arith.divsi %224, %c1024_i64 : i64
                %230 = arith.extui %228 : i1 to i64
                %231 = arith.subi %229, %230 : i64
                %232 = arith.trunci %1 : i64 to i32
                %233 = arith.extui %232 : i32 to i64
                %234 = arith.subi %1, %233 : i64
                %235 = arith.remsi %234, %c4294967296_i64 : i64
                %236 = arith.cmpi ne, %235, %c0_i64 : i64
                %237 = arith.cmpi slt, %235, %c0_i64 : i64
                %238 = arith.andi %236, %237 : i1
                %239 = arith.divsi %234, %c4294967296_i64 : i64
                %240 = arith.extui %238 : i1 to i64
                %241 = arith.subi %239, %240 : i64
                %242 = arith.trunci %241 : i64 to i32
                %243 = arith.divui %211, %c16_i64 : i64
                %244 = arith.trunci %243 : i64 to i32
                %245 = arith.extui %244 : i32 to i64
                %246 = arith.subi %243, %245 : i64
                %247 = arith.remsi %246, %c4294967296_i64 : i64
                %248 = arith.cmpi ne, %247, %c0_i64 : i64
                %249 = arith.cmpi slt, %247, %c0_i64 : i64
                %250 = arith.andi %248, %249 : i1
                %251 = arith.divsi %246, %c4294967296_i64 : i64
                %252 = arith.extui %250 : i1 to i64
                %253 = arith.subi %251, %252 : i64
                %254 = arith.trunci %253 : i64 to i32
                %255 = arith.trunci %3 : i64 to i32
                %256 = arith.extui %255 : i32 to i64
                %257 = arith.subi %3, %256 : i64
                %258 = arith.remsi %257, %c4294967296_i64 : i64
                %259 = arith.cmpi ne, %258, %c0_i64 : i64
                %260 = arith.cmpi slt, %258, %c0_i64 : i64
                %261 = arith.andi %259, %260 : i1
                %262 = arith.divsi %257, %c4294967296_i64 : i64
                %263 = arith.extui %261 : i1 to i64
                %264 = arith.subi %262, %263 : i64
                %265 = arith.trunci %264 : i64 to i32
                %266 = arith.divui %255, %c4_i32 : i32
                %267 = arith.remui %265, %c4_i32 : i32
                %268 = arith.muli %267, %c1073741824_i32 : i32
                %269 = arith.addi %266, %268 : i32
                %270 = arith.divui %265, %c4_i32 : i32
                %271 = arith.muli %231, %c64_i64 : i64
                %272 = arith.divui %243, %c16384_i64 : i64
                %273 = arith.addi %243, %271 : i64
                %274 = arith.subi %273, %c1_i64 : i64
                %275 = arith.divui %274, %c16384_i64 : i64
                %276 = arith.cmpi eq, %272, %275 : i64
                scf.if %276 {
                  %286 = arith.trunci %272 : i64 to i32
                  %287 = arith.extui %286 : i32 to i64
                  %288 = arith.subi %272, %287 : i64
                  %289 = arith.remsi %288, %c4294967296_i64 : i64
                  %290 = arith.cmpi ne, %289, %c0_i64 : i64
                  %291 = arith.cmpi slt, %289, %c0_i64 : i64
                  %292 = arith.andi %290, %291 : i1
                  %293 = arith.divsi %288, %c4294967296_i64 : i64
                  %294 = arith.extui %292 : i1 to i64
                  %295 = arith.subi %293, %294 : i64
                  %296 = arith.trunci %295 : i64 to i32
                  %297 = arith.addi %269, %286 : i32
                  %298 = arith.addi %270, %296 : i32
                  %299 = arith.cmpi ult, %297, %269 : i32
                  %300 = arith.extui %299 : i1 to i32
                  %301 = arith.addi %298, %300 : i32
                  %302 = arith.extui %297 : i32 to i64
                  %303 = arith.muli %302, %c3449720151_i64 : i64
                  %304 = arith.trunci %303 : i64 to i32
                  %305 = arith.extui %304 : i32 to i64
                  %306 = arith.subi %303, %305 : i64
                  %307 = arith.remsi %306, %c4294967296_i64 : i64
                  %308 = arith.cmpi ne, %307, %c0_i64 : i64
                  %309 = arith.cmpi slt, %307, %c0_i64 : i64
                  %310 = arith.andi %308, %309 : i1
                  %311 = arith.divsi %306, %c4294967296_i64 : i64
                  %312 = arith.extui %310 : i1 to i64
                  %313 = arith.subi %311, %312 : i64
                  %314 = arith.trunci %313 : i64 to i32
                  %315 = arith.xori %301, %242 : i32
                  %316 = arith.extui %315 : i32 to i64
                  %317 = arith.muli %316, %c3449720151_i64 : i64
                  %318 = arith.trunci %317 : i64 to i32
                  %319 = arith.extui %318 : i32 to i64
                  %320 = arith.subi %317, %319 : i64
                  %321 = arith.remsi %320, %c4294967296_i64 : i64
                  %322 = arith.cmpi ne, %321, %c0_i64 : i64
                  %323 = arith.cmpi slt, %321, %c0_i64 : i64
                  %324 = arith.andi %322, %323 : i1
                  %325 = arith.divsi %320, %c4294967296_i64 : i64
                  %326 = arith.extui %324 : i1 to i64
                  %327 = arith.subi %325, %326 : i64
                  %328 = arith.trunci %327 : i64 to i32
                  %329 = arith.addi %232, %c-1640531527_i32 : i32
                  %330 = arith.addi %242, %c-1150833019_i32 : i32
                  %331 = arith.xori %328, %304 : i32
                  %332 = arith.xori %331, %329 : i32
                  %333 = arith.extui %332 : i32 to i64
                  %334 = arith.muli %333, %c3528531795_i64 : i64
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
                  %346 = arith.xori %314, %232 : i32
                  %347 = arith.addi %232, %c1013904242_i32 : i32
                  %348 = arith.xori %318, %347 : i32
                  %349 = arith.addi %242, %c1993301258_i32 : i32
                  %350 = arith.xori %345, %349 : i32
                  ascsimt.vf(%244, %346, %330, %348, %350, %232, %242, %335, %alloc_8, %271 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
                    %c256_i64_10 = arith.constant 256 : i64
                    %c3_i64_11 = arith.constant 3 : i64
                    %c2_i64_12 = arith.constant 2 : i64
                    %c1_i64_13 = arith.constant 1 : i64
                    %c4_i64_14 = arith.constant 4 : i64
                    %c1_i32_15 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c3_i32 = arith.constant 3 : i32
                    %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %351 = ascsimt.thread_idx {ascsimt.scope = 44 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %352 = arith.extsi %351 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg17 = %352 to %arg16 step %c256_i64_10  : i64 {
                      %353 = arith.trunci %arg17 : i64 to i32
                      %354 = arith.addi %arg7, %353 : i32
                      %355 = arith.remui %354, %c16384_i32 : i32
                      %356 = arith.xori %355, %arg8 : i32
                      %357 = arith.muli %356, %c-766435501_i32 : i32
                      %358 = "ascsimt.bitcast"(%356) : (i32) -> ui32
                      %359 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %360 = ascsimt.math_binary %358, %359, "mulhi" : ui32, ui32 -> ui32
                      %361 = "ascsimt.bitcast"(%360) : (ui32) -> i32
                      %362 = arith.xori %361, %arg9 : i32
                      %363 = arith.muli %362, %c-845247145_i32 : i32
                      %364 = "ascsimt.bitcast"(%362) : (i32) -> ui32
                      %365 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %366 = ascsimt.math_binary %364, %365, "mulhi" : ui32, ui32 -> ui32
                      %367 = "ascsimt.bitcast"(%366) : (ui32) -> i32
                      %368 = arith.xori %367, %arg10 : i32
                      %369 = arith.xori %357, %arg11 : i32
                      %370:4 = scf.for %arg18 = %c3_i32 to %c10_i32 step %c1_i32_15 iter_args(%arg19 = %368, %arg20 = %363, %arg21 = %369, %arg22 = %arg14) -> (i32, i32, i32, i32)  : i32 {
                        %379 = arith.muli %arg19, %c-766435501_i32 : i32
                        %380 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %381 = ascsimt.math_binary %380, %359, "mulhi" : ui32, ui32 -> ui32
                        %382 = "ascsimt.bitcast"(%381) : (ui32) -> i32
                        %383 = arith.muli %arg21, %c-845247145_i32 : i32
                        %384 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %385 = ascsimt.math_binary %384, %365, "mulhi" : ui32, ui32 -> ui32
                        %386 = "ascsimt.bitcast"(%385) : (ui32) -> i32
                        %387 = arith.muli %arg18, %c-1640531527_i32_17 : i32
                        %388 = arith.addi %arg12, %387 : i32
                        %389 = arith.muli %arg18, %c-1150833019_i32_16 : i32
                        %390 = arith.addi %arg13, %389 : i32
                        %391 = arith.xori %386, %arg20 : i32
                        %392 = arith.xori %391, %388 : i32
                        %393 = arith.xori %382, %arg22 : i32
                        %394 = arith.xori %393, %390 : i32
                        scf.yield %392, %383, %394, %379 : i32, i32, i32, i32
                      }
                      %371 = arith.muli %arg17, %c4_i64_14 : i64
                      %372 = arith.index_cast %371 : i64 to index
                      ascsimt.store %arg15[%372], %370#0 : memref<8192xi32, 1>, i32
                      %373 = arith.addi %371, %c1_i64_13 : i64
                      %374 = arith.index_cast %373 : i64 to index
                      ascsimt.store %arg15[%374], %370#1 : memref<8192xi32, 1>, i32
                      %375 = arith.addi %371, %c2_i64_12 : i64
                      %376 = arith.index_cast %375 : i64 to index
                      ascsimt.store %arg15[%376], %370#2 : memref<8192xi32, 1>, i32
                      %377 = arith.addi %371, %c3_i64_11 : i64
                      %378 = arith.index_cast %377 : i64 to index
                      ascsimt.store %arg15[%378], %370#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%244, %254, %269, %270, %232, %242, %alloc_8, %271 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: memref<8192xi32, 1>, %arg14: i64):
                    %c256_i64_10 = arith.constant 256 : i64
                    %c3_i64_11 = arith.constant 3 : i64
                    %c2_i64_12 = arith.constant 2 : i64
                    %c1_i64_13 = arith.constant 1 : i64
                    %c4_i64_14 = arith.constant 4 : i64
                    %c1_i32_15 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c0_i32_16 = arith.constant 0 : i32
                    %c-1150833019_i32_17 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_18 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c262144_i32 = arith.constant 262144 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %286 = ascsimt.thread_idx {ascsimt.scope = 45 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %287 = arith.extsi %286 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg15 = %287 to %arg14 step %c256_i64_10  : i64 {
                      %288 = arith.trunci %arg15 : i64 to i32
                      %289 = arith.addi %arg7, %288 : i32
                      %290 = arith.cmpi ult, %289, %arg7 : i32
                      %291 = arith.extui %290 : i1 to i32
                      %292 = arith.addi %arg8, %291 : i32
                      %293 = arith.divui %289, %c16384_i32 : i32
                      %294 = arith.remui %292, %c16384_i32 : i32
                      %295 = arith.muli %294, %c262144_i32 : i32
                      %296 = arith.addi %293, %295 : i32
                      %297 = arith.divui %292, %c16384_i32 : i32
                      %298 = arith.addi %arg9, %296 : i32
                      %299 = arith.addi %arg10, %297 : i32
                      %300 = arith.cmpi ult, %298, %arg9 : i32
                      %301 = arith.extui %300 : i1 to i32
                      %302 = arith.addi %299, %301 : i32
                      %303 = arith.remui %289, %c16384_i32 : i32
                      %304:4 = scf.for %arg16 = %c0_i32_16 to %c10_i32 step %c1_i32_15 iter_args(%arg17 = %c0_i32_16, %arg18 = %303, %arg19 = %298, %arg20 = %302) -> (i32, i32, i32, i32)  : i32 {
                        %313 = arith.muli %arg17, %c-766435501_i32 : i32
                        %314 = "ascsimt.bitcast"(%arg17) : (i32) -> ui32
                        %315 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %316 = ascsimt.math_binary %314, %315, "mulhi" : ui32, ui32 -> ui32
                        %317 = "ascsimt.bitcast"(%316) : (ui32) -> i32
                        %318 = arith.muli %arg19, %c-845247145_i32 : i32
                        %319 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %320 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %321 = ascsimt.math_binary %319, %320, "mulhi" : ui32, ui32 -> ui32
                        %322 = "ascsimt.bitcast"(%321) : (ui32) -> i32
                        %323 = arith.muli %arg16, %c-1640531527_i32_18 : i32
                        %324 = arith.addi %arg11, %323 : i32
                        %325 = arith.muli %arg16, %c-1150833019_i32_17 : i32
                        %326 = arith.addi %arg12, %325 : i32
                        %327 = arith.xori %322, %arg18 : i32
                        %328 = arith.xori %327, %324 : i32
                        %329 = arith.xori %317, %arg20 : i32
                        %330 = arith.xori %329, %326 : i32
                        scf.yield %328, %318, %330, %313 : i32, i32, i32, i32
                      }
                      %305 = arith.muli %arg15, %c4_i64_14 : i64
                      %306 = arith.index_cast %305 : i64 to index
                      ascsimt.store %arg13[%306], %304#0 : memref<8192xi32, 1>, i32
                      %307 = arith.addi %305, %c1_i64_13 : i64
                      %308 = arith.index_cast %307 : i64 to index
                      ascsimt.store %arg13[%308], %304#1 : memref<8192xi32, 1>, i32
                      %309 = arith.addi %305, %c2_i64_12 : i64
                      %310 = arith.index_cast %309 : i64 to index
                      ascsimt.store %arg13[%310], %304#2 : memref<8192xi32, 1>, i32
                      %311 = arith.addi %305, %c3_i64_11 : i64
                      %312 = arith.index_cast %311 : i64 to index
                      ascsimt.store %arg13[%312], %304#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %277 = arith.trunci %31 : i64 to i32
                %278 = arith.divsi %277, %c2_i32 : i32
                %279 = arith.trunci %278 : i32 to i8
                %280 = arith.trunci %223 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %281 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%279, %alloc_6, %133, %281, %alloc_9, %280 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, memref<2048xui16, 1>, i32) {
                ^bb0(%arg7: i8, %arg8: memref<2048xui8, 1>, %arg9: memref<32768xbf16, 1>, %arg10: memref<?xui16, 1>, %arg11: memref<2048xui16, 1>, %arg12: i32):
                  %c16_10 = arith.constant 16 : index
                  %c4_i32_11 = arith.constant 4 : i32
                  %c0_i32_12 = arith.constant 0 : i32
                  %c3_i32 = arith.constant 3 : i32
                  %c2_i32_13 = arith.constant 2 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c1_i32_14 = arith.constant 1 : i32
                  %c0_15 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %286 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %287 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %288 = ascvec.duplicate_scalar %c32767_i16, %286 : i16 -> <ui16>
                  %289 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %290 = ascvec.shr %289, 4, %286 : <ui16> -> <ui16>
                  %291 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %292 = ascvec.duplicate_scalar %c1_i8, %287 : i8 -> <ui8>
                  %293 = ascvec.and %291, %292, %287 : <ui8>, <ui8> -> <ui8>
                  %294 = ascvec.cmp_lt_scalar %291, %c16_i8, %287 : <ui8>, i8 -> !ascvec.mask
                  %295 = ascvec.cmp_lt_scalar %293, %arg7, %287 : <ui8>, i8 -> !ascvec.mask
                  %296 = ascvec.mask_and %294, %295, %287
                  ascvec.store_unalign_begin %arg8 : memref<2048xui8, 1>
                  %mask_16, %remaining_out_17 = ascvec.update_mask %c1_i32_14 elem_bits 16 : !ascvec.mask
                  %297 = ascvec.duplicate_scalar %c-1_i16, %mask_16 : i16 -> <ui16>
                  %298 = scf.for %arg13 = %c0_i32_12 to %arg12 step %c4_i32_11 iter_args(%arg14 = %297) -> (!ascvec.vreg<ui16>)  : i32 {
                    %299 = ascvec.duplicate_scalar %c-1_i16, %286 : i16 -> <ui16>
                    %300 = ascvec.duplicate_scalar %c0_i16, %286 : i16 -> <ui16>
                    %301 = arith.muli %arg13, %c256_i32 : i32
                    %302 = arith.index_cast %301 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg9[%302] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %303 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %304 = ascvec.and %303, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %305 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %306 = ascvec.and %305, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %307 = ascvec.max %304, %306, %286 : <ui16>
                    %308 = ascvec.reduce_max_datablock %307, %286 : <ui16>
                    %309 = ascvec.shr %308, 7, %286 : <ui16> -> <ui16>
                    %310 = ascvec.cmp_ge_scalar %309, %c255_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %311 = ascvec.max_scalar %309, %c15_i16, %286 : <ui16>, i16
                    %312 = ascvec.add_scalar %311, %c-15_i16, %286 : <ui16>, i16
                    %313 = ascvec.duplicate_scalar %c255_i16, %310 merge %312 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %314 = ascvec.duplicate_scalar %c254_i16, %286 : i16 -> <ui16>
                    %315 = ascvec.sub %314, %313, %286 : <ui16>
                    %316 = ascvec.cmp_ne_scalar %313, %c0_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %317 = ascvec.shl %315, 7, %316 : <ui16> -> <ui16>
                    %318 = ascvec.max %300, %313, %mask : <ui16>
                    %319 = ascvec.gather_reg %317, %290 : <ui16>, <ui16> -> <ui16>
                    %320 = ascvec.reinterpret %319 : <ui16> -> <bf16>
                    %321 = ascvec.mul %result0, %320, %286 : <bf16>
                    %322 = ascvec.mul %result1, %320, %286 : <bf16>
                    ascvec.storealign_intlv %arg9[%302], %321, %322 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %323 = ascvec.reinterpret %321 : <bf16> -> <ui16>
                    %324 = ascvec.and %323, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %325 = ascvec.reinterpret %322 : <bf16> -> <ui16>
                    %326 = ascvec.and %325, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %327 = ascvec.min %324, %326, %286 : <ui16>
                    %328 = ascvec.min %299, %327, %286 : <ui16>
                    %329 = arith.muli %arg13, %c16_i32 : i32
                    %330 = arith.index_cast %329 : i32 to index
                    ascvec.storealign %arg10[%330], %313, %mask : memref<?xui16, 1>, index, <ui16>
                    %331 = ascvec.reinterpret %313 : <ui16> -> <ui8>
                    %332 = ascvec.squeeze_store_reg %331, %296 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_15], %332 : memref<2048xui8, 1>, <ui8>
                    %333 = arith.addi %arg13, %c1_i32_14 : i32
                    %334 = arith.muli %333, %c256_i32 : i32
                    %335 = arith.index_cast %334 : i32 to index
                    %result0_20, %result1_21 = ascvec.loadalign_deintlv %arg9[%335] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %336 = ascvec.reinterpret %result0_20 : <bf16> -> <ui16>
                    %337 = ascvec.and %336, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %338 = ascvec.reinterpret %result1_21 : <bf16> -> <ui16>
                    %339 = ascvec.and %338, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %340 = ascvec.max %337, %339, %286 : <ui16>
                    %341 = ascvec.reduce_max_datablock %340, %286 : <ui16>
                    %342 = ascvec.shr %341, 7, %286 : <ui16> -> <ui16>
                    %343 = ascvec.cmp_ge_scalar %342, %c255_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %344 = ascvec.max_scalar %342, %c15_i16, %286 : <ui16>, i16
                    %345 = ascvec.add_scalar %344, %c-15_i16, %286 : <ui16>, i16
                    %346 = ascvec.duplicate_scalar %c255_i16, %343 merge %345 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %347 = ascvec.sub %314, %346, %286 : <ui16>
                    %348 = ascvec.cmp_ne_scalar %346, %c0_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %349 = ascvec.shl %347, 7, %348 : <ui16> -> <ui16>
                    %350 = ascvec.max %318, %346, %mask : <ui16>
                    %351 = ascvec.gather_reg %349, %290 : <ui16>, <ui16> -> <ui16>
                    %352 = ascvec.reinterpret %351 : <ui16> -> <bf16>
                    %353 = ascvec.mul %result0_20, %352, %286 : <bf16>
                    %354 = ascvec.mul %result1_21, %352, %286 : <bf16>
                    ascvec.storealign_intlv %arg9[%335], %353, %354 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %355 = ascvec.reinterpret %353 : <bf16> -> <ui16>
                    %356 = ascvec.and %355, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %357 = ascvec.reinterpret %354 : <bf16> -> <ui16>
                    %358 = ascvec.and %357, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %359 = ascvec.min %356, %358, %286 : <ui16>
                    %360 = ascvec.min %328, %359, %286 : <ui16>
                    %361 = arith.muli %333, %c16_i32 : i32
                    %362 = arith.index_cast %361 : i32 to index
                    ascvec.storealign %arg10[%362], %346, %mask : memref<?xui16, 1>, index, <ui16>
                    %363 = ascvec.reinterpret %346 : <ui16> -> <ui8>
                    %364 = ascvec.squeeze_store_reg %363, %296 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_15], %364 : memref<2048xui8, 1>, <ui8>
                    %365 = arith.addi %arg13, %c2_i32_13 : i32
                    %366 = arith.muli %365, %c256_i32 : i32
                    %367 = arith.index_cast %366 : i32 to index
                    %result0_22, %result1_23 = ascvec.loadalign_deintlv %arg9[%367] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %368 = ascvec.reinterpret %result0_22 : <bf16> -> <ui16>
                    %369 = ascvec.and %368, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %370 = ascvec.reinterpret %result1_23 : <bf16> -> <ui16>
                    %371 = ascvec.and %370, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %372 = ascvec.max %369, %371, %286 : <ui16>
                    %373 = ascvec.reduce_max_datablock %372, %286 : <ui16>
                    %374 = ascvec.shr %373, 7, %286 : <ui16> -> <ui16>
                    %375 = ascvec.cmp_ge_scalar %374, %c255_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %376 = ascvec.max_scalar %374, %c15_i16, %286 : <ui16>, i16
                    %377 = ascvec.add_scalar %376, %c-15_i16, %286 : <ui16>, i16
                    %378 = ascvec.duplicate_scalar %c255_i16, %375 merge %377 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %379 = ascvec.sub %314, %378, %286 : <ui16>
                    %380 = ascvec.cmp_ne_scalar %378, %c0_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %381 = ascvec.shl %379, 7, %380 : <ui16> -> <ui16>
                    %382 = ascvec.max %350, %378, %mask : <ui16>
                    %383 = ascvec.gather_reg %381, %290 : <ui16>, <ui16> -> <ui16>
                    %384 = ascvec.reinterpret %383 : <ui16> -> <bf16>
                    %385 = ascvec.mul %result0_22, %384, %286 : <bf16>
                    %386 = ascvec.mul %result1_23, %384, %286 : <bf16>
                    ascvec.storealign_intlv %arg9[%367], %385, %386 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %387 = ascvec.reinterpret %385 : <bf16> -> <ui16>
                    %388 = ascvec.and %387, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %389 = ascvec.reinterpret %386 : <bf16> -> <ui16>
                    %390 = ascvec.and %389, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %391 = ascvec.min %388, %390, %286 : <ui16>
                    %392 = ascvec.min %360, %391, %286 : <ui16>
                    %393 = arith.muli %365, %c16_i32 : i32
                    %394 = arith.index_cast %393 : i32 to index
                    ascvec.storealign %arg10[%394], %378, %mask : memref<?xui16, 1>, index, <ui16>
                    %395 = ascvec.reinterpret %378 : <ui16> -> <ui8>
                    %396 = ascvec.squeeze_store_reg %395, %296 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_15], %396 : memref<2048xui8, 1>, <ui8>
                    %397 = arith.addi %arg13, %c3_i32 : i32
                    %398 = arith.muli %397, %c256_i32 : i32
                    %399 = arith.index_cast %398 : i32 to index
                    %result0_24, %result1_25 = ascvec.loadalign_deintlv %arg9[%399] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %400 = ascvec.reinterpret %result0_24 : <bf16> -> <ui16>
                    %401 = ascvec.and %400, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %402 = ascvec.reinterpret %result1_25 : <bf16> -> <ui16>
                    %403 = ascvec.and %402, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %404 = ascvec.max %401, %403, %286 : <ui16>
                    %405 = ascvec.reduce_max_datablock %404, %286 : <ui16>
                    %406 = ascvec.shr %405, 7, %286 : <ui16> -> <ui16>
                    %407 = ascvec.cmp_ge_scalar %406, %c255_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %408 = ascvec.max_scalar %406, %c15_i16, %286 : <ui16>, i16
                    %409 = ascvec.add_scalar %408, %c-15_i16, %286 : <ui16>, i16
                    %410 = ascvec.duplicate_scalar %c255_i16, %407 merge %409 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %411 = ascvec.sub %314, %410, %286 : <ui16>
                    %412 = ascvec.cmp_ne_scalar %410, %c0_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %413 = ascvec.shl %411, 7, %412 : <ui16> -> <ui16>
                    %414 = ascvec.max %382, %410, %mask : <ui16>
                    %415 = ascvec.gather_reg %413, %290 : <ui16>, <ui16> -> <ui16>
                    %416 = ascvec.reinterpret %415 : <ui16> -> <bf16>
                    %417 = ascvec.mul %result0_24, %416, %286 : <bf16>
                    %418 = ascvec.mul %result1_25, %416, %286 : <bf16>
                    ascvec.storealign_intlv %arg9[%399], %417, %418 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %419 = ascvec.reinterpret %417 : <bf16> -> <ui16>
                    %420 = ascvec.and %419, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %421 = ascvec.reinterpret %418 : <bf16> -> <ui16>
                    %422 = ascvec.and %421, %288, %286 : <ui16>, <ui16> -> <ui16>
                    %423 = ascvec.min %420, %422, %286 : <ui16>
                    %424 = ascvec.min %392, %423, %286 : <ui16>
                    %425 = arith.muli %397, %c16_i32 : i32
                    %426 = arith.index_cast %425 : i32 to index
                    ascvec.storealign %arg10[%426], %410, %mask : memref<?xui16, 1>, index, <ui16>
                    %427 = ascvec.reinterpret %410 : <ui16> -> <ui8>
                    %428 = ascvec.squeeze_store_reg %427, %296 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_15], %428 : memref<2048xui8, 1>, <ui8>
                    %429 = ascvec.reduce_min %424, %286 : <ui16>
                    %430 = ascvec.reduce_max %414, %mask : <ui16>
                    %431 = ascvec.cmp_eq_scalar %430, %c255_i16, %286 : <ui16>, i16 -> !ascvec.mask
                    %432 = ascvec.duplicate_scalar %c0_i16, %431 merge %429 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.storealign_1st %arg11[%330], %432 : memref<2048xui16, 1>, index, <ui16>
                    %433 = ascvec.min %arg14, %432, %mask_16 : <ui16>
                    scf.yield %433 : !ascvec.vreg<ui16>
                  }
                  %c0_18 = arith.constant 0 : index
                  %c1_19 = arith.constant 1 : index
                  scf.for %arg13 = %c0_18 to %c1_19 step %c1_19 {
                    ascvec.store_unalign_post %arg8[%c0_15] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg11[%c16_10], %298 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %282 = memref.load %alloc_9[%c16] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %283 = cannir.scalar_storage_cast %282 : ui16 -> i16
                %284 = arith.extui %283 : i16 to i64
                %285 = arith.cmpi sge, %284, %c14464_i64 : i64
                scf.if %285 {
                  ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %286 = ascscalar.bitcast_memref %133 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%133, %alloc_8, %286, %alloc_5, %280 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg7: memref<32768xbf16, 1>, %arg8: memref<8192xi32, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<32768xui8, 1>, %arg11: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_10 = arith.constant 1 : i32
                    %c0_i32_11 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c64_i32 = arith.constant 64 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c1_i16 = arith.constant 1 : i16
                    %c0_i16 = arith.constant 0 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %287 = ascvec.create_mask : !ascvec.mask
                    %288 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %289 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %290 = ascvec.duplicate_scalar %c255_i16, %288 : i16 -> <ui16>
                    %291 = ascvec.duplicate_scalar %c-128_i8, %289 : i8 -> <ui8>
                    %292 = ascvec.arange %c0_i16 : i16 -> <ui16>
                    %293 = ascvec.duplicate_scalar %c1_i16, %288 : i16 -> <ui16>
                    %294 = ascvec.xor %292, %293, %288 : <ui16>, <ui16> -> <ui16>
                    scf.for %arg12 = %c0_i32_11 to %arg11 step %c1_i32_10  : i32 {
                      %295 = arith.muli %arg12, %c256_i32 : i32
                      %296 = arith.index_cast %295 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg7[%296] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %297 = arith.muli %arg12, %c64_i32 : i32
                      %298 = arith.index_cast %297 : i32 to index
                      %299 = ascvec.loadalign %arg8[%298] : memref<8192xi32, 1>, index -> <i32>
                      %300 = ascvec.reinterpret %299 : <i32> -> <ui16>
                      %301 = ascvec.reinterpret %300 : <ui16> -> <ui32>
                      %302 = ascvec.shl %301, 8, %287 : <ui32> -> <ui32>
                      %303 = ascvec.shr %301, 24, %287 : <ui32> -> <ui32>
                      %304 = ascvec.or %302, %303, %287 : <ui32>, <ui32> -> <ui32>
                      %305 = ascvec.reinterpret %304 : <ui32> -> <ui16>
                      %306 = ascvec.gather_reg %300, %294 : <ui16>, <ui16> -> <ui16>
                      %307 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %308 = ascvec.shr %306, 11, %288 : <ui16> -> <ui16>
                      %309 = ascvec.add %307, %308, %288 : <ui16>
                      %310 = ascvec.shr %309, 5, %288 : <ui16> -> <ui16>
                      %311 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %312 = ascvec.shr %305, 11, %288 : <ui16> -> <ui16>
                      %313 = ascvec.add %311, %312, %288 : <ui16>
                      %314 = ascvec.shr %313, 5, %288 : <ui16> -> <ui16>
                      %315 = ascvec.and %310, %290, %288 : <ui16>, <ui16> -> <ui16>
                      %316 = ascvec.shl %314, 8, %288 : <ui16> -> <ui16>
                      %317 = ascvec.or %315, %316, %288 : <ui16>, <ui16> -> <ui16>
                      %318 = ascvec.reinterpret %317 : <ui16> -> <ui8>
                      %319 = ascvec.add_scalar %318, %c64_i8, %289 : <ui8>, i8
                      %320 = ascvec.min_scalar %319, %c123_i8, %289 : <ui8>, i8
                      %321 = arith.muli %arg12, %c512_i32 : i32
                      %322 = arith.index_cast %321 : i32 to index
                      %result0_12, %result1_13 = ascvec.loadalign_deintlv %arg9[%322] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %323 = ascvec.and %result1_13, %291, %289 : <ui8>, <ui8> -> <ui8>
                      %324 = ascvec.or %320, %323, %289 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg10[%296], %324, %289 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
                } else {
                  scf.for %arg7 = %c0_i64 to %223 step %c16_i64  : i64 {
                    %286 = arith.addi %arg7, %c16_i64 : i64
                    %287 = arith.cmpi sgt, %286, %223 : i64
                    %288 = arith.select %287, %223, %286 : i64
                    %289 = arith.muli %arg7, %c16_i64 : i64
                    ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %290 = arith.index_cast %289 : i64 to index
                    %291 = memref.load %alloc_9[%290] : memref<2048xui16, 1>
                    ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %292 = cannir.scalar_storage_cast %291 : ui16 -> i16
                    %293 = arith.extui %292 : i16 to i64
                    %294 = arith.addi %arg7, %c4_i64 : i64
                    %295 = scf.for %arg8 = %294 to %288 step %c4_i64 iter_args(%arg9 = %293) -> (i64)  : i64 {
                      %297 = arith.muli %arg8, %c16_i64 : i64
                      ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %298 = arith.index_cast %297 : i64 to index
                      %299 = memref.load %alloc_9[%298] : memref<2048xui16, 1>
                      ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %300 = cannir.scalar_storage_cast %299 : ui16 -> i16
                      %301 = arith.extui %300 : i16 to i64
                      %302 = arith.cmpi slt, %301, %arg9 : i64
                      %303 = arith.select %302, %301, %arg9 : i64
                      scf.yield %303 : i64
                    }
                    %296 = arith.cmpi sge, %295, %c14464_i64 : i64
                    scf.if %296 {
                      %297 = arith.trunci %arg7 : i64 to i32
                      %298 = arith.trunci %288 : i64 to i32
                      ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %299 = ascscalar.bitcast_memref %133 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                      ascvec.vf(%133, %alloc_8, %299, %alloc_5, %297, %298 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                      ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<8192xi32, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32, %arg13: i32):
                        %c512_i32 = arith.constant 512 : i32
                        %c1_i32_10 = arith.constant 1 : i32
                        %c123_i8 = arith.constant 123 : i8
                        %c64_i8 = arith.constant 64 : i8
                        %c64_i32 = arith.constant 64 : i32
                        %c256_i32 = arith.constant 256 : i32
                        %c1_i16 = arith.constant 1 : i16
                        %c0_i16 = arith.constant 0 : i16
                        %c-128_i8 = arith.constant -128 : i8
                        %c255_i16 = arith.constant 255 : i16
                        %300 = ascvec.create_mask : !ascvec.mask
                        %301 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                        %302 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                        %303 = ascvec.duplicate_scalar %c255_i16, %301 : i16 -> <ui16>
                        %304 = ascvec.duplicate_scalar %c-128_i8, %302 : i8 -> <ui8>
                        %305 = ascvec.arange %c0_i16 : i16 -> <ui16>
                        %306 = ascvec.duplicate_scalar %c1_i16, %301 : i16 -> <ui16>
                        %307 = ascvec.xor %305, %306, %301 : <ui16>, <ui16> -> <ui16>
                        scf.for %arg14 = %arg12 to %arg13 step %c1_i32_10  : i32 {
                          %308 = arith.muli %arg14, %c256_i32 : i32
                          %309 = arith.index_cast %308 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg8[%309] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                          %310 = arith.muli %arg14, %c64_i32 : i32
                          %311 = arith.index_cast %310 : i32 to index
                          %312 = ascvec.loadalign %arg9[%311] : memref<8192xi32, 1>, index -> <i32>
                          %313 = ascvec.reinterpret %312 : <i32> -> <ui16>
                          %314 = ascvec.reinterpret %313 : <ui16> -> <ui32>
                          %315 = ascvec.shl %314, 8, %300 : <ui32> -> <ui32>
                          %316 = ascvec.shr %314, 24, %300 : <ui32> -> <ui32>
                          %317 = ascvec.or %315, %316, %300 : <ui32>, <ui32> -> <ui32>
                          %318 = ascvec.reinterpret %317 : <ui32> -> <ui16>
                          %319 = ascvec.gather_reg %313, %307 : <ui16>, <ui16> -> <ui16>
                          %320 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                          %321 = ascvec.shr %319, 11, %301 : <ui16> -> <ui16>
                          %322 = ascvec.add %320, %321, %301 : <ui16>
                          %323 = ascvec.shr %322, 5, %301 : <ui16> -> <ui16>
                          %324 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                          %325 = ascvec.shr %318, 11, %301 : <ui16> -> <ui16>
                          %326 = ascvec.add %324, %325, %301 : <ui16>
                          %327 = ascvec.shr %326, 5, %301 : <ui16> -> <ui16>
                          %328 = ascvec.and %323, %303, %301 : <ui16>, <ui16> -> <ui16>
                          %329 = ascvec.shl %327, 8, %301 : <ui16> -> <ui16>
                          %330 = ascvec.or %328, %329, %301 : <ui16>, <ui16> -> <ui16>
                          %331 = ascvec.reinterpret %330 : <ui16> -> <ui8>
                          %332 = ascvec.add_scalar %331, %c64_i8, %302 : <ui8>, i8
                          %333 = ascvec.min_scalar %332, %c123_i8, %302 : <ui8>, i8
                          %334 = arith.muli %arg14, %c512_i32 : i32
                          %335 = arith.index_cast %334 : i32 to index
                          %result0_11, %result1_12 = ascvec.loadalign_deintlv %arg10[%335] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                          %336 = ascvec.and %result1_12, %304, %302 : <ui8>, <ui8> -> <ui8>
                          %337 = ascvec.or %333, %336, %302 : <ui8>, <ui8> -> <ui8>
                          ascvec.storealign %arg11[%309], %337, %302 : memref<32768xui8, 1>, index, <ui8>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
                    } else {
                      %297 = arith.cmpi sge, %295, %c13568_i64 : i64
                      scf.if %297 {
                        %298 = arith.trunci %arg7 : i64 to i32
                        %299 = arith.trunci %288 : i64 to i32
                        ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %300 = ascscalar.bitcast_memref %133 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%133, %alloc_8, %300, %alloc_5, %298, %299 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<8192xi32, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32, %arg13: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_10 = arith.constant 1 : i32
                          %c123_i8 = arith.constant 123 : i8
                          %c64_i8 = arith.constant 64 : i8
                          %c448_i16 = arith.constant 448 : i16
                          %c14464_i16 = arith.constant 14464 : i16
                          %c64_i32 = arith.constant 64 : i32
                          %c256_i32 = arith.constant 256 : i32
                          %cst = arith.constant 2.684350e+08 : bf16
                          %c32767_i16 = arith.constant 32767 : i16
                          %c1_i16 = arith.constant 1 : i16
                          %c0_i16 = arith.constant 0 : i16
                          %c-128_i8 = arith.constant -128 : i8
                          %c255_i16 = arith.constant 255 : i16
                          %301 = ascvec.create_mask : !ascvec.mask
                          %302 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %303 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %304 = ascvec.duplicate_scalar %c255_i16, %302 : i16 -> <ui16>
                          %305 = ascvec.duplicate_scalar %c-128_i8, %303 : i8 -> <ui8>
                          %306 = ascvec.arange %c0_i16 : i16 -> <ui16>
                          %307 = ascvec.duplicate_scalar %c1_i16, %302 : i16 -> <ui16>
                          %308 = ascvec.xor %306, %307, %302 : <ui16>, <ui16> -> <ui16>
                          %309 = ascvec.duplicate_scalar %c32767_i16, %302 : i16 -> <ui16>
                          %310 = ascvec.duplicate_scalar %cst, %302 : bf16 -> <bf16>
                          scf.for %arg14 = %arg12 to %arg13 step %c1_i32_10  : i32 {
                            %311 = arith.muli %arg14, %c256_i32 : i32
                            %312 = arith.index_cast %311 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg8[%312] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %313 = arith.muli %arg14, %c64_i32 : i32
                            %314 = arith.index_cast %313 : i32 to index
                            %315 = ascvec.loadalign %arg9[%314] : memref<8192xi32, 1>, index -> <i32>
                            %316 = ascvec.reinterpret %315 : <i32> -> <ui16>
                            %317 = ascvec.reinterpret %316 : <ui16> -> <ui32>
                            %318 = ascvec.shl %317, 8, %301 : <ui32> -> <ui32>
                            %319 = ascvec.shr %317, 24, %301 : <ui32> -> <ui32>
                            %320 = ascvec.or %318, %319, %301 : <ui32>, <ui32> -> <ui32>
                            %321 = ascvec.reinterpret %320 : <ui32> -> <ui16>
                            %322 = ascvec.gather_reg %316, %308 : <ui16>, <ui16> -> <ui16>
                            %323 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %324 = ascvec.and %323, %309, %302 : <ui16>, <ui16> -> <ui16>
                            %325 = ascvec.cmp_lt_scalar %324, %c14464_i16, %302 : <ui16>, i16 -> !ascvec.mask
                            %326 = ascvec.shr %322, 11, %302 : <ui16> -> <ui16>
                            %327 = ascvec.add %324, %326, %302 : <ui16>
                            %328 = ascvec.shr %327, 5, %302 : <ui16> -> <ui16>
                            %329 = ascvec.reinterpret %324 : <ui16> -> <bf16>
                            %330 = ascvec.mul %329, %310, %325 : <bf16>
                            %331 = ascvec.cast %330, %302 {rounding = 4 : i32} : <bf16> -> <f16>
                            %332 = ascvec.cast %331, %302 {rounding = 4 : i32} : <f16> -> <i16>
                            %333 = ascvec.reinterpret %332 : <i16> -> <ui16>
                            %334 = ascvec.shr %322, 4, %302 : <ui16> -> <ui16>
                            %335 = ascvec.add %333, %334, %302 : <ui16>
                            %336 = ascvec.shr %335, 12, %302 : <ui16> -> <ui16>
                            %337 = ascvec.add_scalar %336, %c448_i16, %302 : <ui16>, i16
                            %338 = ascvec.select %337, %328, %325 : <ui16>
                            %339 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %340 = ascvec.and %339, %309, %302 : <ui16>, <ui16> -> <ui16>
                            %341 = ascvec.cmp_lt_scalar %340, %c14464_i16, %302 : <ui16>, i16 -> !ascvec.mask
                            %342 = ascvec.shr %321, 11, %302 : <ui16> -> <ui16>
                            %343 = ascvec.add %340, %342, %302 : <ui16>
                            %344 = ascvec.shr %343, 5, %302 : <ui16> -> <ui16>
                            %345 = ascvec.reinterpret %340 : <ui16> -> <bf16>
                            %346 = ascvec.mul %345, %310, %341 : <bf16>
                            %347 = ascvec.cast %346, %302 {rounding = 4 : i32} : <bf16> -> <f16>
                            %348 = ascvec.cast %347, %302 {rounding = 4 : i32} : <f16> -> <i16>
                            %349 = ascvec.reinterpret %348 : <i16> -> <ui16>
                            %350 = ascvec.shr %321, 4, %302 : <ui16> -> <ui16>
                            %351 = ascvec.add %349, %350, %302 : <ui16>
                            %352 = ascvec.shr %351, 12, %302 : <ui16> -> <ui16>
                            %353 = ascvec.add_scalar %352, %c448_i16, %302 : <ui16>, i16
                            %354 = ascvec.select %353, %344, %341 : <ui16>
                            %355 = ascvec.and %338, %304, %302 : <ui16>, <ui16> -> <ui16>
                            %356 = ascvec.shl %354, 8, %302 : <ui16> -> <ui16>
                            %357 = ascvec.or %355, %356, %302 : <ui16>, <ui16> -> <ui16>
                            %358 = ascvec.reinterpret %357 : <ui16> -> <ui8>
                            %359 = ascvec.add_scalar %358, %c64_i8, %303 : <ui8>, i8
                            %360 = ascvec.min_scalar %359, %c123_i8, %303 : <ui8>, i8
                            %361 = arith.muli %arg14, %c512_i32 : i32
                            %362 = arith.index_cast %361 : i32 to index
                            %result0_11, %result1_12 = ascvec.loadalign_deintlv %arg10[%362] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %363 = ascvec.and %result1_12, %305, %303 : <ui8>, <ui8> -> <ui8>
                            %364 = ascvec.or %360, %363, %303 : <ui8>, <ui8> -> <ui8>
                            ascvec.storealign %arg11[%312], %364, %303 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
                      } else {
                        %298 = arith.trunci %arg7 : i64 to i32
                        %299 = arith.trunci %288 : i64 to i32
                        ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %300 = ascscalar.bitcast_memref %133 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%133, %alloc_8, %300, %281, %alloc_5, %298, %299 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<8192xi32, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<?xui16, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32, %arg14: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_10 = arith.constant 1 : i32
                          %c127_i8 = arith.constant 127 : i8
                          %c16_i32 = arith.constant 16 : i32
                          %c0_i16 = arith.constant 0 : i16
                          %c123_i8 = arith.constant 123 : i8
                          %c-1_i8 = arith.constant -1 : i8
                          %c60_i8 = arith.constant 60 : i8
                          %cst = arith.constant 6.10351526E-5 : f32
                          %c64_i32 = arith.constant 64 : i32
                          %c32767_i16 = arith.constant 32767 : i16
                          %c0_11 = arith.constant 0 : index
                          %c256_i32 = arith.constant 256 : i32
                          %c65535_i32 = arith.constant 65535 : i32
                          %c3_i8 = arith.constant 3 : i8
                          %c-128_i8 = arith.constant -128 : i8
                          %c0_i8 = arith.constant 0 : i8
                          %c8_i32 = arith.constant 8 : i32
                          %301 = ascvec.create_mask : !ascvec.mask
                          %302 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %303 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                          %304 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %305 = ascvec.duplicate_scalar %c-128_i8, %303 : i8 -> <ui8>
                          %306 = ascvec.shr %304, 5, %303 : <ui8> -> <ui8>
                          %307 = ascvec.duplicate_scalar %c3_i8, %303 : i8 -> <ui8>
                          %308 = ascvec.and %304, %307, %303 : <ui8>, <ui8> -> <ui8>
                          %309 = ascvec.shl %308, 6, %303 : <ui8> -> <ui8>
                          %310 = ascvec.shr %304, 2, %303 : <ui8> -> <ui8>
                          %311 = ascvec.or %309, %310, %303 : <ui8>, <ui8> -> <ui8>
                          %312 = ascvec.duplicate_scalar %c65535_i32, %301 : i32 -> <ui32>
                          scf.for %arg15 = %arg13 to %arg14 step %c1_i32_10  : i32 {
                            %313 = arith.muli %arg15, %c256_i32 : i32
                            %314 = arith.index_cast %313 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg8[%314] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %315 = ascvec.duplicate_scalar %c32767_i16, %302 : i16 -> <ui16>
                            %316 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %317 = ascvec.and %316, %315, %302 : <ui16>, <ui16> -> <ui16>
                            %318 = ascvec.reinterpret %317 : <ui16> -> <bf16>
                            %319 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %320 = ascvec.and %319, %315, %302 : <ui16>, <ui16> -> <ui16>
                            %321 = ascvec.reinterpret %320 : <ui16> -> <bf16>
                            %322 = ascvec.cast %318, %302 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %323 = ascvec.cast %318, %302 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %324 = ascvec.cast %321, %302 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %325 = ascvec.cast %321, %302 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %326 = arith.muli %arg15, %c64_i32 : i32
                            %327 = arith.index_cast %326 : i32 to index
                            %328 = ascvec.loadalign %arg9[%327] : memref<8192xi32, 1>, index -> <i32>
                            %329 = ascvec.reinterpret %328 : <i32> -> <ui32>
                            %330 = ascvec.shl %329, 8, %301 : <ui32> -> <ui32>
                            %331 = ascvec.shr %329, 24, %301 : <ui32> -> <ui32>
                            %332 = ascvec.or %330, %331, %301 : <ui32>, <ui32> -> <ui32>
                            %333 = ascvec.shr %329, 16, %301 : <ui32> -> <ui32>
                            %334 = ascvec.and %332, %312, %301 : <ui32>, <ui32> -> <ui32>
                            %335 = ascvec.and %329, %312, %301 : <ui32>, <ui32> -> <ui32>
                            %336 = ascvec.shr %332, 16, %301 : <ui32> -> <ui32>
                            %337 = ascvec.reinterpret %322 : <f32> -> <ui32>
                            %338 = ascvec.reinterpret %337 : <ui32> -> <f32>
                            %339 = ascvec.max_scalar %338, %cst, %301 : <f32>, f32
                            %340 = ascvec.add %338, %339, %301 : <f32>
                            %341 = ascvec.reinterpret %340 : <f32> -> <ui32>
                            %342 = ascvec.shr %341, 5, %301 : <ui32> -> <ui32>
                            %343 = ascvec.add %342, %333, %301 : <ui32>
                            %344 = ascvec.reinterpret %324 : <f32> -> <ui32>
                            %345 = ascvec.reinterpret %344 : <ui32> -> <f32>
                            %346 = ascvec.max_scalar %345, %cst, %301 : <f32>, f32
                            %347 = ascvec.add %345, %346, %301 : <f32>
                            %348 = ascvec.reinterpret %347 : <f32> -> <ui32>
                            %349 = ascvec.shr %348, 5, %301 : <ui32> -> <ui32>
                            %350 = ascvec.add %349, %334, %301 : <ui32>
                            %351 = ascvec.reinterpret %323 : <f32> -> <ui32>
                            %352 = ascvec.reinterpret %351 : <ui32> -> <f32>
                            %353 = ascvec.max_scalar %352, %cst, %301 : <f32>, f32
                            %354 = ascvec.add %352, %353, %301 : <f32>
                            %355 = ascvec.reinterpret %354 : <f32> -> <ui32>
                            %356 = ascvec.shr %355, 5, %301 : <ui32> -> <ui32>
                            %357 = ascvec.add %356, %335, %301 : <ui32>
                            %358 = ascvec.reinterpret %325 : <f32> -> <ui32>
                            %359 = ascvec.reinterpret %358 : <ui32> -> <f32>
                            %360 = ascvec.max_scalar %359, %cst, %301 : <f32>, f32
                            %361 = ascvec.add %359, %360, %301 : <f32>
                            %362 = ascvec.reinterpret %361 : <f32> -> <ui32>
                            %363 = ascvec.shr %362, 5, %301 : <ui32> -> <ui32>
                            %364 = ascvec.add %363, %336, %301 : <ui32>
                            %365 = ascvec.reinterpret %343 : <ui32> -> <ui16>
                            %366 = ascvec.reinterpret %350 : <ui32> -> <ui16>
                            %result0_12, %result1_13 = ascvec.deintlv %365, %366 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %367 = ascvec.reinterpret %357 : <ui32> -> <ui16>
                            %368 = ascvec.reinterpret %364 : <ui32> -> <ui16>
                            %result0_14, %result1_15 = ascvec.deintlv %367, %368 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %369 = ascvec.reinterpret %result1_13 : <ui16> -> <ui8>
                            %370 = ascvec.reinterpret %result1_15 : <ui16> -> <ui8>
                            %result0_16, %result1_17 = ascvec.deintlv %369, %370 : <ui8>, <ui8> -> <ui8>, <ui8>
                            %371 = ascvec.gather_reg %result0_16, %311 : <ui8>, <ui8> -> <ui8>
                            %372 = ascvec.add_scalar %371, %c60_i8, %303 : <ui8>, i8
                            %373 = ascvec.cmp_ne_scalar %372, %c-1_i8, %303 : <ui8>, i8 -> !ascvec.mask
                            %374 = ascvec.min_scalar %372, %c123_i8, %373 : <ui8>, i8
                            %375 = arith.muli %arg15, %c512_i32 : i32
                            %376 = arith.index_cast %375 : i32 to index
                            %result0_18, %result1_19 = ascvec.loadalign_deintlv %arg10[%376] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %377 = ascvec.and %result1_19, %305, %303 : <ui8>, <ui8> -> <ui8>
                            %378 = ascvec.or %374, %377, %303 : <ui8>, <ui8> -> <ui8>
                            %379 = ascvec.arange %c0_i16 : i16 -> <ui16>
                            %380 = arith.muli %arg15, %c16_i32 : i32
                            %381 = arith.trunci %380 : i32 to i16
                            %382 = ascvec.add_scalar %379, %381, %302 : <ui16>, i16
                            %383 = ascvec.gather %arg11[%c0_11], %382, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                            %384 = ascvec.pack %383 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                            %385 = ascvec.gather_reg %384, %306 : <ui8>, <ui8> -> <ui8>
                            %386 = ascvec.cmp_eq_scalar %385, %c-1_i8, %303 : <ui8>, i8 -> !ascvec.mask
                            %387 = ascvec.duplicate_scalar %c127_i8, %386 merge %378 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                            ascvec.storealign %arg12[%314], %387, %303 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
                      }
                    }
                  }
                }
              } else {
                %223 = arith.divsi %98, %c256_i64 : i64
                %224 = arith.addi %98, %c1023_i64 : i64
                %225 = arith.remsi %224, %c1024_i64 : i64
                %226 = arith.cmpi ne, %225, %c0_i64 : i64
                %227 = arith.cmpi slt, %225, %c0_i64 : i64
                %228 = arith.andi %226, %227 : i1
                %229 = arith.divsi %224, %c1024_i64 : i64
                %230 = arith.extui %228 : i1 to i64
                %231 = arith.subi %229, %230 : i64
                %232 = arith.trunci %1 : i64 to i32
                %233 = arith.extui %232 : i32 to i64
                %234 = arith.subi %1, %233 : i64
                %235 = arith.remsi %234, %c4294967296_i64 : i64
                %236 = arith.cmpi ne, %235, %c0_i64 : i64
                %237 = arith.cmpi slt, %235, %c0_i64 : i64
                %238 = arith.andi %236, %237 : i1
                %239 = arith.divsi %234, %c4294967296_i64 : i64
                %240 = arith.extui %238 : i1 to i64
                %241 = arith.subi %239, %240 : i64
                %242 = arith.trunci %241 : i64 to i32
                %243 = arith.divui %211, %c16_i64 : i64
                %244 = arith.trunci %243 : i64 to i32
                %245 = arith.extui %244 : i32 to i64
                %246 = arith.subi %243, %245 : i64
                %247 = arith.remsi %246, %c4294967296_i64 : i64
                %248 = arith.cmpi ne, %247, %c0_i64 : i64
                %249 = arith.cmpi slt, %247, %c0_i64 : i64
                %250 = arith.andi %248, %249 : i1
                %251 = arith.divsi %246, %c4294967296_i64 : i64
                %252 = arith.extui %250 : i1 to i64
                %253 = arith.subi %251, %252 : i64
                %254 = arith.trunci %253 : i64 to i32
                %255 = arith.trunci %3 : i64 to i32
                %256 = arith.extui %255 : i32 to i64
                %257 = arith.subi %3, %256 : i64
                %258 = arith.remsi %257, %c4294967296_i64 : i64
                %259 = arith.cmpi ne, %258, %c0_i64 : i64
                %260 = arith.cmpi slt, %258, %c0_i64 : i64
                %261 = arith.andi %259, %260 : i1
                %262 = arith.divsi %257, %c4294967296_i64 : i64
                %263 = arith.extui %261 : i1 to i64
                %264 = arith.subi %262, %263 : i64
                %265 = arith.trunci %264 : i64 to i32
                %266 = arith.divui %255, %c4_i32 : i32
                %267 = arith.remui %265, %c4_i32 : i32
                %268 = arith.muli %267, %c1073741824_i32 : i32
                %269 = arith.addi %266, %268 : i32
                %270 = arith.divui %265, %c4_i32 : i32
                %271 = arith.muli %231, %c64_i64 : i64
                %272 = arith.divui %243, %c16384_i64 : i64
                %273 = arith.addi %243, %271 : i64
                %274 = arith.subi %273, %c1_i64 : i64
                %275 = arith.divui %274, %c16384_i64 : i64
                %276 = arith.cmpi eq, %272, %275 : i64
                scf.if %276 {
                  %282 = arith.trunci %272 : i64 to i32
                  %283 = arith.extui %282 : i32 to i64
                  %284 = arith.subi %272, %283 : i64
                  %285 = arith.remsi %284, %c4294967296_i64 : i64
                  %286 = arith.cmpi ne, %285, %c0_i64 : i64
                  %287 = arith.cmpi slt, %285, %c0_i64 : i64
                  %288 = arith.andi %286, %287 : i1
                  %289 = arith.divsi %284, %c4294967296_i64 : i64
                  %290 = arith.extui %288 : i1 to i64
                  %291 = arith.subi %289, %290 : i64
                  %292 = arith.trunci %291 : i64 to i32
                  %293 = arith.addi %269, %282 : i32
                  %294 = arith.addi %270, %292 : i32
                  %295 = arith.cmpi ult, %293, %269 : i32
                  %296 = arith.extui %295 : i1 to i32
                  %297 = arith.addi %294, %296 : i32
                  %298 = arith.extui %293 : i32 to i64
                  %299 = arith.muli %298, %c3449720151_i64 : i64
                  %300 = arith.trunci %299 : i64 to i32
                  %301 = arith.extui %300 : i32 to i64
                  %302 = arith.subi %299, %301 : i64
                  %303 = arith.remsi %302, %c4294967296_i64 : i64
                  %304 = arith.cmpi ne, %303, %c0_i64 : i64
                  %305 = arith.cmpi slt, %303, %c0_i64 : i64
                  %306 = arith.andi %304, %305 : i1
                  %307 = arith.divsi %302, %c4294967296_i64 : i64
                  %308 = arith.extui %306 : i1 to i64
                  %309 = arith.subi %307, %308 : i64
                  %310 = arith.trunci %309 : i64 to i32
                  %311 = arith.xori %297, %242 : i32
                  %312 = arith.extui %311 : i32 to i64
                  %313 = arith.muli %312, %c3449720151_i64 : i64
                  %314 = arith.trunci %313 : i64 to i32
                  %315 = arith.extui %314 : i32 to i64
                  %316 = arith.subi %313, %315 : i64
                  %317 = arith.remsi %316, %c4294967296_i64 : i64
                  %318 = arith.cmpi ne, %317, %c0_i64 : i64
                  %319 = arith.cmpi slt, %317, %c0_i64 : i64
                  %320 = arith.andi %318, %319 : i1
                  %321 = arith.divsi %316, %c4294967296_i64 : i64
                  %322 = arith.extui %320 : i1 to i64
                  %323 = arith.subi %321, %322 : i64
                  %324 = arith.trunci %323 : i64 to i32
                  %325 = arith.addi %232, %c-1640531527_i32 : i32
                  %326 = arith.addi %242, %c-1150833019_i32 : i32
                  %327 = arith.xori %324, %300 : i32
                  %328 = arith.xori %327, %325 : i32
                  %329 = arith.extui %328 : i32 to i64
                  %330 = arith.muli %329, %c3528531795_i64 : i64
                  %331 = arith.trunci %330 : i64 to i32
                  %332 = arith.extui %331 : i32 to i64
                  %333 = arith.subi %330, %332 : i64
                  %334 = arith.remsi %333, %c4294967296_i64 : i64
                  %335 = arith.cmpi ne, %334, %c0_i64 : i64
                  %336 = arith.cmpi slt, %334, %c0_i64 : i64
                  %337 = arith.andi %335, %336 : i1
                  %338 = arith.divsi %333, %c4294967296_i64 : i64
                  %339 = arith.extui %337 : i1 to i64
                  %340 = arith.subi %338, %339 : i64
                  %341 = arith.trunci %340 : i64 to i32
                  %342 = arith.xori %310, %232 : i32
                  %343 = arith.addi %232, %c1013904242_i32 : i32
                  %344 = arith.xori %314, %343 : i32
                  %345 = arith.addi %242, %c1993301258_i32 : i32
                  %346 = arith.xori %341, %345 : i32
                  ascsimt.vf(%244, %342, %326, %344, %346, %232, %242, %331, %alloc_8, %271 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
                    %c256_i64_10 = arith.constant 256 : i64
                    %c3_i64_11 = arith.constant 3 : i64
                    %c2_i64_12 = arith.constant 2 : i64
                    %c1_i64_13 = arith.constant 1 : i64
                    %c4_i64_14 = arith.constant 4 : i64
                    %c1_i32_15 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c3_i32 = arith.constant 3 : i32
                    %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %347 = ascsimt.thread_idx {ascsimt.scope = 51 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %348 = arith.extsi %347 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg17 = %348 to %arg16 step %c256_i64_10  : i64 {
                      %349 = arith.trunci %arg17 : i64 to i32
                      %350 = arith.addi %arg7, %349 : i32
                      %351 = arith.remui %350, %c16384_i32 : i32
                      %352 = arith.xori %351, %arg8 : i32
                      %353 = arith.muli %352, %c-766435501_i32 : i32
                      %354 = "ascsimt.bitcast"(%352) : (i32) -> ui32
                      %355 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %356 = ascsimt.math_binary %354, %355, "mulhi" : ui32, ui32 -> ui32
                      %357 = "ascsimt.bitcast"(%356) : (ui32) -> i32
                      %358 = arith.xori %357, %arg9 : i32
                      %359 = arith.muli %358, %c-845247145_i32 : i32
                      %360 = "ascsimt.bitcast"(%358) : (i32) -> ui32
                      %361 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %362 = ascsimt.math_binary %360, %361, "mulhi" : ui32, ui32 -> ui32
                      %363 = "ascsimt.bitcast"(%362) : (ui32) -> i32
                      %364 = arith.xori %363, %arg10 : i32
                      %365 = arith.xori %353, %arg11 : i32
                      %366:4 = scf.for %arg18 = %c3_i32 to %c10_i32 step %c1_i32_15 iter_args(%arg19 = %364, %arg20 = %359, %arg21 = %365, %arg22 = %arg14) -> (i32, i32, i32, i32)  : i32 {
                        %375 = arith.muli %arg19, %c-766435501_i32 : i32
                        %376 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %377 = ascsimt.math_binary %376, %355, "mulhi" : ui32, ui32 -> ui32
                        %378 = "ascsimt.bitcast"(%377) : (ui32) -> i32
                        %379 = arith.muli %arg21, %c-845247145_i32 : i32
                        %380 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %381 = ascsimt.math_binary %380, %361, "mulhi" : ui32, ui32 -> ui32
                        %382 = "ascsimt.bitcast"(%381) : (ui32) -> i32
                        %383 = arith.muli %arg18, %c-1640531527_i32_17 : i32
                        %384 = arith.addi %arg12, %383 : i32
                        %385 = arith.muli %arg18, %c-1150833019_i32_16 : i32
                        %386 = arith.addi %arg13, %385 : i32
                        %387 = arith.xori %382, %arg20 : i32
                        %388 = arith.xori %387, %384 : i32
                        %389 = arith.xori %378, %arg22 : i32
                        %390 = arith.xori %389, %386 : i32
                        scf.yield %388, %379, %390, %375 : i32, i32, i32, i32
                      }
                      %367 = arith.muli %arg17, %c4_i64_14 : i64
                      %368 = arith.index_cast %367 : i64 to index
                      ascsimt.store %arg15[%368], %366#0 : memref<8192xi32, 1>, i32
                      %369 = arith.addi %367, %c1_i64_13 : i64
                      %370 = arith.index_cast %369 : i64 to index
                      ascsimt.store %arg15[%370], %366#1 : memref<8192xi32, 1>, i32
                      %371 = arith.addi %367, %c2_i64_12 : i64
                      %372 = arith.index_cast %371 : i64 to index
                      ascsimt.store %arg15[%372], %366#2 : memref<8192xi32, 1>, i32
                      %373 = arith.addi %367, %c3_i64_11 : i64
                      %374 = arith.index_cast %373 : i64 to index
                      ascsimt.store %arg15[%374], %366#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%244, %254, %269, %270, %232, %242, %alloc_8, %271 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: memref<8192xi32, 1>, %arg14: i64):
                    %c256_i64_10 = arith.constant 256 : i64
                    %c3_i64_11 = arith.constant 3 : i64
                    %c2_i64_12 = arith.constant 2 : i64
                    %c1_i64_13 = arith.constant 1 : i64
                    %c4_i64_14 = arith.constant 4 : i64
                    %c1_i32_15 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c0_i32_16 = arith.constant 0 : i32
                    %c-1150833019_i32_17 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_18 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c262144_i32 = arith.constant 262144 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %282 = ascsimt.thread_idx {ascsimt.scope = 52 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %283 = arith.extsi %282 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg15 = %283 to %arg14 step %c256_i64_10  : i64 {
                      %284 = arith.trunci %arg15 : i64 to i32
                      %285 = arith.addi %arg7, %284 : i32
                      %286 = arith.cmpi ult, %285, %arg7 : i32
                      %287 = arith.extui %286 : i1 to i32
                      %288 = arith.addi %arg8, %287 : i32
                      %289 = arith.divui %285, %c16384_i32 : i32
                      %290 = arith.remui %288, %c16384_i32 : i32
                      %291 = arith.muli %290, %c262144_i32 : i32
                      %292 = arith.addi %289, %291 : i32
                      %293 = arith.divui %288, %c16384_i32 : i32
                      %294 = arith.addi %arg9, %292 : i32
                      %295 = arith.addi %arg10, %293 : i32
                      %296 = arith.cmpi ult, %294, %arg9 : i32
                      %297 = arith.extui %296 : i1 to i32
                      %298 = arith.addi %295, %297 : i32
                      %299 = arith.remui %285, %c16384_i32 : i32
                      %300:4 = scf.for %arg16 = %c0_i32_16 to %c10_i32 step %c1_i32_15 iter_args(%arg17 = %c0_i32_16, %arg18 = %299, %arg19 = %294, %arg20 = %298) -> (i32, i32, i32, i32)  : i32 {
                        %309 = arith.muli %arg17, %c-766435501_i32 : i32
                        %310 = "ascsimt.bitcast"(%arg17) : (i32) -> ui32
                        %311 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %312 = ascsimt.math_binary %310, %311, "mulhi" : ui32, ui32 -> ui32
                        %313 = "ascsimt.bitcast"(%312) : (ui32) -> i32
                        %314 = arith.muli %arg19, %c-845247145_i32 : i32
                        %315 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %316 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %317 = ascsimt.math_binary %315, %316, "mulhi" : ui32, ui32 -> ui32
                        %318 = "ascsimt.bitcast"(%317) : (ui32) -> i32
                        %319 = arith.muli %arg16, %c-1640531527_i32_18 : i32
                        %320 = arith.addi %arg11, %319 : i32
                        %321 = arith.muli %arg16, %c-1150833019_i32_17 : i32
                        %322 = arith.addi %arg12, %321 : i32
                        %323 = arith.xori %318, %arg18 : i32
                        %324 = arith.xori %323, %320 : i32
                        %325 = arith.xori %313, %arg20 : i32
                        %326 = arith.xori %325, %322 : i32
                        scf.yield %324, %314, %326, %309 : i32, i32, i32, i32
                      }
                      %301 = arith.muli %arg15, %c4_i64_14 : i64
                      %302 = arith.index_cast %301 : i64 to index
                      ascsimt.store %arg13[%302], %300#0 : memref<8192xi32, 1>, i32
                      %303 = arith.addi %301, %c1_i64_13 : i64
                      %304 = arith.index_cast %303 : i64 to index
                      ascsimt.store %arg13[%304], %300#1 : memref<8192xi32, 1>, i32
                      %305 = arith.addi %301, %c2_i64_12 : i64
                      %306 = arith.index_cast %305 : i64 to index
                      ascsimt.store %arg13[%306], %300#2 : memref<8192xi32, 1>, i32
                      %307 = arith.addi %301, %c3_i64_11 : i64
                      %308 = arith.index_cast %307 : i64 to index
                      ascsimt.store %arg13[%308], %300#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %277 = arith.trunci %31 : i64 to i32
                %278 = arith.divsi %277, %c2_i32 : i32
                %279 = arith.trunci %278 : i32 to i8
                %280 = arith.trunci %223 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                %281 = ascscalar.bitcast_memref %133 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                ascvec.vf(%279, %alloc_6, %133, %281, %alloc_8, %alloc_5, %280 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui8, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, i32) {
                ^bb0(%arg7: i8, %arg8: memref<2048xui8, 1>, %arg9: memref<32768xbf16, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<8192xi32, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32):
                  %c512_i32 = arith.constant 512 : i32
                  %c1_i32_10 = arith.constant 1 : i32
                  %c2139062143_i32 = arith.constant 2139062143 : i32
                  %c-8323200_i32 = arith.constant -8323200 : i32
                  %c123_i8 = arith.constant 123 : i8
                  %c-1_i8 = arith.constant -1 : i8
                  %c60_i8 = arith.constant 60 : i8
                  %cst = arith.constant 6.10351526E-5 : f32
                  %c64_i32 = arith.constant 64 : i32
                  %c0_i16 = arith.constant 0 : i16
                  %c0_i32_11 = arith.constant 0 : i32
                  %c254_i32 = arith.constant 254 : i32
                  %c-15_i32 = arith.constant -15 : i32
                  %c15_i32 = arith.constant 15 : i32
                  %c255_i32 = arith.constant 255 : i32
                  %c32767_i16 = arith.constant 32767 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c0_12 = arith.constant 0 : index
                  %c32_i8 = arith.constant 32 : i8
                  %c65535_i32 = arith.constant 65535 : i32
                  %c3_i8 = arith.constant 3 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c-128_i8 = arith.constant -128 : i8
                  %282 = ascvec.create_mask : !ascvec.mask
                  %283 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %284 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %285 = ascvec.duplicate_scalar %c-128_i8, %284 : i8 -> <ui8>
                  %286 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %287 = ascvec.duplicate_scalar %c3_i8, %284 : i8 -> <ui8>
                  %288 = ascvec.and %286, %287, %284 : <ui8>, <ui8> -> <ui8>
                  %289 = ascvec.shl %288, 6, %284 : <ui8> -> <ui8>
                  %290 = ascvec.shr %286, 2, %284 : <ui8> -> <ui8>
                  %291 = ascvec.or %289, %290, %284 : <ui8>, <ui8> -> <ui8>
                  %292 = ascvec.duplicate_scalar %c65535_i32, %282 : i32 -> <ui32>
                  %293 = ascvec.cmp_lt_scalar %286, %c32_i8, %284 : <ui8>, i8 -> !ascvec.mask
                  %294 = ascvec.cmp_lt_scalar %288, %arg7, %284 : <ui8>, i8 -> !ascvec.mask
                  %295 = ascvec.mask_and %293, %294, %284
                  ascvec.store_unalign_begin %arg8 : memref<2048xui8, 1>
                  scf.for %arg14 = %c0_i32_11 to %arg13 step %c1_i32_10  : i32 {
                    %296 = arith.muli %arg14, %c256_i32 : i32
                    %297 = arith.index_cast %296 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg9[%297] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %298 = ascvec.duplicate_scalar %c32767_i16, %283 : i16 -> <ui16>
                    %299 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %300 = ascvec.and %299, %298, %283 : <ui16>, <ui16> -> <ui16>
                    %301 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %302 = ascvec.and %301, %298, %283 : <ui16>, <ui16> -> <ui16>
                    %303 = ascvec.max %300, %302, %283 : <ui16>
                    %304 = ascvec.reduce_max_datablock %303, %283 : <ui16>
                    %305 = ascvec.unpack %304 {part = "lower"} : <ui16> -> <ui32>
                    %306 = ascvec.shl %305, 16, %282 : <ui32> -> <ui32>
                    %307 = arith.muli %arg14, %c512_i32 : i32
                    %308 = arith.index_cast %307 : i32 to index
                    %result0_15, %result1_16 = ascvec.loadalign_deintlv %arg10[%308] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                    %309 = ascvec.and %result1_16, %285, %284 : <ui8>, <ui8> -> <ui8>
                    %310 = ascvec.reinterpret %309 : <ui8> -> <ui32>
                    %311 = ascvec.shr %306, 23, %282 : <ui32> -> <ui32>
                    %312 = ascvec.cmp_ge_scalar %311, %c255_i32, %282 : <ui32>, i32 -> !ascvec.mask
                    %313 = ascvec.max_scalar %311, %c15_i32, %282 : <ui32>, i32
                    %314 = ascvec.add_scalar %313, %c-15_i32, %282 : <ui32>, i32
                    %315 = ascvec.duplicate_scalar %c255_i32, %312 merge %314 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                    %316 = ascvec.duplicate_scalar %c254_i32, %282 : i32 -> <ui32>
                    %317 = ascvec.sub %316, %315, %282 : <ui32>
                    %318 = ascvec.cmp_ne_scalar %315, %c0_i32_11, %282 : <ui32>, i32 -> !ascvec.mask
                    %319 = ascvec.shl %317, 23, %318 : <ui32> -> <ui32>
                    %320 = ascvec.reinterpret %319 : <ui32> -> <f32>
                    %321 = ascvec.reinterpret %320 : <f32> -> <ui32>
                    %322 = ascvec.shr %321, 16, %282 : <ui32> -> <ui32>
                    %323 = ascvec.pack %322 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                    %324 = ascvec.arange %c0_i16 : i16 -> <ui16>
                    %325 = ascvec.shr %324, 4, %283 : <ui16> -> <ui16>
                    %326 = ascvec.gather_reg %323, %325 : <ui16>, <ui16> -> <ui16>
                    %327 = ascvec.reinterpret %326 : <ui16> -> <bf16>
                    %328 = ascvec.reinterpret %300 : <ui16> -> <bf16>
                    %329 = ascvec.mul %328, %327, %283 : <bf16>
                    %330 = ascvec.reinterpret %302 : <ui16> -> <bf16>
                    %331 = ascvec.mul %330, %327, %283 : <bf16>
                    %332 = ascvec.cast %329, %283 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %333 = ascvec.cast %329, %283 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %334 = ascvec.cast %331, %283 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %335 = ascvec.cast %331, %283 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %336 = ascvec.reinterpret %327 : <bf16> -> <ui32>
                    %337 = arith.muli %arg14, %c64_i32 : i32
                    %338 = arith.index_cast %337 : i32 to index
                    %339 = ascvec.loadalign %arg11[%338] : memref<8192xi32, 1>, index -> <i32>
                    %340 = ascvec.reinterpret %339 : <i32> -> <ui32>
                    %341 = ascvec.shl %340, 8, %282 : <ui32> -> <ui32>
                    %342 = ascvec.shr %340, 24, %282 : <ui32> -> <ui32>
                    %343 = ascvec.or %341, %342, %282 : <ui32>, <ui32> -> <ui32>
                    %344 = ascvec.shr %340, 16, %282 : <ui32> -> <ui32>
                    %345 = ascvec.and %343, %292, %282 : <ui32>, <ui32> -> <ui32>
                    %346 = ascvec.and %340, %292, %282 : <ui32>, <ui32> -> <ui32>
                    %347 = ascvec.shr %343, 16, %282 : <ui32> -> <ui32>
                    %348 = ascvec.reinterpret %332 : <f32> -> <ui32>
                    %349 = ascvec.reinterpret %348 : <ui32> -> <f32>
                    %350 = ascvec.max_scalar %349, %cst, %282 : <f32>, f32
                    %351 = ascvec.add %349, %350, %282 : <f32>
                    %352 = ascvec.reinterpret %351 : <f32> -> <ui32>
                    %353 = ascvec.shr %352, 5, %282 : <ui32> -> <ui32>
                    %354 = ascvec.add %353, %344, %282 : <ui32>
                    %355 = ascvec.reinterpret %334 : <f32> -> <ui32>
                    %356 = ascvec.reinterpret %355 : <ui32> -> <f32>
                    %357 = ascvec.max_scalar %356, %cst, %282 : <f32>, f32
                    %358 = ascvec.add %356, %357, %282 : <f32>
                    %359 = ascvec.reinterpret %358 : <f32> -> <ui32>
                    %360 = ascvec.shr %359, 5, %282 : <ui32> -> <ui32>
                    %361 = ascvec.add %360, %345, %282 : <ui32>
                    %362 = ascvec.reinterpret %333 : <f32> -> <ui32>
                    %363 = ascvec.reinterpret %362 : <ui32> -> <f32>
                    %364 = ascvec.max_scalar %363, %cst, %282 : <f32>, f32
                    %365 = ascvec.add %363, %364, %282 : <f32>
                    %366 = ascvec.reinterpret %365 : <f32> -> <ui32>
                    %367 = ascvec.shr %366, 5, %282 : <ui32> -> <ui32>
                    %368 = ascvec.add %367, %346, %282 : <ui32>
                    %369 = ascvec.reinterpret %335 : <f32> -> <ui32>
                    %370 = ascvec.reinterpret %369 : <ui32> -> <f32>
                    %371 = ascvec.max_scalar %370, %cst, %282 : <f32>, f32
                    %372 = ascvec.add %370, %371, %282 : <f32>
                    %373 = ascvec.reinterpret %372 : <f32> -> <ui32>
                    %374 = ascvec.shr %373, 5, %282 : <ui32> -> <ui32>
                    %375 = ascvec.add %374, %347, %282 : <ui32>
                    %376 = ascvec.reinterpret %354 : <ui32> -> <ui16>
                    %377 = ascvec.reinterpret %361 : <ui32> -> <ui16>
                    %result0_17, %result1_18 = ascvec.deintlv %376, %377 : <ui16>, <ui16> -> <ui16>, <ui16>
                    %378 = ascvec.reinterpret %368 : <ui32> -> <ui16>
                    %379 = ascvec.reinterpret %375 : <ui32> -> <ui16>
                    %result0_19, %result1_20 = ascvec.deintlv %378, %379 : <ui16>, <ui16> -> <ui16>, <ui16>
                    %380 = ascvec.reinterpret %result1_18 : <ui16> -> <ui8>
                    %381 = ascvec.reinterpret %result1_20 : <ui16> -> <ui8>
                    %result0_21, %result1_22 = ascvec.deintlv %380, %381 : <ui8>, <ui8> -> <ui8>, <ui8>
                    %382 = ascvec.gather_reg %result0_21, %291 : <ui8>, <ui8> -> <ui8>
                    %383 = ascvec.reinterpret %382 : <ui8> -> <ui32>
                    %384 = ascvec.reinterpret %383 : <ui32> -> <ui8>
                    %385 = ascvec.add_scalar %384, %c60_i8, %284 : <ui8>, i8
                    %386 = ascvec.cmp_ne_scalar %385, %c-1_i8, %284 : <ui8>, i8 -> !ascvec.mask
                    %387 = ascvec.min_scalar %385, %c123_i8, %386 : <ui8>, i8
                    %388 = ascvec.reinterpret %387 : <ui8> -> <ui32>
                    %389 = ascvec.or %388, %310, %282 : <ui32>, <ui32> -> <ui32>
                    %390 = ascvec.reinterpret %336 : <ui32> -> <ui32>
                    %391 = ascvec.cmp_eq_scalar %390, %c-8323200_i32, %282 : <ui32>, i32 -> !ascvec.mask
                    %392 = ascvec.duplicate_scalar %c2139062143_i32, %391 merge %389 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                    %393 = ascvec.reinterpret %392 : <ui32> -> <ui8>
                    ascvec.storealign %arg12[%297], %393, %284 : memref<32768xui8, 1>, index, <ui8>
                    %394 = ascvec.reinterpret %315 : <ui32> -> <ui8>
                    %395 = ascvec.squeeze_store_reg %394, %295 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg8[%c0_12], %395 : memref<2048xui8, 1>, <ui8>
                  }
                  %c0_13 = arith.constant 0 : index
                  %c1_14 = arith.constant 1 : index
                  scf.for %arg14 = %c0_13 to %c1_14 step %c1_14 {
                    ascvec.store_unalign_post %arg8[%c0_12] : memref<2048xui8, 1>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              }
            }
          } else {
            %210 = arith.muli %92, %11 : i64
            %211 = arith.addi %210, %95 : i64
            %212 = arith.remsi %arg3, %c256_i64 : i64
            %213 = arith.cmpi ne, %212, %c0_i64 : i64
            %214 = arith.cmpi slt, %212, %c0_i64 : i64
            %215 = arith.andi %213, %214 : i1
            %216 = arith.select %215, %c256_i64, %c0_i64 : i64
            %217 = arith.addi %212, %216 overflow<nsw> : i64
            %218 = arith.cmpi eq, %217, %c0_i64 : i64
            scf.if %218 {
              %219 = arith.addi %98, %c63_i64 : i64
              %220 = arith.remsi %219, %c64_i64 : i64
              %221 = arith.cmpi ne, %220, %c0_i64 : i64
              %222 = arith.cmpi slt, %220, %c0_i64 : i64
              %223 = arith.andi %221, %222 : i1
              %224 = arith.divsi %219, %c64_i64 : i64
              %225 = arith.extui %223 : i1 to i64
              %226 = arith.subi %224, %225 : i64
              %227 = arith.addi %98, %c1023_i64 : i64
              %228 = arith.remsi %227, %c1024_i64 : i64
              %229 = arith.cmpi ne, %228, %c0_i64 : i64
              %230 = arith.cmpi slt, %228, %c0_i64 : i64
              %231 = arith.andi %229, %230 : i1
              %232 = arith.divsi %227, %c1024_i64 : i64
              %233 = arith.extui %231 : i1 to i64
              %234 = arith.subi %232, %233 : i64
              %235 = arith.trunci %1 : i64 to i32
              %236 = arith.extui %235 : i32 to i64
              %237 = arith.subi %1, %236 : i64
              %238 = arith.remsi %237, %c4294967296_i64 : i64
              %239 = arith.cmpi ne, %238, %c0_i64 : i64
              %240 = arith.cmpi slt, %238, %c0_i64 : i64
              %241 = arith.andi %239, %240 : i1
              %242 = arith.divsi %237, %c4294967296_i64 : i64
              %243 = arith.extui %241 : i1 to i64
              %244 = arith.subi %242, %243 : i64
              %245 = arith.trunci %244 : i64 to i32
              %246 = arith.divui %211, %c32_i64 : i64
              %247 = arith.trunci %246 : i64 to i32
              %248 = arith.extui %247 : i32 to i64
              %249 = arith.subi %246, %248 : i64
              %250 = arith.remsi %249, %c4294967296_i64 : i64
              %251 = arith.cmpi ne, %250, %c0_i64 : i64
              %252 = arith.cmpi slt, %250, %c0_i64 : i64
              %253 = arith.andi %251, %252 : i1
              %254 = arith.divsi %249, %c4294967296_i64 : i64
              %255 = arith.extui %253 : i1 to i64
              %256 = arith.subi %254, %255 : i64
              %257 = arith.trunci %256 : i64 to i32
              %258 = arith.trunci %3 : i64 to i32
              %259 = arith.extui %258 : i32 to i64
              %260 = arith.subi %3, %259 : i64
              %261 = arith.remsi %260, %c4294967296_i64 : i64
              %262 = arith.cmpi ne, %261, %c0_i64 : i64
              %263 = arith.cmpi slt, %261, %c0_i64 : i64
              %264 = arith.andi %262, %263 : i1
              %265 = arith.divsi %260, %c4294967296_i64 : i64
              %266 = arith.extui %264 : i1 to i64
              %267 = arith.subi %265, %266 : i64
              %268 = arith.trunci %267 : i64 to i32
              %269 = arith.divui %258, %c4_i32 : i32
              %270 = arith.remui %268, %c4_i32 : i32
              %271 = arith.muli %270, %c1073741824_i32 : i32
              %272 = arith.addi %269, %271 : i32
              %273 = arith.divui %268, %c4_i32 : i32
              %274 = arith.muli %234, %c32_i64 : i64
              %275 = arith.divui %246, %c16384_i64 : i64
              %276 = arith.addi %246, %274 : i64
              %277 = arith.subi %276, %c1_i64 : i64
              %278 = arith.divui %277, %c16384_i64 : i64
              %279 = arith.cmpi eq, %275, %278 : i64
              scf.if %279 {
                %285 = arith.trunci %275 : i64 to i32
                %286 = arith.extui %285 : i32 to i64
                %287 = arith.subi %275, %286 : i64
                %288 = arith.remsi %287, %c4294967296_i64 : i64
                %289 = arith.cmpi ne, %288, %c0_i64 : i64
                %290 = arith.cmpi slt, %288, %c0_i64 : i64
                %291 = arith.andi %289, %290 : i1
                %292 = arith.divsi %287, %c4294967296_i64 : i64
                %293 = arith.extui %291 : i1 to i64
                %294 = arith.subi %292, %293 : i64
                %295 = arith.trunci %294 : i64 to i32
                %296 = arith.addi %272, %285 : i32
                %297 = arith.addi %273, %295 : i32
                %298 = arith.cmpi ult, %296, %272 : i32
                %299 = arith.extui %298 : i1 to i32
                %300 = arith.addi %297, %299 : i32
                %301 = arith.extui %296 : i32 to i64
                %302 = arith.muli %301, %c3449720151_i64 : i64
                %303 = arith.trunci %302 : i64 to i32
                %304 = arith.extui %303 : i32 to i64
                %305 = arith.subi %302, %304 : i64
                %306 = arith.remsi %305, %c4294967296_i64 : i64
                %307 = arith.cmpi ne, %306, %c0_i64 : i64
                %308 = arith.cmpi slt, %306, %c0_i64 : i64
                %309 = arith.andi %307, %308 : i1
                %310 = arith.divsi %305, %c4294967296_i64 : i64
                %311 = arith.extui %309 : i1 to i64
                %312 = arith.subi %310, %311 : i64
                %313 = arith.trunci %312 : i64 to i32
                %314 = arith.xori %300, %245 : i32
                %315 = arith.extui %314 : i32 to i64
                %316 = arith.muli %315, %c3449720151_i64 : i64
                %317 = arith.trunci %316 : i64 to i32
                %318 = arith.extui %317 : i32 to i64
                %319 = arith.subi %316, %318 : i64
                %320 = arith.remsi %319, %c4294967296_i64 : i64
                %321 = arith.cmpi ne, %320, %c0_i64 : i64
                %322 = arith.cmpi slt, %320, %c0_i64 : i64
                %323 = arith.andi %321, %322 : i1
                %324 = arith.divsi %319, %c4294967296_i64 : i64
                %325 = arith.extui %323 : i1 to i64
                %326 = arith.subi %324, %325 : i64
                %327 = arith.trunci %326 : i64 to i32
                %328 = arith.addi %235, %c-1640531527_i32 : i32
                %329 = arith.addi %245, %c-1150833019_i32 : i32
                %330 = arith.xori %327, %303 : i32
                %331 = arith.xori %330, %328 : i32
                %332 = arith.extui %331 : i32 to i64
                %333 = arith.muli %332, %c3528531795_i64 : i64
                %334 = arith.trunci %333 : i64 to i32
                %335 = arith.extui %334 : i32 to i64
                %336 = arith.subi %333, %335 : i64
                %337 = arith.remsi %336, %c4294967296_i64 : i64
                %338 = arith.cmpi ne, %337, %c0_i64 : i64
                %339 = arith.cmpi slt, %337, %c0_i64 : i64
                %340 = arith.andi %338, %339 : i1
                %341 = arith.divsi %336, %c4294967296_i64 : i64
                %342 = arith.extui %340 : i1 to i64
                %343 = arith.subi %341, %342 : i64
                %344 = arith.trunci %343 : i64 to i32
                %345 = arith.xori %313, %235 : i32
                %346 = arith.addi %235, %c1013904242_i32 : i32
                %347 = arith.xori %317, %346 : i32
                %348 = arith.addi %245, %c1993301258_i32 : i32
                %349 = arith.xori %344, %348 : i32
                ascsimt.vf(%247, %345, %329, %347, %349, %235, %245, %334, %alloc_8, %274 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
                  %c256_i64_10 = arith.constant 256 : i64
                  %c3_i64_11 = arith.constant 3 : i64
                  %c2_i64_12 = arith.constant 2 : i64
                  %c1_i64_13 = arith.constant 1 : i64
                  %c4_i64_14 = arith.constant 4 : i64
                  %c1_i32_15 = arith.constant 1 : i32
                  %c10_i32 = arith.constant 10 : i32
                  %c3_i32 = arith.constant 3 : i32
                  %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                  %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                  %c-845247145_i32 = arith.constant -845247145 : i32
                  %c-766435501_i32 = arith.constant -766435501 : i32
                  %c16384_i32 = arith.constant 16384 : i32
                  %350 = ascsimt.thread_idx {ascsimt.scope = 54 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                  %351 = arith.extsi %350 {cannir.vf_threads = 256 : i64} : i32 to i64
                  scf.for %arg17 = %351 to %arg16 step %c256_i64_10  : i64 {
                    %352 = arith.trunci %arg17 : i64 to i32
                    %353 = arith.addi %arg7, %352 : i32
                    %354 = arith.remui %353, %c16384_i32 : i32
                    %355 = arith.xori %354, %arg8 : i32
                    %356 = arith.muli %355, %c-766435501_i32 : i32
                    %357 = "ascsimt.bitcast"(%355) : (i32) -> ui32
                    %358 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                    %359 = ascsimt.math_binary %357, %358, "mulhi" : ui32, ui32 -> ui32
                    %360 = "ascsimt.bitcast"(%359) : (ui32) -> i32
                    %361 = arith.xori %360, %arg9 : i32
                    %362 = arith.muli %361, %c-845247145_i32 : i32
                    %363 = "ascsimt.bitcast"(%361) : (i32) -> ui32
                    %364 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                    %365 = ascsimt.math_binary %363, %364, "mulhi" : ui32, ui32 -> ui32
                    %366 = "ascsimt.bitcast"(%365) : (ui32) -> i32
                    %367 = arith.xori %366, %arg10 : i32
                    %368 = arith.xori %356, %arg11 : i32
                    %369:4 = scf.for %arg18 = %c3_i32 to %c10_i32 step %c1_i32_15 iter_args(%arg19 = %367, %arg20 = %362, %arg21 = %368, %arg22 = %arg14) -> (i32, i32, i32, i32)  : i32 {
                      %378 = arith.muli %arg19, %c-766435501_i32 : i32
                      %379 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                      %380 = ascsimt.math_binary %379, %358, "mulhi" : ui32, ui32 -> ui32
                      %381 = "ascsimt.bitcast"(%380) : (ui32) -> i32
                      %382 = arith.muli %arg21, %c-845247145_i32 : i32
                      %383 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                      %384 = ascsimt.math_binary %383, %364, "mulhi" : ui32, ui32 -> ui32
                      %385 = "ascsimt.bitcast"(%384) : (ui32) -> i32
                      %386 = arith.muli %arg18, %c-1640531527_i32_17 : i32
                      %387 = arith.addi %arg12, %386 : i32
                      %388 = arith.muli %arg18, %c-1150833019_i32_16 : i32
                      %389 = arith.addi %arg13, %388 : i32
                      %390 = arith.xori %385, %arg20 : i32
                      %391 = arith.xori %390, %387 : i32
                      %392 = arith.xori %381, %arg22 : i32
                      %393 = arith.xori %392, %389 : i32
                      scf.yield %391, %382, %393, %378 : i32, i32, i32, i32
                    }
                    %370 = arith.muli %arg17, %c4_i64_14 : i64
                    %371 = arith.index_cast %370 : i64 to index
                    ascsimt.store %arg15[%371], %369#0 : memref<8192xi32, 1>, i32
                    %372 = arith.addi %370, %c1_i64_13 : i64
                    %373 = arith.index_cast %372 : i64 to index
                    ascsimt.store %arg15[%373], %369#1 : memref<8192xi32, 1>, i32
                    %374 = arith.addi %370, %c2_i64_12 : i64
                    %375 = arith.index_cast %374 : i64 to index
                    ascsimt.store %arg15[%375], %369#2 : memref<8192xi32, 1>, i32
                    %376 = arith.addi %370, %c3_i64_11 : i64
                    %377 = arith.index_cast %376 : i64 to index
                    ascsimt.store %arg15[%377], %369#3 : memref<8192xi32, 1>, i32
                  } {cannir.vf_threads = 256 : i64}
                } {threads = 256 : i64}
              } else {
                ascsimt.vf(%247, %257, %272, %273, %235, %245, %alloc_8, %274 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                ^bb0(%arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: memref<8192xi32, 1>, %arg14: i64):
                  %c256_i64_10 = arith.constant 256 : i64
                  %c3_i64_11 = arith.constant 3 : i64
                  %c2_i64_12 = arith.constant 2 : i64
                  %c1_i64_13 = arith.constant 1 : i64
                  %c4_i64_14 = arith.constant 4 : i64
                  %c1_i32_15 = arith.constant 1 : i32
                  %c10_i32 = arith.constant 10 : i32
                  %c0_i32_16 = arith.constant 0 : i32
                  %c-1150833019_i32_17 = arith.constant -1150833019 : i32
                  %c-1640531527_i32_18 = arith.constant -1640531527 : i32
                  %c-845247145_i32 = arith.constant -845247145 : i32
                  %c-766435501_i32 = arith.constant -766435501 : i32
                  %c262144_i32 = arith.constant 262144 : i32
                  %c16384_i32 = arith.constant 16384 : i32
                  %285 = ascsimt.thread_idx {ascsimt.scope = 55 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                  %286 = arith.extsi %285 {cannir.vf_threads = 256 : i64} : i32 to i64
                  scf.for %arg15 = %286 to %arg14 step %c256_i64_10  : i64 {
                    %287 = arith.trunci %arg15 : i64 to i32
                    %288 = arith.addi %arg7, %287 : i32
                    %289 = arith.cmpi ult, %288, %arg7 : i32
                    %290 = arith.extui %289 : i1 to i32
                    %291 = arith.addi %arg8, %290 : i32
                    %292 = arith.divui %288, %c16384_i32 : i32
                    %293 = arith.remui %291, %c16384_i32 : i32
                    %294 = arith.muli %293, %c262144_i32 : i32
                    %295 = arith.addi %292, %294 : i32
                    %296 = arith.divui %291, %c16384_i32 : i32
                    %297 = arith.addi %arg9, %295 : i32
                    %298 = arith.addi %arg10, %296 : i32
                    %299 = arith.cmpi ult, %297, %arg9 : i32
                    %300 = arith.extui %299 : i1 to i32
                    %301 = arith.addi %298, %300 : i32
                    %302 = arith.remui %288, %c16384_i32 : i32
                    %303:4 = scf.for %arg16 = %c0_i32_16 to %c10_i32 step %c1_i32_15 iter_args(%arg17 = %c0_i32_16, %arg18 = %302, %arg19 = %297, %arg20 = %301) -> (i32, i32, i32, i32)  : i32 {
                      %312 = arith.muli %arg17, %c-766435501_i32 : i32
                      %313 = "ascsimt.bitcast"(%arg17) : (i32) -> ui32
                      %314 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %315 = ascsimt.math_binary %313, %314, "mulhi" : ui32, ui32 -> ui32
                      %316 = "ascsimt.bitcast"(%315) : (ui32) -> i32
                      %317 = arith.muli %arg19, %c-845247145_i32 : i32
                      %318 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                      %319 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %320 = ascsimt.math_binary %318, %319, "mulhi" : ui32, ui32 -> ui32
                      %321 = "ascsimt.bitcast"(%320) : (ui32) -> i32
                      %322 = arith.muli %arg16, %c-1640531527_i32_18 : i32
                      %323 = arith.addi %arg11, %322 : i32
                      %324 = arith.muli %arg16, %c-1150833019_i32_17 : i32
                      %325 = arith.addi %arg12, %324 : i32
                      %326 = arith.xori %321, %arg18 : i32
                      %327 = arith.xori %326, %323 : i32
                      %328 = arith.xori %316, %arg20 : i32
                      %329 = arith.xori %328, %325 : i32
                      scf.yield %327, %317, %329, %312 : i32, i32, i32, i32
                    }
                    %304 = arith.muli %arg15, %c4_i64_14 : i64
                    %305 = arith.index_cast %304 : i64 to index
                    ascsimt.store %arg13[%305], %303#0 : memref<8192xi32, 1>, i32
                    %306 = arith.addi %304, %c1_i64_13 : i64
                    %307 = arith.index_cast %306 : i64 to index
                    ascsimt.store %arg13[%307], %303#1 : memref<8192xi32, 1>, i32
                    %308 = arith.addi %304, %c2_i64_12 : i64
                    %309 = arith.index_cast %308 : i64 to index
                    ascsimt.store %arg13[%309], %303#2 : memref<8192xi32, 1>, i32
                    %310 = arith.addi %304, %c3_i64_11 : i64
                    %311 = arith.index_cast %310 : i64 to index
                    ascsimt.store %arg13[%311], %303#3 : memref<8192xi32, 1>, i32
                  } {cannir.vf_threads = 256 : i64}
                } {threads = 256 : i64}
              }
              %280 = arith.trunci %226 : i64 to i32
              %281 = arith.trunci %98 : i64 to i32
              %282 = arith.trunci %31 : i64 to i32
              %283 = arith.muli %280, %c2_i32 : i32
              %284 = arith.cmpi eq, %282, %c4_i32 : i32
              ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%98, %133, %alloc_8, %alloc_5, %alloc_7, %280, %281, %283, %284, %alloc_6 : i64, memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, memref<1024xui32, 1>, i32, i32, i32, i1, memref<2048xui8, 1>) {
              ^bb0(%arg7: i64, %arg8: memref<32768xbf16, 1>, %arg9: memref<8192xi32, 1>, %arg10: memref<32768xui8, 1>, %arg11: memref<1024xui32, 1>, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i1, %arg16: memref<2048xui8, 1>):
                %c2139095040_i32 = arith.constant 2139095040 : i32
                %c127_i32 = arith.constant 127 : i32
                %c123_i32 = arith.constant 123 : i32
                %c947912704_i32 = arith.constant 947912704 : i32
                %cst = arith.constant 4.2949673E+9 : f32
                %c29360128_i32 = arith.constant 29360128 : i32
                %c128_i32 = arith.constant 128 : i32
                %c65535_i32 = arith.constant 65535 : i32
                %c2_i32_10 = arith.constant 2 : i32
                %c7_i32 = arith.constant 7 : i32
                %c1_i32_11 = arith.constant 1 : i32
                %c4_i32_12 = arith.constant 4 : i32
                %c2139160576_i32 = arith.constant 2139160576 : i32
                %c254_i32 = arith.constant 254 : i32
                %c-15_i32 = arith.constant -15 : i32
                %c15_i32 = arith.constant 15 : i32
                %c255_i32 = arith.constant 255 : i32
                %c2147483647_i32 = arith.constant 2147483647 : i32
                %cst_13 = arith.constant 0.000000e+00 : f32
                %c0_14 = arith.constant 0 : index
                %c64_i32 = arith.constant 64 : i32
                %c32_i32 = arith.constant 32 : i32
                %c0_i32_15 = arith.constant 0 : i32
                %285 = ascvec.create_mask : !ascvec.mask
                %286 = ascvec.arange %c0_i32_15 : i32 -> <ui32>
                %287 = ascvec.cmp_lt_scalar %286, %c32_i32, %285 : <ui32>, i32 -> !ascvec.mask
                %288 = ascvec.cmp_ge_scalar %286, %c32_i32, %285 : <ui32>, i32 -> !ascvec.mask
                scf.for %arg17 = %c0_i32_15 to %arg12 step %c1_i32_11  : i32 {
                  %290 = arith.muli %arg17, %c64_i32 : i32
                  %291 = arith.trunci %arg7 : i64 to i32
                  %292 = arith.subi %291, %290 : i32
                  %mask, %remaining_out = ascvec.update_mask %292 : !ascvec.mask
                  %293 = arith.index_cast %290 : i32 to index
                  %294 = ascvec.loadalign_unpack %arg8[%293] {mode = "b16_to_b32"} : memref<32768xbf16, 1>, index -> <bf16>
                  %295 = ascvec.cast %294, %285 {rounding = 6 : i32} : <bf16> -> <f32>
                  %296 = ascvec.duplicate_scalar %cst_13, %285 : f32 -> <f32>
                  %297 = ascvec.select %295, %296, %mask : <f32>
                  %298 = ascvec.reinterpret %297 : <f32> -> <ui32>
                  %299 = ascvec.duplicate_scalar %c2147483647_i32, %285 : i32 -> <ui32>
                  %300 = ascvec.and %298, %299, %285 : <ui32>, <ui32> -> <ui32>
                  %301 = ascvec.reduce_max %300, %287 : <ui32>
                  %302 = ascvec.duplicate %301, %285 : <ui32> -> <ui32>
                  %303 = ascvec.reduce_max %300, %288 : <ui32>
                  %304 = ascvec.duplicate %303, %285 : <ui32> -> <ui32>
                  %305 = ascvec.select %302, %304, %287 : <ui32>
                  %306 = ascvec.shr %305, 23, %285 : <ui32> -> <ui32>
                  %307 = ascvec.cmp_ge_scalar %306, %c255_i32, %285 : <ui32>, i32 -> !ascvec.mask
                  %308 = ascvec.max_scalar %306, %c15_i32, %285 : <ui32>, i32
                  %309 = ascvec.add_scalar %308, %c-15_i32, %285 : <ui32>, i32
                  %310 = ascvec.duplicate_scalar %c255_i32, %307 merge %309 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %311 = ascvec.duplicate_scalar %c254_i32, %285 : i32 -> <ui32>
                  %312 = ascvec.sub %311, %310, %285 : <ui32>
                  %313 = ascvec.cmp_ne_scalar %310, %c0_i32_15, %285 : <ui32>, i32 -> !ascvec.mask
                  %314 = ascvec.shl %312, 23, %313 : <ui32> -> <ui32>
                  %315 = ascvec.duplicate_scalar %c2139160576_i32, %307 merge %314 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %316 = ascvec.reinterpret %315 : <ui32> -> <f32>
                  %317 = ascvec.mul %297, %316, %285 : <f32>
                  %318 = ascvec.cast %317, %285 : <f32> -> <bf16>
                  %319 = ascvec.cast %318, %285 {rounding = 6 : i32} : <bf16> -> <f32>
                  %320 = arith.remsi %arg17, %c4_i32_12 : i32
                  %321 = arith.cmpi ne, %320, %c0_i32_15 : i32
                  %322 = arith.cmpi slt, %320, %c0_i32_15 : i32
                  %323 = arith.andi %321, %322 : i1
                  %324 = arith.select %323, %c4_i32_12, %c0_i32_15 : i32
                  %325 = arith.addi %320, %324 overflow<nsw> : i32
                  %326 = arith.muli %325, %c64_i32 overflow<nsw> : i32
                  %327 = ascvec.add_scalar %286, %326, %285 : <ui32>, i32
                  %328 = arith.divsi %arg17, %c4_i32_12 : i32
                  %329 = arith.extui %323 : i1 to i32
                  %330 = arith.subi %328, %329 : i32
                  %331 = ascvec.shr %327, 3, %285 : <ui32> -> <ui32>
                  %332 = arith.muli %330, %c32_i32 : i32
                  %333 = ascvec.add_scalar %331, %332, %285 : <ui32>, i32
                  %334 = ascvec.gather %arg9[%c0_14], %333, %285 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                  %335 = ascvec.reinterpret %334 : <i32> -> <ui32>
                  %336 = ascvec.duplicate_scalar %c7_i32, %285 : i32 -> <ui32>
                  %337 = ascvec.and %327, %336, %285 : <ui32>, <ui32> -> <ui32>
                  %338 = ascvec.shl %335, 11, %285 : <ui32> -> <ui32>
                  %339 = ascvec.shr %335, 21, %285 : <ui32> -> <ui32>
                  %340 = ascvec.or %338, %339, %285 : <ui32>, <ui32> -> <ui32>
                  %341 = ascvec.shr %340, 3, %285 : <ui32> -> <ui32>
                  %342 = ascvec.shl %340, 29, %285 : <ui32> -> <ui32>
                  %343 = ascvec.or %341, %342, %285 : <ui32>, <ui32> -> <ui32>
                  %344 = ascvec.duplicate_scalar %c1_i32_11, %285 : i32 -> <ui32>
                  %345 = ascvec.and %337, %344, %285 : <ui32>, <ui32> -> <ui32>
                  %346 = ascvec.cmp_ne_scalar %345, %c0_i32_15, %285 : <ui32>, i32 -> !ascvec.mask
                  %347 = ascvec.select %343, %340, %346 : <ui32>
                  %348 = ascvec.shr %347, 8, %285 : <ui32> -> <ui32>
                  %349 = ascvec.shl %347, 24, %285 : <ui32> -> <ui32>
                  %350 = ascvec.or %348, %349, %285 : <ui32>, <ui32> -> <ui32>
                  %351 = ascvec.duplicate_scalar %c2_i32_10, %285 : i32 -> <ui32>
                  %352 = ascvec.and %337, %351, %285 : <ui32>, <ui32> -> <ui32>
                  %353 = ascvec.cmp_ne_scalar %352, %c0_i32_15, %285 : <ui32>, i32 -> !ascvec.mask
                  %354 = ascvec.select %350, %347, %353 : <ui32>
                  %355 = ascvec.shr %354, 16, %285 : <ui32> -> <ui32>
                  %356 = ascvec.shl %354, 16, %285 : <ui32> -> <ui32>
                  %357 = ascvec.or %355, %356, %285 : <ui32>, <ui32> -> <ui32>
                  %358 = ascvec.duplicate_scalar %c4_i32_12, %285 : i32 -> <ui32>
                  %359 = ascvec.and %337, %358, %285 : <ui32>, <ui32> -> <ui32>
                  %360 = ascvec.cmp_ne_scalar %359, %c0_i32_15, %285 : <ui32>, i32 -> !ascvec.mask
                  %361 = ascvec.select %357, %354, %360 : <ui32>
                  %362 = ascvec.duplicate_scalar %c65535_i32, %285 : i32 -> <ui32>
                  %363 = ascvec.and %361, %362, %285 : <ui32>, <ui32> -> <ui32>
                  %364 = ascvec.reinterpret %319 : <f32> -> <ui32>
                  %365 = ascvec.shr %364, 24, %285 : <ui32> -> <ui32>
                  %366 = ascvec.duplicate_scalar %c128_i32, %285 : i32 -> <ui32>
                  %367 = ascvec.and %365, %366, %285 : <ui32>, <ui32> -> <ui32>
                  %368 = ascvec.and %364, %299, %285 : <ui32>, <ui32> -> <ui32>
                  %369 = ascvec.shr %368, 5, %285 : <ui32> -> <ui32>
                  %370 = ascvec.duplicate_scalar %c29360128_i32, %285 : i32 -> <ui32>
                  %371 = ascvec.sub %369, %370, %285 : <ui32>
                  %372 = ascvec.reinterpret %368 : <ui32> -> <f32>
                  %373 = ascvec.mul_scalar %372, %cst, %285 : <f32>, f32
                  %374 = ascvec.cast %373, %285 {rounding = 4 : i32} : <f32> -> <i32>
                  %375 = ascvec.reinterpret %374 : <i32> -> <ui32>
                  %376 = ascvec.cmp_ge_scalar %368, %c947912704_i32, %285 : <ui32>, i32 -> !ascvec.mask
                  %377 = ascvec.select %371, %375, %376 : <ui32>
                  %378 = ascvec.and %363, %362, %285 : <ui32>, <ui32> -> <ui32>
                  %379 = ascvec.add %377, %378, %285 : <ui32>
                  %380 = ascvec.shr %379, 16, %285 : <ui32> -> <ui32>
                  %381 = ascvec.min_scalar %380, %c123_i32, %285 : <ui32>, i32
                  %382 = ascvec.or %381, %367, %285 : <ui32>, <ui32> -> <ui32>
                  %383 = ascvec.duplicate_scalar %c127_i32, %285 : i32 -> <ui32>
                  %384 = ascvec.cmp_gt_scalar %368, %c2139095040_i32, %285 : <ui32>, i32 -> !ascvec.mask
                  %385 = ascvec.select %383, %382, %384 : <ui32>
                  ascvec.storealign_pack %arg10[%293], %385, %mask {mode = "b32_to_b8"} : memref<32768xui8, 1>, index, <ui32>
                  %386 = arith.muli %arg17, %c2_i32_10 : i32
                  %387 = arith.index_cast %386 : i32 to index
                  ascvec.storealign_1st %arg11[%387], %310 : memref<1024xui32, 1>, index, <ui32>
                  %388 = ascvec.duplicate_scalar %c32_i32, %285 : i32 -> <ui32>
                  %389 = ascvec.gather_reg %310, %388 : <ui32>, <ui32> -> <ui32>
                  %390 = arith.addi %386, %c1_i32_11 : i32
                  %391 = arith.index_cast %390 : i32 to index
                  ascvec.storealign_1st %arg11[%391], %389 : memref<1024xui32, 1>, index, <ui32>
                }
                ascvec.mem_bar vst_vld
                %289 = scf.if %arg15 -> (i32) {
                  %290 = arith.remsi %arg13, %c32_i32 : i32
                  %291 = arith.cmpi ne, %290, %c0_i32_15 : i32
                  %292 = arith.cmpi slt, %290, %c0_i32_15 : i32
                  %293 = arith.andi %291, %292 : i1
                  %294 = arith.divsi %arg13, %c32_i32 : i32
                  %295 = arith.extui %293 : i1 to i32
                  %296 = arith.subi %294, %295 : i32
                  %297 = arith.muli %296, %c2_i32_10 : i32
                  scf.yield %297 : i32
                } else {
                  scf.yield %arg14 : i32
                }
                scf.for %arg17 = %c0_i32_15 to %289 step %c64_i32  : i32 {
                  %290 = arith.subi %289, %arg17 : i32
                  %mask, %remaining_out = ascvec.update_mask %290 : !ascvec.mask
                  %291 = ascvec.add_scalar %286, %arg17, %285 : <ui32>, i32
                  scf.if %arg15 {
                    %292 = ascvec.shr %291, 1, %285 : <ui32> -> <ui32>
                    %293 = ascvec.gather %arg11[%c0_14], %292, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %294 = ascvec.duplicate_scalar %c1_i32_11, %285 : i32 -> <ui32>
                    %295 = ascvec.and %286, %294, %285 : <ui32>, <ui32> -> <ui32>
                    %296 = ascvec.cmp_eq_scalar %295, %c0_i32_15, %285 : <ui32>, i32 -> !ascvec.mask
                    %297 = ascvec.duplicate_scalar %c0_i32_15, %285 : i32 -> <ui32>
                    %298 = ascvec.select %293, %297, %296 : <ui32>
                    %299 = arith.index_cast %arg17 : i32 to index
                    %c0_16 = arith.constant 0 : index
                    %c1_17 = arith.constant 1 : index
                    scf.for %arg18 = %c0_16 to %c1_17 step %c1_17 {
                      ascvec.storealign_pack %arg16[%299], %298, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  } else {
                    %292 = ascvec.gather %arg11[%c0_14], %291, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %293 = arith.index_cast %arg17 : i32 to index
                    %c0_16 = arith.constant 0 : index
                    %c1_17 = arith.constant 1 : index
                    scf.for %arg18 = %c0_16 to %c1_17 step %c1_17 {
                      ascvec.storealign_pack %arg16[%293], %292, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  }
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
            } else {
              %219 = arith.addi %98, %c63_i64 : i64
              %220 = arith.remsi %219, %c64_i64 : i64
              %221 = arith.cmpi ne, %220, %c0_i64 : i64
              %222 = arith.cmpi slt, %220, %c0_i64 : i64
              %223 = arith.andi %221, %222 : i1
              %224 = arith.divsi %219, %c64_i64 : i64
              %225 = arith.extui %223 : i1 to i64
              %226 = arith.subi %224, %225 : i64
              %227 = arith.addi %98, %c1023_i64 : i64
              %228 = arith.remsi %227, %c1024_i64 : i64
              %229 = arith.cmpi ne, %228, %c0_i64 : i64
              %230 = arith.cmpi slt, %228, %c0_i64 : i64
              %231 = arith.andi %229, %230 : i1
              %232 = arith.divsi %227, %c1024_i64 : i64
              %233 = arith.extui %231 : i1 to i64
              %234 = arith.subi %232, %233 : i64
              %235 = arith.trunci %1 : i64 to i32
              %236 = arith.extui %235 : i32 to i64
              %237 = arith.subi %1, %236 : i64
              %238 = arith.remsi %237, %c4294967296_i64 : i64
              %239 = arith.cmpi ne, %238, %c0_i64 : i64
              %240 = arith.cmpi slt, %238, %c0_i64 : i64
              %241 = arith.andi %239, %240 : i1
              %242 = arith.divsi %237, %c4294967296_i64 : i64
              %243 = arith.extui %241 : i1 to i64
              %244 = arith.subi %242, %243 : i64
              %245 = arith.trunci %244 : i64 to i32
              %246 = arith.remsi %211, %c16_i64 : i64
              %247 = arith.cmpi ne, %246, %c0_i64 : i64
              %248 = arith.cmpi slt, %246, %c0_i64 : i64
              %249 = arith.andi %247, %248 : i1
              %250 = arith.divsi %211, %c16_i64 : i64
              %251 = arith.extui %249 : i1 to i64
              %252 = arith.subi %250, %251 : i64
              %253 = arith.remsi %arg3, %c16_i64 : i64
              %254 = arith.cmpi ne, %253, %c0_i64 : i64
              %255 = arith.cmpi slt, %253, %c0_i64 : i64
              %256 = arith.andi %254, %255 : i1
              %257 = arith.divsi %arg3, %c16_i64 : i64
              %258 = arith.extui %256 : i1 to i64
              %259 = arith.subi %257, %258 : i64
              %260 = arith.remsi %252, %259 : i64
              %261 = arith.cmpi ne, %260, %c0_i64 : i64
              %262 = arith.cmpi slt, %260, %c0_i64 : i64
              %263 = arith.cmpi slt, %259, %c0_i64 : i64
              %264 = arith.xori %262, %263 : i1
              %265 = arith.andi %261, %264 : i1
              %266 = arith.divsi %252, %259 : i64
              %267 = arith.extui %265 : i1 to i64
              %268 = arith.subi %266, %267 : i64
              %269 = arith.select %265, %259, %c0_i64 : i64
              %270 = arith.addi %260, %269 : i64
              %271 = arith.trunci %268 : i64 to i32
              %272 = arith.extui %271 : i32 to i64
              %273 = arith.subi %268, %272 : i64
              %274 = arith.remsi %273, %c4294967296_i64 : i64
              %275 = arith.cmpi ne, %274, %c0_i64 : i64
              %276 = arith.cmpi slt, %274, %c0_i64 : i64
              %277 = arith.andi %275, %276 : i1
              %278 = arith.divsi %273, %c4294967296_i64 : i64
              %279 = arith.extui %277 : i1 to i64
              %280 = arith.subi %278, %279 : i64
              %281 = arith.trunci %280 : i64 to i32
              %282 = arith.trunci %270 : i64 to i32
              %283 = arith.extui %282 : i32 to i64
              %284 = arith.subi %270, %283 : i64
              %285 = arith.remsi %284, %c4294967296_i64 : i64
              %286 = arith.cmpi ne, %285, %c0_i64 : i64
              %287 = arith.cmpi slt, %285, %c0_i64 : i64
              %288 = arith.andi %286, %287 : i1
              %289 = arith.divsi %284, %c4294967296_i64 : i64
              %290 = arith.extui %288 : i1 to i64
              %291 = arith.subi %289, %290 : i64
              %292 = arith.trunci %291 : i64 to i32
              %293 = arith.remsi %c64_i64, %259 : i64
              %294 = arith.cmpi ne, %293, %c0_i64 : i64
              %295 = arith.cmpi slt, %293, %c0_i64 : i64
              %296 = arith.xori %295, %263 : i1
              %297 = arith.andi %294, %296 : i1
              %298 = arith.divsi %c64_i64, %259 : i64
              %299 = arith.extui %297 : i1 to i64
              %300 = arith.subi %298, %299 : i64
              %301 = arith.select %297, %259, %c0_i64 : i64
              %302 = arith.addi %293, %301 : i64
              %303 = arith.trunci %234 : i64 to i32
              ascvec.vf(%259, %3, %235, %245, %alloc_8, %302, %300, %303, %271, %281, %282, %292 : i64, i64, i32, i32, memref<8192xi32, 1>, i64, i64, i32, i32, i32, i32, i32) {
              ^bb0(%arg7: i64, %arg8: i64, %arg9: i32, %arg10: i32, %arg11: memref<8192xi32, 1>, %arg12: i64, %arg13: i64, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32):
                %c1_i32_10 = arith.constant 1 : i32
                %c192_i32 = arith.constant 192 : i32
                %c128_i32 = arith.constant 128 : i32
                %c64_i32 = arith.constant 64 : i32
                %c-1767562579_i32 = arith.constant -1767562579 : i32
                %c-1879881855_i32 = arith.constant -1879881855 : i32
                %c-616729560_i32 = arith.constant -616729560 : i32
                %c-239350328_i32 = arith.constant -239350328 : i32
                %c534103459_i32 = arith.constant 534103459 : i32
                %c1401181199_i32 = arith.constant 1401181199 : i32
                %c1684936478_i32 = arith.constant 1684936478 : i32
                %c-1253254570_i32 = arith.constant -1253254570 : i32
                %c-1459197799_i32 = arith.constant -1459197799 : i32
                %c387276957_i32 = arith.constant 387276957 : i32
                %c-308364780_i32 = arith.constant -308364780 : i32
                %c2027808484_i32 = arith.constant 2027808484 : i32
                %c842468239_i32 = arith.constant 842468239 : i32
                %c-626627285_i32 = arith.constant -626627285 : i32
                %c1993301258_i32_11 = arith.constant 1993301258 : i32
                %c1013904242_i32_12 = arith.constant 1013904242 : i32
                %c-1150833019_i32_13 = arith.constant -1150833019 : i32
                %c-1640531527_i32_14 = arith.constant -1640531527 : i32
                %c-845247145_i32 = arith.constant -845247145 : i32
                %c-766435501_i32 = arith.constant -766435501 : i32
                %c16383_i32 = arith.constant 16383 : i32
                %c1073741824_i32_15 = arith.constant 1073741824 : i32
                %c4_i32_16 = arith.constant 4 : i32
                %c0_i64_17 = arith.constant 0 : i64
                %c0_i32_18 = arith.constant 0 : i32
                %c256_i32 = arith.constant 256 : i32
                %c4294967296_i64_19 = arith.constant 4294967296 : i64
                %309:4 = scf.for %arg19 = %c0_i32_18 to %arg14 step %c1_i32_10 iter_args(%arg20 = %arg15, %arg21 = %arg16, %arg22 = %arg17, %arg23 = %arg18) -> (i32, i32, i32, i32)  : i32 {
                  %310 = arith.extui %arg20 : i32 to i64
                  %311 = arith.extui %arg21 : i32 to i64
                  %312 = arith.muli %311, %c4294967296_i64_19 : i64
                  %313 = arith.addi %310, %312 : i64
                  %314 = arith.extui %arg22 : i32 to i64
                  %315 = arith.extui %arg23 : i32 to i64
                  %316 = arith.muli %315, %c4294967296_i64_19 : i64
                  %317 = arith.addi %314, %316 : i64
                  %318 = arith.muli %arg19, %c256_i32 : i32
                  %319 = ascvec.create_mask : !ascvec.mask
                  %320 = ascvec.arange %c0_i32_18 : i32 -> <ui32>
                  %321 = arith.muli %313, %arg7 : i64
                  %322 = arith.addi %321, %317 : i64
                  %323 = arith.trunci %322 : i64 to i32
                  %324 = arith.extui %323 : i32 to i64
                  %325 = arith.subi %322, %324 : i64
                  %326 = arith.remsi %325, %c4294967296_i64_19 : i64
                  %327 = arith.cmpi ne, %326, %c0_i64_17 : i64
                  %328 = arith.cmpi slt, %326, %c0_i64_17 : i64
                  %329 = arith.andi %327, %328 : i1
                  %330 = arith.divsi %325, %c4294967296_i64_19 : i64
                  %331 = arith.extui %329 : i1 to i64
                  %332 = arith.subi %330, %331 : i64
                  %333 = arith.trunci %332 : i64 to i32
                  %334 = ascvec.duplicate_scalar %323, %319 : i32 -> <ui32>
                  %335 = ascvec.duplicate_scalar %333, %319 : i32 -> <ui32>
                  %336 = ascvec.duplicate_scalar %c0_i32_18, %319 : i32 -> <ui32>
                  %carry, %dst = ascvec.addc %334, %320, %319 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %carry_20, %dst_21 = ascvec.addc %335, %336, %carry, %319 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %337 = arith.trunci %arg8 : i64 to i32
                  %338 = arith.extui %337 : i32 to i64
                  %339 = arith.subi %arg8, %338 : i64
                  %340 = arith.remsi %339, %c4294967296_i64_19 : i64
                  %341 = arith.cmpi ne, %340, %c0_i64_17 : i64
                  %342 = arith.cmpi slt, %340, %c0_i64_17 : i64
                  %343 = arith.andi %341, %342 : i1
                  %344 = arith.divsi %339, %c4294967296_i64_19 : i64
                  %345 = arith.extui %343 : i1 to i64
                  %346 = arith.subi %344, %345 : i64
                  %347 = arith.trunci %346 : i64 to i32
                  %348 = arith.divui %337, %c4_i32_16 : i32
                  %349 = arith.remui %347, %c4_i32_16 : i32
                  %350 = arith.muli %349, %c1073741824_i32_15 : i32
                  %351 = arith.addi %348, %350 : i32
                  %352 = arith.divui %347, %c4_i32_16 : i32
                  %353 = ascvec.shr %dst, 14, %319 : <ui32> -> <ui32>
                  %354 = ascvec.shl %dst_21, 18, %319 : <ui32> -> <ui32>
                  %355 = ascvec.or %353, %354, %319 : <ui32>, <ui32> -> <ui32>
                  %356 = ascvec.shr %dst_21, 14, %319 : <ui32> -> <ui32>
                  %357 = ascvec.duplicate_scalar %351, %319 : i32 -> <ui32>
                  %358 = ascvec.duplicate_scalar %352, %319 : i32 -> <ui32>
                  %carry_22, %dst_23 = ascvec.addc %357, %355, %319 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %carry_24, %dst_25 = ascvec.addc %358, %356, %carry_22, %319 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %359 = ascvec.duplicate_scalar %c16383_i32, %319 : i32 -> <ui32>
                  %360 = ascvec.and %dst, %359, %319 : <ui32>, <ui32> -> <ui32>
                  %361 = ascvec.duplicate_scalar %c-766435501_i32, %319 : i32 -> <ui32>
                  %362 = ascvec.duplicate_scalar %c-845247145_i32, %319 : i32 -> <ui32>
                  %dst0, %dst1 = ascvec.mull %336, %361, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_26, %dst1_27 = ascvec.mull %dst_23, %362, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %363 = ascvec.xor %dst1_27, %360, %319 : <ui32>, <ui32> -> <ui32>
                  %364 = ascvec.xor %dst1, %dst_25, %319 : <ui32>, <ui32> -> <ui32>
                  %365 = ascvec.duplicate_scalar %arg9, %319 : i32 -> <ui32>
                  %366 = ascvec.duplicate_scalar %arg10, %319 : i32 -> <ui32>
                  %367 = ascvec.xor %363, %365, %319 : <ui32>, <ui32> -> <ui32>
                  %368 = ascvec.xor %364, %366, %319 : <ui32>, <ui32> -> <ui32>
                  %dst0_28, %dst1_29 = ascvec.mull %367, %361, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_30, %dst1_31 = ascvec.mull %368, %362, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %369 = ascvec.xor %dst1_31, %dst0_26, %319 : <ui32>, <ui32> -> <ui32>
                  %370 = ascvec.xor %dst1_29, %dst0, %319 : <ui32>, <ui32> -> <ui32>
                  %371 = arith.addi %arg9, %c-1640531527_i32_14 : i32
                  %372 = ascvec.duplicate_scalar %371, %319 : i32 -> <ui32>
                  %373 = arith.addi %arg10, %c-1150833019_i32_13 : i32
                  %374 = ascvec.duplicate_scalar %373, %319 : i32 -> <ui32>
                  %375 = ascvec.xor %369, %372, %319 : <ui32>, <ui32> -> <ui32>
                  %376 = ascvec.xor %370, %374, %319 : <ui32>, <ui32> -> <ui32>
                  %dst0_32, %dst1_33 = ascvec.mull %375, %361, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_34, %dst1_35 = ascvec.mull %376, %362, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %377 = ascvec.xor %dst1_35, %dst0_30, %319 : <ui32>, <ui32> -> <ui32>
                  %378 = ascvec.xor %dst1_33, %dst0_28, %319 : <ui32>, <ui32> -> <ui32>
                  %379 = arith.addi %arg9, %c1013904242_i32_12 : i32
                  %380 = ascvec.duplicate_scalar %379, %319 : i32 -> <ui32>
                  %381 = arith.addi %arg10, %c1993301258_i32_11 : i32
                  %382 = ascvec.duplicate_scalar %381, %319 : i32 -> <ui32>
                  %383 = ascvec.xor %377, %380, %319 : <ui32>, <ui32> -> <ui32>
                  %384 = ascvec.xor %378, %382, %319 : <ui32>, <ui32> -> <ui32>
                  %dst0_36, %dst1_37 = ascvec.mull %383, %361, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_38, %dst1_39 = ascvec.mull %384, %362, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %385 = ascvec.xor %dst1_39, %dst0_34, %319 : <ui32>, <ui32> -> <ui32>
                  %386 = ascvec.xor %dst1_37, %dst0_32, %319 : <ui32>, <ui32> -> <ui32>
                  %387 = arith.addi %arg9, %c-626627285_i32 : i32
                  %388 = ascvec.duplicate_scalar %387, %319 : i32 -> <ui32>
                  %389 = arith.addi %arg10, %c842468239_i32 : i32
                  %390 = ascvec.duplicate_scalar %389, %319 : i32 -> <ui32>
                  %391 = ascvec.xor %385, %388, %319 : <ui32>, <ui32> -> <ui32>
                  %392 = ascvec.xor %386, %390, %319 : <ui32>, <ui32> -> <ui32>
                  %dst0_40, %dst1_41 = ascvec.mull %391, %361, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_42, %dst1_43 = ascvec.mull %392, %362, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %393 = ascvec.xor %dst1_43, %dst0_38, %319 : <ui32>, <ui32> -> <ui32>
                  %394 = ascvec.xor %dst1_41, %dst0_36, %319 : <ui32>, <ui32> -> <ui32>
                  %395 = arith.addi %arg9, %c2027808484_i32 : i32
                  %396 = ascvec.duplicate_scalar %395, %319 : i32 -> <ui32>
                  %397 = arith.addi %arg10, %c-308364780_i32 : i32
                  %398 = ascvec.duplicate_scalar %397, %319 : i32 -> <ui32>
                  %399 = ascvec.xor %393, %396, %319 : <ui32>, <ui32> -> <ui32>
                  %400 = ascvec.xor %394, %398, %319 : <ui32>, <ui32> -> <ui32>
                  %dst0_44, %dst1_45 = ascvec.mull %399, %361, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_46, %dst1_47 = ascvec.mull %400, %362, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %401 = ascvec.xor %dst1_47, %dst0_42, %319 : <ui32>, <ui32> -> <ui32>
                  %402 = ascvec.xor %dst1_45, %dst0_40, %319 : <ui32>, <ui32> -> <ui32>
                  %403 = arith.addi %arg9, %c387276957_i32 : i32
                  %404 = ascvec.duplicate_scalar %403, %319 : i32 -> <ui32>
                  %405 = arith.addi %arg10, %c-1459197799_i32 : i32
                  %406 = ascvec.duplicate_scalar %405, %319 : i32 -> <ui32>
                  %407 = ascvec.xor %401, %404, %319 : <ui32>, <ui32> -> <ui32>
                  %408 = ascvec.xor %402, %406, %319 : <ui32>, <ui32> -> <ui32>
                  %dst0_48, %dst1_49 = ascvec.mull %407, %361, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_50, %dst1_51 = ascvec.mull %408, %362, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %409 = ascvec.xor %dst1_51, %dst0_46, %319 : <ui32>, <ui32> -> <ui32>
                  %410 = ascvec.xor %dst1_49, %dst0_44, %319 : <ui32>, <ui32> -> <ui32>
                  %411 = arith.addi %arg9, %c-1253254570_i32 : i32
                  %412 = ascvec.duplicate_scalar %411, %319 : i32 -> <ui32>
                  %413 = arith.addi %arg10, %c1684936478_i32 : i32
                  %414 = ascvec.duplicate_scalar %413, %319 : i32 -> <ui32>
                  %415 = ascvec.xor %409, %412, %319 : <ui32>, <ui32> -> <ui32>
                  %416 = ascvec.xor %410, %414, %319 : <ui32>, <ui32> -> <ui32>
                  %dst0_52, %dst1_53 = ascvec.mull %415, %361, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_54, %dst1_55 = ascvec.mull %416, %362, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %417 = ascvec.xor %dst1_55, %dst0_50, %319 : <ui32>, <ui32> -> <ui32>
                  %418 = ascvec.xor %dst1_53, %dst0_48, %319 : <ui32>, <ui32> -> <ui32>
                  %419 = arith.addi %arg9, %c1401181199_i32 : i32
                  %420 = ascvec.duplicate_scalar %419, %319 : i32 -> <ui32>
                  %421 = arith.addi %arg10, %c534103459_i32 : i32
                  %422 = ascvec.duplicate_scalar %421, %319 : i32 -> <ui32>
                  %423 = ascvec.xor %417, %420, %319 : <ui32>, <ui32> -> <ui32>
                  %424 = ascvec.xor %418, %422, %319 : <ui32>, <ui32> -> <ui32>
                  %dst0_56, %dst1_57 = ascvec.mull %423, %361, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_58, %dst1_59 = ascvec.mull %424, %362, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %425 = ascvec.xor %dst1_59, %dst0_54, %319 : <ui32>, <ui32> -> <ui32>
                  %426 = ascvec.xor %dst1_57, %dst0_52, %319 : <ui32>, <ui32> -> <ui32>
                  %427 = arith.addi %arg9, %c-239350328_i32 : i32
                  %428 = ascvec.duplicate_scalar %427, %319 : i32 -> <ui32>
                  %429 = arith.addi %arg10, %c-616729560_i32 : i32
                  %430 = ascvec.duplicate_scalar %429, %319 : i32 -> <ui32>
                  %431 = ascvec.xor %425, %428, %319 : <ui32>, <ui32> -> <ui32>
                  %432 = ascvec.xor %426, %430, %319 : <ui32>, <ui32> -> <ui32>
                  %dst0_60, %dst1_61 = ascvec.mull %431, %361, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_62, %dst1_63 = ascvec.mull %432, %362, %319 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %433 = ascvec.xor %dst1_63, %dst0_58, %319 : <ui32>, <ui32> -> <ui32>
                  %434 = ascvec.xor %dst1_61, %dst0_56, %319 : <ui32>, <ui32> -> <ui32>
                  %435 = arith.addi %arg9, %c-1879881855_i32 : i32
                  %436 = ascvec.duplicate_scalar %435, %319 : i32 -> <ui32>
                  %437 = arith.addi %arg10, %c-1767562579_i32 : i32
                  %438 = ascvec.duplicate_scalar %437, %319 : i32 -> <ui32>
                  %439 = ascvec.xor %433, %436, %319 : <ui32>, <ui32> -> <ui32>
                  %440 = ascvec.xor %434, %438, %319 : <ui32>, <ui32> -> <ui32>
                  %441 = ascvec.reinterpret %439 : <ui32> -> <i32>
                  %442 = arith.index_cast %318 : i32 to index
                  ascvec.storealign %arg11[%442], %441, %319 : memref<8192xi32, 1>, index, <i32>
                  %443 = arith.addi %318, %c64_i32 : i32
                  %444 = ascvec.reinterpret %dst0_62 : <ui32> -> <i32>
                  %445 = arith.index_cast %443 : i32 to index
                  ascvec.storealign %arg11[%445], %444, %319 : memref<8192xi32, 1>, index, <i32>
                  %446 = arith.addi %318, %c128_i32 : i32
                  %447 = ascvec.reinterpret %440 : <ui32> -> <i32>
                  %448 = arith.index_cast %446 : i32 to index
                  ascvec.storealign %arg11[%448], %447, %319 : memref<8192xi32, 1>, index, <i32>
                  %449 = arith.addi %318, %c192_i32 : i32
                  %450 = ascvec.reinterpret %dst0_60 : <ui32> -> <i32>
                  %451 = arith.index_cast %449 : i32 to index
                  ascvec.storealign %arg11[%451], %450, %319 : memref<8192xi32, 1>, index, <i32>
                  %452 = arith.addi %317, %arg12 : i64
                  %453 = arith.cmpi sge, %452, %arg7 : i64
                  %454 = arith.extui %453 : i1 to i64
                  %455 = arith.addi %313, %arg13 : i64
                  %456 = arith.addi %455, %454 : i64
                  %457 = arith.muli %454, %arg7 : i64
                  %458 = arith.subi %452, %457 : i64
                  %459 = arith.trunci %456 : i64 to i32
                  %460 = arith.extui %459 : i32 to i64
                  %461 = arith.subi %456, %460 : i64
                  %462 = arith.remsi %461, %c4294967296_i64_19 : i64
                  %463 = arith.cmpi ne, %462, %c0_i64_17 : i64
                  %464 = arith.cmpi slt, %462, %c0_i64_17 : i64
                  %465 = arith.andi %463, %464 : i1
                  %466 = arith.divsi %461, %c4294967296_i64_19 : i64
                  %467 = arith.extui %465 : i1 to i64
                  %468 = arith.subi %466, %467 : i64
                  %469 = arith.trunci %468 : i64 to i32
                  %470 = arith.trunci %458 : i64 to i32
                  %471 = arith.extui %470 : i32 to i64
                  %472 = arith.subi %458, %471 : i64
                  %473 = arith.remsi %472, %c4294967296_i64_19 : i64
                  %474 = arith.cmpi ne, %473, %c0_i64_17 : i64
                  %475 = arith.cmpi slt, %473, %c0_i64_17 : i64
                  %476 = arith.andi %474, %475 : i1
                  %477 = arith.divsi %472, %c4294967296_i64_19 : i64
                  %478 = arith.extui %476 : i1 to i64
                  %479 = arith.subi %477, %478 : i64
                  %480 = arith.trunci %479 : i64 to i32
                  scf.yield %459, %469, %470, %480 : i32, i32, i32, i32
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              %304 = arith.trunci %226 : i64 to i32
              %305 = arith.trunci %98 : i64 to i32
              %306 = arith.trunci %31 : i64 to i32
              %307 = arith.muli %304, %c2_i32 : i32
              %308 = arith.cmpi eq, %306, %c4_i32 : i32
              ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%98, %133, %alloc_8, %alloc_5, %alloc_7, %304, %305, %307, %308, %alloc_6 : i64, memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, memref<1024xui32, 1>, i32, i32, i32, i1, memref<2048xui8, 1>) {
              ^bb0(%arg7: i64, %arg8: memref<32768xbf16, 1>, %arg9: memref<8192xi32, 1>, %arg10: memref<32768xui8, 1>, %arg11: memref<1024xui32, 1>, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i1, %arg16: memref<2048xui8, 1>):
                %c2_i32_10 = arith.constant 2 : i32
                %c2139095040_i32 = arith.constant 2139095040 : i32
                %c127_i32 = arith.constant 127 : i32
                %c123_i32 = arith.constant 123 : i32
                %c947912704_i32 = arith.constant 947912704 : i32
                %cst = arith.constant 4.2949673E+9 : f32
                %c29360128_i32 = arith.constant 29360128 : i32
                %c128_i32 = arith.constant 128 : i32
                %c65535_i32 = arith.constant 65535 : i32
                %c4_i32_11 = arith.constant 4 : i32
                %c256_i32 = arith.constant 256 : i32
                %c1_i32_12 = arith.constant 1 : i32
                %c16_i32 = arith.constant 16 : i32
                %c3_i32 = arith.constant 3 : i32
                %c2139160576_i32 = arith.constant 2139160576 : i32
                %c254_i32 = arith.constant 254 : i32
                %c-15_i32 = arith.constant -15 : i32
                %c15_i32 = arith.constant 15 : i32
                %c255_i32 = arith.constant 255 : i32
                %c2147483647_i32 = arith.constant 2147483647 : i32
                %cst_13 = arith.constant 0.000000e+00 : f32
                %c0_14 = arith.constant 0 : index
                %c64_i32 = arith.constant 64 : i32
                %c32_i32 = arith.constant 32 : i32
                %c0_i32_15 = arith.constant 0 : i32
                %309 = ascvec.create_mask : !ascvec.mask
                %310 = ascvec.arange %c0_i32_15 : i32 -> <ui32>
                %311 = ascvec.cmp_lt_scalar %310, %c32_i32, %309 : <ui32>, i32 -> !ascvec.mask
                %312 = ascvec.cmp_ge_scalar %310, %c32_i32, %309 : <ui32>, i32 -> !ascvec.mask
                scf.for %arg17 = %c0_i32_15 to %arg12 step %c1_i32_12  : i32 {
                  %314 = arith.muli %arg17, %c64_i32 : i32
                  %315 = arith.trunci %arg7 : i64 to i32
                  %316 = arith.subi %315, %314 : i32
                  %mask, %remaining_out = ascvec.update_mask %316 : !ascvec.mask
                  %317 = arith.index_cast %314 : i32 to index
                  %318 = ascvec.loadalign_unpack %arg8[%317] {mode = "b16_to_b32"} : memref<32768xbf16, 1>, index -> <bf16>
                  %319 = ascvec.cast %318, %309 {rounding = 6 : i32} : <bf16> -> <f32>
                  %320 = ascvec.duplicate_scalar %cst_13, %309 : f32 -> <f32>
                  %321 = ascvec.select %319, %320, %mask : <f32>
                  %322 = ascvec.reinterpret %321 : <f32> -> <ui32>
                  %323 = ascvec.duplicate_scalar %c2147483647_i32, %309 : i32 -> <ui32>
                  %324 = ascvec.and %322, %323, %309 : <ui32>, <ui32> -> <ui32>
                  %325 = ascvec.reduce_max %324, %311 : <ui32>
                  %326 = ascvec.duplicate %325, %309 : <ui32> -> <ui32>
                  %327 = ascvec.reduce_max %324, %312 : <ui32>
                  %328 = ascvec.duplicate %327, %309 : <ui32> -> <ui32>
                  %329 = ascvec.select %326, %328, %311 : <ui32>
                  %330 = ascvec.shr %329, 23, %309 : <ui32> -> <ui32>
                  %331 = ascvec.cmp_ge_scalar %330, %c255_i32, %309 : <ui32>, i32 -> !ascvec.mask
                  %332 = ascvec.max_scalar %330, %c15_i32, %309 : <ui32>, i32
                  %333 = ascvec.add_scalar %332, %c-15_i32, %309 : <ui32>, i32
                  %334 = ascvec.duplicate_scalar %c255_i32, %331 merge %333 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %335 = ascvec.duplicate_scalar %c254_i32, %309 : i32 -> <ui32>
                  %336 = ascvec.sub %335, %334, %309 : <ui32>
                  %337 = ascvec.cmp_ne_scalar %334, %c0_i32_15, %309 : <ui32>, i32 -> !ascvec.mask
                  %338 = ascvec.shl %336, 23, %337 : <ui32> -> <ui32>
                  %339 = ascvec.duplicate_scalar %c2139160576_i32, %331 merge %338 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %340 = ascvec.reinterpret %339 : <ui32> -> <f32>
                  %341 = ascvec.mul %321, %340, %309 : <f32>
                  %342 = ascvec.shr %310, 2, %309 : <ui32> -> <ui32>
                  %343 = ascvec.duplicate_scalar %c3_i32, %309 : i32 -> <ui32>
                  %344 = ascvec.and %342, %343, %309 : <ui32>, <ui32> -> <ui32>
                  %345 = ascvec.shl %344, 6, %309 : <ui32> -> <ui32>
                  %346 = ascvec.shr %310, 4, %309 : <ui32> -> <ui32>
                  %347 = ascvec.add %345, %346, %309 : <ui32>
                  %348 = arith.remsi %arg17, %c16_i32 : i32
                  %349 = arith.cmpi ne, %348, %c0_i32_15 : i32
                  %350 = arith.cmpi slt, %348, %c0_i32_15 : i32
                  %351 = arith.andi %349, %350 : i1
                  %352 = arith.divsi %arg17, %c16_i32 : i32
                  %353 = arith.extui %351 : i1 to i32
                  %354 = arith.subi %352, %353 : i32
                  %355 = arith.muli %354, %c256_i32 : i32
                  %356 = arith.select %351, %c16_i32, %c0_i32_15 : i32
                  %357 = arith.addi %348, %356 overflow<nsw> : i32
                  %358 = arith.muli %357, %c4_i32_11 overflow<nsw> : i32
                  %359 = arith.addi %355, %358 : i32
                  %360 = ascvec.add_scalar %347, %359, %309 : <ui32>, i32
                  %361 = ascvec.gather %arg9[%c0_14], %360, %309 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                  %362 = ascvec.reinterpret %361 : <i32> -> <ui32>
                  %363 = ascvec.shl %362, 8, %309 : <ui32> -> <ui32>
                  %364 = ascvec.shr %362, 24, %309 : <ui32> -> <ui32>
                  %365 = ascvec.or %363, %364, %309 : <ui32>, <ui32> -> <ui32>
                  %366 = ascvec.and %310, %343, %309 : <ui32>, <ui32> -> <ui32>
                  %367 = ascvec.duplicate_scalar %c1_i32_12, %309 : i32 -> <ui32>
                  %368 = ascvec.and %366, %367, %309 : <ui32>, <ui32> -> <ui32>
                  %369 = ascvec.cmp_eq_scalar %368, %c0_i32_15, %309 : <ui32>, i32 -> !ascvec.mask
                  %370 = ascvec.select %362, %365, %369 : <ui32>
                  %371 = ascvec.cmp_eq_scalar %366, %c0_i32_15, %309 : <ui32>, i32 -> !ascvec.mask
                  %372 = ascvec.cmp_eq_scalar %366, %c3_i32, %309 : <ui32>, i32 -> !ascvec.mask
                  %373 = ascvec.mask_or %371, %372, %309
                  %374 = ascvec.shr %370, 16, %309 : <ui32> -> <ui32>
                  %375 = ascvec.duplicate_scalar %c65535_i32, %309 : i32 -> <ui32>
                  %376 = ascvec.and %370, %375, %309 : <ui32>, <ui32> -> <ui32>
                  %377 = ascvec.select %374, %376, %373 : <ui32>
                  %378 = ascvec.reinterpret %341 : <f32> -> <ui32>
                  %379 = ascvec.shr %378, 24, %309 : <ui32> -> <ui32>
                  %380 = ascvec.duplicate_scalar %c128_i32, %309 : i32 -> <ui32>
                  %381 = ascvec.and %379, %380, %309 : <ui32>, <ui32> -> <ui32>
                  %382 = ascvec.and %378, %323, %309 : <ui32>, <ui32> -> <ui32>
                  %383 = ascvec.shr %382, 5, %309 : <ui32> -> <ui32>
                  %384 = ascvec.duplicate_scalar %c29360128_i32, %309 : i32 -> <ui32>
                  %385 = ascvec.sub %383, %384, %309 : <ui32>
                  %386 = ascvec.reinterpret %382 : <ui32> -> <f32>
                  %387 = ascvec.mul_scalar %386, %cst, %309 : <f32>, f32
                  %388 = ascvec.cast %387, %309 {rounding = 4 : i32} : <f32> -> <i32>
                  %389 = ascvec.reinterpret %388 : <i32> -> <ui32>
                  %390 = ascvec.cmp_ge_scalar %382, %c947912704_i32, %309 : <ui32>, i32 -> !ascvec.mask
                  %391 = ascvec.select %385, %389, %390 : <ui32>
                  %392 = ascvec.and %377, %375, %309 : <ui32>, <ui32> -> <ui32>
                  %393 = ascvec.add %391, %392, %309 : <ui32>
                  %394 = ascvec.shr %393, 16, %309 : <ui32> -> <ui32>
                  %395 = ascvec.min_scalar %394, %c123_i32, %309 : <ui32>, i32
                  %396 = ascvec.or %395, %381, %309 : <ui32>, <ui32> -> <ui32>
                  %397 = ascvec.duplicate_scalar %c127_i32, %309 : i32 -> <ui32>
                  %398 = ascvec.cmp_gt_scalar %382, %c2139095040_i32, %309 : <ui32>, i32 -> !ascvec.mask
                  %399 = ascvec.select %397, %396, %398 : <ui32>
                  ascvec.storealign_pack %arg10[%317], %399, %mask {mode = "b32_to_b8"} : memref<32768xui8, 1>, index, <ui32>
                  %400 = arith.muli %arg17, %c2_i32_10 : i32
                  %401 = arith.index_cast %400 : i32 to index
                  ascvec.storealign_1st %arg11[%401], %334 : memref<1024xui32, 1>, index, <ui32>
                  %402 = ascvec.duplicate_scalar %c32_i32, %309 : i32 -> <ui32>
                  %403 = ascvec.gather_reg %334, %402 : <ui32>, <ui32> -> <ui32>
                  %404 = arith.addi %400, %c1_i32_12 : i32
                  %405 = arith.index_cast %404 : i32 to index
                  ascvec.storealign_1st %arg11[%405], %403 : memref<1024xui32, 1>, index, <ui32>
                }
                ascvec.mem_bar vst_vld
                %313 = scf.if %arg15 -> (i32) {
                  %314 = arith.remsi %arg13, %c32_i32 : i32
                  %315 = arith.cmpi ne, %314, %c0_i32_15 : i32
                  %316 = arith.cmpi slt, %314, %c0_i32_15 : i32
                  %317 = arith.andi %315, %316 : i1
                  %318 = arith.divsi %arg13, %c32_i32 : i32
                  %319 = arith.extui %317 : i1 to i32
                  %320 = arith.subi %318, %319 : i32
                  %321 = arith.muli %320, %c2_i32_10 : i32
                  scf.yield %321 : i32
                } else {
                  scf.yield %arg14 : i32
                }
                scf.for %arg17 = %c0_i32_15 to %313 step %c64_i32  : i32 {
                  %314 = arith.subi %313, %arg17 : i32
                  %mask, %remaining_out = ascvec.update_mask %314 : !ascvec.mask
                  %315 = ascvec.add_scalar %310, %arg17, %309 : <ui32>, i32
                  scf.if %arg15 {
                    %316 = ascvec.shr %315, 1, %309 : <ui32> -> <ui32>
                    %317 = ascvec.gather %arg11[%c0_14], %316, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %318 = ascvec.duplicate_scalar %c1_i32_12, %309 : i32 -> <ui32>
                    %319 = ascvec.and %310, %318, %309 : <ui32>, <ui32> -> <ui32>
                    %320 = ascvec.cmp_eq_scalar %319, %c0_i32_15, %309 : <ui32>, i32 -> !ascvec.mask
                    %321 = ascvec.duplicate_scalar %c0_i32_15, %309 : i32 -> <ui32>
                    %322 = ascvec.select %317, %321, %320 : <ui32>
                    %323 = arith.index_cast %arg17 : i32 to index
                    %c0_16 = arith.constant 0 : index
                    %c1_17 = arith.constant 1 : index
                    scf.for %arg18 = %c0_16 to %c1_17 step %c1_17 {
                      ascvec.storealign_pack %arg16[%323], %322, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  } else {
                    %316 = ascvec.gather %arg11[%c0_14], %315, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %317 = arith.index_cast %arg17 : i32 to index
                    %c0_16 = arith.constant 0 : index
                    %c1_17 = arith.constant 1 : index
                    scf.for %arg18 = %c0_16 to %c1_17 step %c1_17 {
                      ascvec.storealign_pack %arg16[%317], %316, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  }
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
            }
          }
          %138 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %139 = cannir.extract_shape %138 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
          %140 = cannir.extract_dim %139[0] : !cannir.shape<(-1, -1)>
          %141 = arith.minsi %92, %140 : i64
          %142 = arith.subi %140, %141 : i64
          %143 = arith.minsi %142, %c1_i64 : i64
          %144 = arith.cmpi sle, %92, %140 : i64
          %145 = arith.select %144, %143, %c0_i64 : i64
          %146 = arith.cmpi sgt, %145, %c1_i64 : i64
          %147 = arith.select %146, %c1_i64, %145 : i64
          %148 = cannir.extract_dim %139[1] : !cannir.shape<(-1, -1)>
          %149 = arith.divsi %148, %116 : i64
          %150 = arith.minsi %94, %149 : i64
          %151 = arith.muli %150, %69 : i64
          %152 = arith.subi %148, %151 : i64
          %153 = arith.minsi %69, %152 : i64
          %154 = arith.cmpi sle, %94, %149 : i64
          %155 = arith.select %154, %153, %c0_i64 : i64
          %156 = arith.cmpi sgt, %155, %c32768_i64 : i64
          %157 = arith.select %156, %c32768_i64, %155 : i64
          %158 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %159 = cannir.extract_stride %158 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %160 = cannir.extract_dim %159[0] : !cannir.stride<(-1, 1)>
          %161 = arith.muli %92, %160 : i64
          %162 = arith.addi %161, %95 : i64
          %163 = arith.muli %162, %c8_i64 : i64
          %164 = ascscalar.memref_address %alloc_0 : memref<1xui8>
          %165 = arith.divui %163, %c8_i64 : i64
          %166 = arith.addi %164, %165 : i64
          %167 = ascscalar.bind_gm_ptr %166 : memref<1xui8>
          %168 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<32768xui8, 1> -> memref<?xui8, 1>
          %169 = cannir.extract_stride %138 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %170 = cannir.extract_dim %169[0] : !cannir.stride<(-1, 1)>
          %171 = arith.cmpi ugt, %147, %c0_i64 : i64
          %172 = arith.cmpi ugt, %157, %c0_i64 : i64
          %173 = arith.andi %171, %172 : i1
          scf.if %173 {
            ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
            %210 = arith.trunci %147 : i64 to i32
            %211 = arith.trunci %157 : i64 to i32
            ascvec.copy_ub2gm %167[%c0], %168[%c0], %210, %211, %170, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
          }
          %174 = arith.muli %68, %31 : i64
          %175 = cannir.extract_dim %21[0] : !cannir.shape<(-1, -1)>
          %176 = arith.minsi %92, %175 : i64
          %177 = arith.subi %175, %176 : i64
          %178 = arith.minsi %177, %c1_i64 : i64
          %179 = arith.cmpi sle, %92, %175 : i64
          %180 = arith.select %179, %178, %c0_i64 : i64
          %181 = arith.cmpi sgt, %180, %c1_i64 : i64
          %182 = arith.select %181, %c1_i64, %180 : i64
          %183 = arith.maxsi %174, %c1_i64 : i64
          %184 = arith.divsi %22, %183 : i64
          %185 = arith.minsi %94, %184 : i64
          %186 = arith.muli %185, %174 : i64
          %187 = arith.subi %22, %186 : i64
          %188 = arith.minsi %174, %187 : i64
          %189 = arith.cmpi sle, %94, %184 : i64
          %190 = arith.select %189, %188, %c0_i64 : i64
          %191 = arith.cmpi sgt, %190, %c2048_i64 : i64
          %192 = arith.select %191, %c2048_i64, %190 : i64
          %193 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %194 = cannir.extract_stride %193 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %195 = cannir.extract_dim %194[0] : !cannir.stride<(-1, 1)>
          %196 = arith.muli %92, %195 : i64
          %197 = arith.muli %94, %174 : i64
          %198 = arith.addi %196, %197 : i64
          %199 = arith.muli %198, %c8_i64 : i64
          %200 = ascscalar.memref_address %alloc_1 : memref<1xui8>
          %201 = arith.divui %199, %c8_i64 : i64
          %202 = arith.addi %200, %201 : i64
          %203 = ascscalar.bind_gm_ptr %202 : memref<1xui8>
          %204 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<2048xui8, 1> -> memref<?xui8, 1>
          %205 = cannir.extract_stride %20 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %206 = cannir.extract_dim %205[0] : !cannir.stride<(-1, 1)>
          %207 = arith.cmpi ugt, %182, %c0_i64 : i64
          %208 = arith.cmpi ugt, %192, %c0_i64 : i64
          %209 = arith.andi %207, %208 : i1
          scf.if %209 {
            ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
            %210 = arith.trunci %182 : i64 to i32
            %211 = arith.trunci %192 : i64 to i32
            ascvec.copy_ub2gm %203[%c0], %204[%c0], %210, %211, %206, %c2048_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %106 : i64
        }
      }
    }
    return
  }
}
