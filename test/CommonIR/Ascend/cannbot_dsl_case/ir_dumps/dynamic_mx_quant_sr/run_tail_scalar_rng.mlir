module {
  func.func @run_tail_scalar_rng_de4ff57a528fb4da68658b150d4f77fd(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64) attributes {cannbotdsl.contract_digest = "de4ff57a528fb4da68658b150d4f77fddfd9b51a65d47cc65222f5292cccf796", cannbotdsl.public_slot_count = 7 : i64, cannbotdsl.slot_dtype_bits = [16, 8, 8, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 1, 1, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 2, 2, 2, 2], cannbotdsl.slot_type_digests = ["b61381ca951a0f603976832b4955166996f3e8a7d1460da4be1b89bae4e92f47", "481dbcd23d79e136c72cda6b88f83676c3549f81b3cd3be5caf1bc2d99f3440f", "f3b37a0bd876490cb7c40dd697b264fbf7c668be5608a770181d9307f4b9c55d", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel_run_tail_scalar_rng_0(%arg4) [%arg0, %arg1, %arg2, %arg3, %arg5, %arg6] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2>, name = "_DynamicMxQuantSrKernel_run_tail_scalar_rng", op_type = "_DynamicMxQuantSrKernel_run_tail_scalar_rng", roles = array<i64: 0, 1, 1>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, i64, i64
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel_run_tail_scalar_rng_0(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel_run_tail_scalar_rng_0", cannir.resource.ids_prepared} {
    %c2048_i32 = arith.constant 2048 : i32
    %c32768_i32 = arith.constant 32768 : i32
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
    %alloc_2 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<32768xbf16, 1>
    %alloc_3 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<32768xbf16, 1>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 131072 : i64} : memref<32768xui8, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 163840 : i64} : memref<2048xui8, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 165888 : i64} : memref<1024xui32, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 169984 : i64} : memref<8192xi32, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 202752 : i64} : memref<2048xui16, 1>
    %0 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
    %1 = cannir.extract_shape %0 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
    %2 = cannir.extract_dim %1[0] : !cannir.shape<(-1, -1)>
    %3 = arith.cmpi eq, %2, %c1_i64 : i64
    %4 = scf.if %3 -> (i1) {
      %5 = cannir.extract_dim %1[1] : !cannir.shape<(-1, -1)>
      %6 = arith.remsi %5, %c64_i64 : i64
      %7 = arith.cmpi ne, %6, %c0_i64 : i64
      %8 = arith.cmpi slt, %6, %c0_i64 : i64
      %9 = arith.andi %7, %8 : i1
      %10 = arith.select %9, %c64_i64, %c0_i64 : i64
      %11 = arith.addi %6, %10 overflow<nsw> : i64
      %12 = arith.cmpi eq, %11, %c0_i64 : i64
      %13 = scf.if %12 -> (i1) {
        %14 = ascscalar.get_block_num
        %15 = arith.muli %14, %c32768_i64 : i64
        %16 = arith.cmpi sgt, %5, %15 : i64
        %17 = scf.if %16 -> (i1) {
          %18 = arith.remsi %5, %15 : i64
          %19 = arith.cmpi ne, %18, %c0_i64 : i64
          %20 = arith.cmpi slt, %18, %c0_i64 : i64
          %21 = arith.cmpi slt, %15, %c0_i64 : i64
          %22 = arith.xori %20, %21 : i1
          %23 = arith.andi %19, %22 : i1
          %24 = arith.select %23, %15, %c0_i64 : i64
          %25 = arith.addi %18, %24 : i64
          %26 = arith.cmpi ne, %25, %c0_i64 : i64
          scf.yield %26 : i1
        } else {
          scf.yield %false : i1
        }
        scf.yield %17 : i1
      } else {
        scf.yield %false : i1
      }
      scf.yield %13 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %4 {
      %5 = cannir.extract_dim %1[1] : !cannir.shape<(-1, -1)>
      %6 = arith.remsi %5, %c64_i64 : i64
      %7 = arith.cmpi ne, %6, %c0_i64 : i64
      %8 = arith.cmpi slt, %6, %c0_i64 : i64
      %9 = arith.andi %7, %8 : i1
      %10 = arith.divsi %5, %c64_i64 : i64
      %11 = arith.extui %9 : i1 to i64
      %12 = arith.subi %10, %11 : i64
      %13 = ascscalar.get_block_num
      %14 = arith.addi %12, %13 : i64
      %15 = arith.subi %14, %c1_i64 : i64
      %16 = arith.remsi %15, %13 : i64
      %17 = arith.cmpi ne, %16, %c0_i64 : i64
      %18 = arith.cmpi slt, %16, %c0_i64 : i64
      %19 = arith.cmpi slt, %13, %c0_i64 : i64
      %20 = arith.xori %18, %19 : i1
      %21 = arith.andi %17, %20 : i1
      %22 = arith.divsi %15, %13 : i64
      %23 = arith.extui %21 : i1 to i64
      %24 = arith.subi %22, %23 : i64
      %25 = arith.cmpi sge, %24, %c4_i64 : i64
      %26 = scf.if %25 -> (i64) {
        %46 = arith.addi %24, %c3_i64 : i64
        %47 = arith.remsi %46, %c4_i64 : i64
        %48 = arith.cmpi ne, %47, %c0_i64 : i64
        %49 = arith.cmpi slt, %47, %c0_i64 : i64
        %50 = arith.andi %48, %49 : i1
        %51 = arith.divsi %46, %c4_i64 : i64
        %52 = arith.extui %50 : i1 to i64
        %53 = arith.subi %51, %52 : i64
        %54 = arith.muli %53, %c4_i64 : i64
        scf.yield %54 : i64
      } else {
        scf.yield %24 : i64
      }
      %27 = arith.muli %13, %c64_i64 : i64
      %28 = arith.cmpi sge, %12, %27 : i64
      %29 = scf.if %28 -> (i64) {
        %46 = arith.addi %26, %c15_i64 : i64
        %47 = arith.remsi %46, %c16_i64 : i64
        %48 = arith.cmpi ne, %47, %c0_i64 : i64
        %49 = arith.cmpi slt, %47, %c0_i64 : i64
        %50 = arith.andi %48, %49 : i1
        %51 = arith.divsi %46, %c16_i64 : i64
        %52 = arith.extui %50 : i1 to i64
        %53 = arith.subi %51, %52 : i64
        %54 = arith.muli %53, %c16_i64 : i64
        scf.yield %54 : i64
      } else {
        scf.yield %26 : i64
      }
      %30 = arith.muli %29, %c64_i64 : i64
      %31 = ascscalar.get_block_idx
      %32 = arith.muli %31, %30 : i64
      %33 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %34 = cannir.extract_shape %33 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %35 = cannir.extract_dim %34[1] : !cannir.shape<(-1, -1)>
      %36 = arith.remsi %5, %c16_i64 : i64
      %37 = arith.cmpi ne, %36, %c0_i64 : i64
      %38 = arith.cmpi slt, %36, %c0_i64 : i64
      %39 = arith.andi %37, %38 : i1
      %40 = arith.divsi %5, %c16_i64 : i64
      %41 = arith.extui %39 : i1 to i64
      %42 = arith.subi %40, %41 : i64
      %43 = arith.cmpi eq, %35, %42 : i64
      %44 = arith.select %43, %c4_i64, %c2_i64 : i64
      %45 = arith.cmpi slt, %32, %5 : i64
      scf.if %45 {
        %46 = arith.muli %29, %44 : i64
        %47 = arith.subi %5, %32 : i64
        %48 = arith.cmpi sgt, %47, %30 : i64
        %49 = arith.select %48, %30, %47 : i64
        %50 = arith.addi %49, %c32767_i64 : i64
        %51 = arith.remsi %50, %c32768_i64 : i64
        %52 = arith.cmpi ne, %51, %c0_i64 : i64
        %53 = arith.cmpi slt, %51, %c0_i64 : i64
        %54 = arith.andi %52, %53 : i1
        %55 = arith.divsi %50, %c32768_i64 : i64
        %56 = arith.extui %54 : i1 to i64
        %57 = arith.subi %55, %56 : i64
        %58 = scf.for %arg6 = %c0_i64 to %57 step %c1_i64 iter_args(%arg7 = %c0_i64) -> (i64)  : i64 {
          %59 = arith.muli %arg6, %c32768_i64 : i64
          %60 = arith.subi %49, %59 : i64
          %61 = arith.cmpi sgt, %60, %c32768_i64 : i64
          %62 = arith.select %61, %c32768_i64, %60 : i64
          %63 = arith.cmpi eq, %arg7, %c0_i64 : i64
          %64 = ascscalar.memref_address %alloc_2 : memref<32768xbf16, 1>
          %65 = ascscalar.memref_address %alloc_3 : memref<32768xbf16, 1>
          %66 = arith.select %63, %64, %65 : i64
          %67 = arith.cmpi ne, %arg7, %c0_i64 : i64
          %68 = arith.extui %67 : i1 to i64
          %69 = arith.addi %arg7, %c1_i64 overflow<nsw> : i64
          %70 = arith.remui %69, %c2_i64 : i64
          %71 = arith.minsi %2, %c0_i64 : i64
          %72 = arith.subi %2, %71 : i64
          %73 = arith.minsi %72, %c1_i64 : i64
          %74 = arith.cmpi sge, %2, %c0_i64 : i64
          %75 = arith.select %74, %73, %c0_i64 : i64
          %76 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %77 = cannir.extract_stride %76 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %78 = cannir.extract_dim %77[0] : !cannir.stride<(-1, 1)>
          %79 = arith.muli %78, %c2_i64 : i64
          %80 = arith.maxsi %30, %c1_i64 : i64
          %81 = arith.divsi %5, %80 : i64
          %82 = arith.minsi %31, %81 : i64
          %83 = arith.muli %82, %30 : i64
          %84 = arith.subi %5, %83 : i64
          %85 = arith.minsi %30, %84 : i64
          %86 = arith.cmpi sle, %31, %81 : i64
          %87 = arith.select %86, %85, %c0_i64 : i64
          %88 = arith.divsi %87, %c32768_i64 : i64
          %89 = arith.minsi %arg6, %88 : i64
          %90 = arith.muli %89, %c32768_i64 : i64
          %91 = arith.subi %87, %90 : i64
          %92 = arith.minsi %91, %c32768_i64 : i64
          %93 = arith.cmpi sle, %arg6, %88 : i64
          %94 = arith.select %93, %92, %c0_i64 : i64
          %95 = arith.muli %arg6, %c65536_i64 : i64
          %96 = arith.muli %32, %c2_i64 : i64
          %97 = arith.addi %95, %96 : i64
          %98 = arith.muli %94, %c2_i64 : i64
          %99 = arith.cmpi eq, %98, %c0_i64 : i64
          %100 = arith.select %99, %c0_i64, %75 : i64
          %101 = arith.select %99, %c2_i64, %98 : i64
          %102 = ascscalar.bind_local_ptr %66 : memref<1xbf16, 1>
          ascvec.get_buf mte2, %68, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%102, %alloc, %c0_i32, %c0_i32, %c0_i64, %97, %100, %101, %79, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %68, 0 {cannir.buffer.generated_sync}
          %103 = ascscalar.bind_local_ptr %66 : memref<32768xbf16, 1>
          %104 = arith.addi %32, %59 : i64
          %105 = arith.cmpi sge, %62, %c256_i64 : i64
          %106 = arith.remsi %62, %c256_i64 : i64
          %107 = arith.cmpi eq, %106, %c0_i64 : i64
          %108 = arith.select %105, %107, %false : i1
          scf.if %108 {
            %188 = arith.remsi %arg3, %c256_i64 : i64
            %189 = arith.cmpi ne, %188, %c0_i64 : i64
            %190 = arith.cmpi slt, %188, %c0_i64 : i64
            %191 = arith.andi %189, %190 : i1
            %192 = arith.select %191, %c256_i64, %c0_i64 : i64
            %193 = arith.addi %188, %192 overflow<nsw> : i64
            %194 = arith.cmpi eq, %193, %c0_i64 : i64
            scf.if %194 {
              %195 = arith.cmpi sge, %62, %c1024_i64 : i64
              %196 = arith.remsi %62, %c1024_i64 : i64
              %197 = arith.cmpi eq, %196, %c0_i64 : i64
              %198 = arith.select %195, %197, %false : i1
              scf.if %198 {
                %199 = arith.divsi %62, %c256_i64 : i64
                %200 = arith.addi %62, %c1023_i64 : i64
                %201 = arith.divsi %200, %c1024_i64 : i64
                %202 = arith.trunci %arg4 : i64 to i32
                %203 = arith.extui %202 : i32 to i64
                %204 = arith.subi %arg4, %203 : i64
                %205 = arith.remsi %204, %c4294967296_i64 : i64
                %206 = arith.cmpi ne, %205, %c0_i64 : i64
                %207 = arith.cmpi slt, %205, %c0_i64 : i64
                %208 = arith.andi %206, %207 : i1
                %209 = arith.divsi %204, %c4294967296_i64 : i64
                %210 = arith.extui %208 : i1 to i64
                %211 = arith.subi %209, %210 : i64
                %212 = arith.trunci %211 : i64 to i32
                %213 = arith.divui %104, %c32_i64 : i64
                %214 = arith.trunci %213 : i64 to i32
                %215 = arith.extui %214 : i32 to i64
                %216 = arith.subi %213, %215 : i64
                %217 = arith.remsi %216, %c4294967296_i64 : i64
                %218 = arith.cmpi ne, %217, %c0_i64 : i64
                %219 = arith.cmpi slt, %217, %c0_i64 : i64
                %220 = arith.andi %218, %219 : i1
                %221 = arith.divsi %216, %c4294967296_i64 : i64
                %222 = arith.extui %220 : i1 to i64
                %223 = arith.subi %221, %222 : i64
                %224 = arith.trunci %223 : i64 to i32
                %225 = arith.trunci %arg5 : i64 to i32
                %226 = arith.extui %225 : i32 to i64
                %227 = arith.subi %arg5, %226 : i64
                %228 = arith.remsi %227, %c4294967296_i64 : i64
                %229 = arith.cmpi ne, %228, %c0_i64 : i64
                %230 = arith.cmpi slt, %228, %c0_i64 : i64
                %231 = arith.andi %229, %230 : i1
                %232 = arith.divsi %227, %c4294967296_i64 : i64
                %233 = arith.extui %231 : i1 to i64
                %234 = arith.subi %232, %233 : i64
                %235 = arith.trunci %234 : i64 to i32
                %236 = arith.divui %225, %c4_i32 : i32
                %237 = arith.remui %235, %c4_i32 : i32
                %238 = arith.muli %237, %c1073741824_i32 : i32
                %239 = arith.addi %236, %238 : i32
                %240 = arith.divui %235, %c4_i32 : i32
                %241 = arith.muli %201, %c32_i64 : i64
                %242 = arith.divui %213, %c16384_i64 : i64
                %243 = arith.addi %213, %241 : i64
                %244 = arith.subi %243, %c1_i64 : i64
                %245 = arith.divui %244, %c16384_i64 : i64
                %246 = arith.cmpi eq, %242, %245 : i64
                scf.if %246 {
                  %256 = arith.trunci %242 : i64 to i32
                  %257 = arith.extui %256 : i32 to i64
                  %258 = arith.subi %242, %257 : i64
                  %259 = arith.remsi %258, %c4294967296_i64 : i64
                  %260 = arith.cmpi ne, %259, %c0_i64 : i64
                  %261 = arith.cmpi slt, %259, %c0_i64 : i64
                  %262 = arith.andi %260, %261 : i1
                  %263 = arith.divsi %258, %c4294967296_i64 : i64
                  %264 = arith.extui %262 : i1 to i64
                  %265 = arith.subi %263, %264 : i64
                  %266 = arith.trunci %265 : i64 to i32
                  %267 = arith.addi %239, %256 : i32
                  %268 = arith.addi %240, %266 : i32
                  %269 = arith.cmpi ult, %267, %239 : i32
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
                  %285 = arith.xori %271, %212 : i32
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
                  %299 = arith.addi %202, %c-1640531527_i32 : i32
                  %300 = arith.addi %212, %c-1150833019_i32 : i32
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
                  %316 = arith.xori %284, %202 : i32
                  %317 = arith.addi %202, %c1013904242_i32 : i32
                  %318 = arith.xori %288, %317 : i32
                  %319 = arith.addi %212, %c1993301258_i32 : i32
                  %320 = arith.xori %315, %319 : i32
                  ascsimt.vf(%214, %316, %300, %318, %320, %202, %212, %305, %alloc_7, %241 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: memref<8192xi32, 1>, %arg17: i64):
                    %c256_i64_9 = arith.constant 256 : i64
                    %c3_i64_10 = arith.constant 3 : i64
                    %c2_i64_11 = arith.constant 2 : i64
                    %c1_i64_12 = arith.constant 1 : i64
                    %c4_i64_13 = arith.constant 4 : i64
                    %c1_i32_14 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c3_i32 = arith.constant 3 : i32
                    %c-1150833019_i32_15 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_16 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %321 = ascsimt.thread_idx {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %322 = arith.extsi %321 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg18 = %322 to %arg17 step %c256_i64_9  : i64 {
                      %323 = arith.trunci %arg18 : i64 to i32
                      %324 = arith.addi %arg8, %323 : i32
                      %325 = arith.remui %324, %c16384_i32 : i32
                      %326 = arith.xori %325, %arg9 : i32
                      %327 = arith.muli %326, %c-766435501_i32 : i32
                      %328 = "ascsimt.bitcast"(%326) : (i32) -> ui32
                      %329 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %330 = ascsimt.math_binary %328, %329, "mulhi" : ui32, ui32 -> ui32
                      %331 = "ascsimt.bitcast"(%330) : (ui32) -> i32
                      %332 = arith.xori %331, %arg10 : i32
                      %333 = arith.muli %332, %c-845247145_i32 : i32
                      %334 = "ascsimt.bitcast"(%332) : (i32) -> ui32
                      %335 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %336 = ascsimt.math_binary %334, %335, "mulhi" : ui32, ui32 -> ui32
                      %337 = "ascsimt.bitcast"(%336) : (ui32) -> i32
                      %338 = arith.xori %337, %arg11 : i32
                      %339 = arith.xori %327, %arg12 : i32
                      %340:4 = scf.for %arg19 = %c3_i32 to %c10_i32 step %c1_i32_14 iter_args(%arg20 = %338, %arg21 = %333, %arg22 = %339, %arg23 = %arg15) -> (i32, i32, i32, i32)  : i32 {
                        %349 = arith.muli %arg20, %c-766435501_i32 : i32
                        %350 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                        %351 = ascsimt.math_binary %350, %329, "mulhi" : ui32, ui32 -> ui32
                        %352 = "ascsimt.bitcast"(%351) : (ui32) -> i32
                        %353 = arith.muli %arg22, %c-845247145_i32 : i32
                        %354 = "ascsimt.bitcast"(%arg22) : (i32) -> ui32
                        %355 = ascsimt.math_binary %354, %335, "mulhi" : ui32, ui32 -> ui32
                        %356 = "ascsimt.bitcast"(%355) : (ui32) -> i32
                        %357 = arith.muli %arg19, %c-1640531527_i32_16 : i32
                        %358 = arith.addi %arg13, %357 : i32
                        %359 = arith.muli %arg19, %c-1150833019_i32_15 : i32
                        %360 = arith.addi %arg14, %359 : i32
                        %361 = arith.xori %356, %arg21 : i32
                        %362 = arith.xori %361, %358 : i32
                        %363 = arith.xori %352, %arg23 : i32
                        %364 = arith.xori %363, %360 : i32
                        scf.yield %362, %353, %364, %349 : i32, i32, i32, i32
                      }
                      %341 = arith.muli %arg18, %c4_i64_13 : i64
                      %342 = arith.index_cast %341 : i64 to index
                      ascsimt.store %arg16[%342], %340#0 : memref<8192xi32, 1>, i32
                      %343 = arith.addi %341, %c1_i64_12 : i64
                      %344 = arith.index_cast %343 : i64 to index
                      ascsimt.store %arg16[%344], %340#1 : memref<8192xi32, 1>, i32
                      %345 = arith.addi %341, %c2_i64_11 : i64
                      %346 = arith.index_cast %345 : i64 to index
                      ascsimt.store %arg16[%346], %340#2 : memref<8192xi32, 1>, i32
                      %347 = arith.addi %341, %c3_i64_10 : i64
                      %348 = arith.index_cast %347 : i64 to index
                      ascsimt.store %arg16[%348], %340#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%214, %224, %239, %240, %202, %212, %alloc_7, %241 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: memref<8192xi32, 1>, %arg15: i64):
                    %c256_i64_9 = arith.constant 256 : i64
                    %c3_i64_10 = arith.constant 3 : i64
                    %c2_i64_11 = arith.constant 2 : i64
                    %c1_i64_12 = arith.constant 1 : i64
                    %c4_i64_13 = arith.constant 4 : i64
                    %c1_i32_14 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c0_i32_15 = arith.constant 0 : i32
                    %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c262144_i32 = arith.constant 262144 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %256 = ascsimt.thread_idx {ascsimt.scope = 2 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %257 = arith.extsi %256 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg16 = %257 to %arg15 step %c256_i64_9  : i64 {
                      %258 = arith.trunci %arg16 : i64 to i32
                      %259 = arith.addi %arg8, %258 : i32
                      %260 = arith.cmpi ult, %259, %arg8 : i32
                      %261 = arith.extui %260 : i1 to i32
                      %262 = arith.addi %arg9, %261 : i32
                      %263 = arith.divui %259, %c16384_i32 : i32
                      %264 = arith.remui %262, %c16384_i32 : i32
                      %265 = arith.muli %264, %c262144_i32 : i32
                      %266 = arith.addi %263, %265 : i32
                      %267 = arith.divui %262, %c16384_i32 : i32
                      %268 = arith.addi %arg10, %266 : i32
                      %269 = arith.addi %arg11, %267 : i32
                      %270 = arith.cmpi ult, %268, %arg10 : i32
                      %271 = arith.extui %270 : i1 to i32
                      %272 = arith.addi %269, %271 : i32
                      %273 = arith.remui %259, %c16384_i32 : i32
                      %274:4 = scf.for %arg17 = %c0_i32_15 to %c10_i32 step %c1_i32_14 iter_args(%arg18 = %c0_i32_15, %arg19 = %273, %arg20 = %268, %arg21 = %272) -> (i32, i32, i32, i32)  : i32 {
                        %283 = arith.muli %arg18, %c-766435501_i32 : i32
                        %284 = "ascsimt.bitcast"(%arg18) : (i32) -> ui32
                        %285 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %286 = ascsimt.math_binary %284, %285, "mulhi" : ui32, ui32 -> ui32
                        %287 = "ascsimt.bitcast"(%286) : (ui32) -> i32
                        %288 = arith.muli %arg20, %c-845247145_i32 : i32
                        %289 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                        %290 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %291 = ascsimt.math_binary %289, %290, "mulhi" : ui32, ui32 -> ui32
                        %292 = "ascsimt.bitcast"(%291) : (ui32) -> i32
                        %293 = arith.muli %arg17, %c-1640531527_i32_17 : i32
                        %294 = arith.addi %arg12, %293 : i32
                        %295 = arith.muli %arg17, %c-1150833019_i32_16 : i32
                        %296 = arith.addi %arg13, %295 : i32
                        %297 = arith.xori %292, %arg19 : i32
                        %298 = arith.xori %297, %294 : i32
                        %299 = arith.xori %287, %arg21 : i32
                        %300 = arith.xori %299, %296 : i32
                        scf.yield %298, %288, %300, %283 : i32, i32, i32, i32
                      }
                      %275 = arith.muli %arg16, %c4_i64_13 : i64
                      %276 = arith.index_cast %275 : i64 to index
                      ascsimt.store %arg14[%276], %274#0 : memref<8192xi32, 1>, i32
                      %277 = arith.addi %275, %c1_i64_12 : i64
                      %278 = arith.index_cast %277 : i64 to index
                      ascsimt.store %arg14[%278], %274#1 : memref<8192xi32, 1>, i32
                      %279 = arith.addi %275, %c2_i64_11 : i64
                      %280 = arith.index_cast %279 : i64 to index
                      ascsimt.store %arg14[%280], %274#2 : memref<8192xi32, 1>, i32
                      %281 = arith.addi %275, %c3_i64_10 : i64
                      %282 = arith.index_cast %281 : i64 to index
                      ascsimt.store %arg14[%282], %274#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %247 = arith.trunci %44 : i64 to i32
                %248 = arith.divsi %247, %c2_i32 : i32
                %249 = arith.trunci %248 : i32 to i8
                %250 = arith.trunci %199 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %251 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%249, %alloc_5, %103, %251, %alloc_8, %250 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, memref<2048xui16, 1>, i32) {
                ^bb0(%arg8: i8, %arg9: memref<2048xui8, 1>, %arg10: memref<32768xbf16, 1>, %arg11: memref<?xui16, 1>, %arg12: memref<2048xui16, 1>, %arg13: i32):
                  %c16_9 = arith.constant 16 : index
                  %c4_i32_10 = arith.constant 4 : i32
                  %c0_i32_11 = arith.constant 0 : i32
                  %c3_i32 = arith.constant 3 : i32
                  %c2_i32_12 = arith.constant 2 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c1_i32_13 = arith.constant 1 : i32
                  %c0_14 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %256 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %257 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %258 = ascvec.duplicate_scalar %c32767_i16, %256 : i16 -> <ui16>
                  %259 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %260 = ascvec.shr %259, 4, %256 : <ui16> -> <ui16>
                  %261 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %262 = ascvec.duplicate_scalar %c1_i8, %257 : i8 -> <ui8>
                  %263 = ascvec.and %261, %262, %257 : <ui8>, <ui8> -> <ui8>
                  %264 = ascvec.cmp_lt_scalar %261, %c16_i8, %257 : <ui8>, i8 -> !ascvec.mask
                  %265 = ascvec.cmp_lt_scalar %263, %arg8, %257 : <ui8>, i8 -> !ascvec.mask
                  %266 = ascvec.mask_and %264, %265, %257
                  ascvec.store_unalign_begin %arg9 : memref<2048xui8, 1>
                  %mask_15, %remaining_out_16 = ascvec.update_mask %c1_i32_13 elem_bits 16 : !ascvec.mask
                  %267 = ascvec.duplicate_scalar %c-1_i16, %mask_15 : i16 -> <ui16>
                  %268 = scf.for %arg14 = %c0_i32_11 to %arg13 step %c4_i32_10 iter_args(%arg15 = %267) -> (!ascvec.vreg<ui16>)  : i32 {
                    %269 = ascvec.duplicate_scalar %c-1_i16, %256 : i16 -> <ui16>
                    %270 = ascvec.duplicate_scalar %c0_i16, %256 : i16 -> <ui16>
                    %271 = arith.muli %arg14, %c256_i32 : i32
                    %272 = arith.index_cast %271 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg10[%272] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %273 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %274 = ascvec.and %273, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %275 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %276 = ascvec.and %275, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %277 = ascvec.max %274, %276, %256 : <ui16>
                    %278 = ascvec.reduce_max_datablock %277, %256 : <ui16>
                    %279 = ascvec.shr %278, 7, %256 : <ui16> -> <ui16>
                    %280 = ascvec.cmp_ge_scalar %279, %c255_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %281 = ascvec.max_scalar %279, %c15_i16, %256 : <ui16>, i16
                    %282 = ascvec.add_scalar %281, %c-15_i16, %256 : <ui16>, i16
                    %283 = ascvec.duplicate_scalar %c255_i16, %280 merge %282 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %284 = ascvec.duplicate_scalar %c254_i16, %256 : i16 -> <ui16>
                    %285 = ascvec.sub %284, %283, %256 : <ui16>
                    %286 = ascvec.cmp_ne_scalar %283, %c0_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %287 = ascvec.shl %285, 7, %286 : <ui16> -> <ui16>
                    %288 = ascvec.max %270, %283, %mask : <ui16>
                    %289 = ascvec.gather_reg %287, %260 : <ui16>, <ui16> -> <ui16>
                    %290 = ascvec.reinterpret %289 : <ui16> -> <bf16>
                    %291 = ascvec.mul %result0, %290, %256 : <bf16>
                    %292 = ascvec.mul %result1, %290, %256 : <bf16>
                    ascvec.storealign_intlv %arg10[%272], %291, %292 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %293 = ascvec.reinterpret %291 : <bf16> -> <ui16>
                    %294 = ascvec.and %293, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %295 = ascvec.reinterpret %292 : <bf16> -> <ui16>
                    %296 = ascvec.and %295, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %297 = ascvec.min %294, %296, %256 : <ui16>
                    %298 = ascvec.min %269, %297, %256 : <ui16>
                    %299 = arith.muli %arg14, %c16_i32 : i32
                    %300 = arith.index_cast %299 : i32 to index
                    ascvec.storealign %arg11[%300], %283, %mask : memref<?xui16, 1>, index, <ui16>
                    %301 = ascvec.reinterpret %283 : <ui16> -> <ui8>
                    %302 = ascvec.squeeze_store_reg %301, %266 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_14], %302 : memref<2048xui8, 1>, <ui8>
                    %303 = arith.addi %arg14, %c1_i32_13 : i32
                    %304 = arith.muli %303, %c256_i32 : i32
                    %305 = arith.index_cast %304 : i32 to index
                    %result0_18, %result1_19 = ascvec.loadalign_deintlv %arg10[%305] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %306 = ascvec.reinterpret %result0_18 : <bf16> -> <ui16>
                    %307 = ascvec.and %306, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %308 = ascvec.reinterpret %result1_19 : <bf16> -> <ui16>
                    %309 = ascvec.and %308, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %310 = ascvec.max %307, %309, %256 : <ui16>
                    %311 = ascvec.reduce_max_datablock %310, %256 : <ui16>
                    %312 = ascvec.shr %311, 7, %256 : <ui16> -> <ui16>
                    %313 = ascvec.cmp_ge_scalar %312, %c255_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %314 = ascvec.max_scalar %312, %c15_i16, %256 : <ui16>, i16
                    %315 = ascvec.add_scalar %314, %c-15_i16, %256 : <ui16>, i16
                    %316 = ascvec.duplicate_scalar %c255_i16, %313 merge %315 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %317 = ascvec.sub %284, %316, %256 : <ui16>
                    %318 = ascvec.cmp_ne_scalar %316, %c0_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %319 = ascvec.shl %317, 7, %318 : <ui16> -> <ui16>
                    %320 = ascvec.max %288, %316, %mask : <ui16>
                    %321 = ascvec.gather_reg %319, %260 : <ui16>, <ui16> -> <ui16>
                    %322 = ascvec.reinterpret %321 : <ui16> -> <bf16>
                    %323 = ascvec.mul %result0_18, %322, %256 : <bf16>
                    %324 = ascvec.mul %result1_19, %322, %256 : <bf16>
                    ascvec.storealign_intlv %arg10[%305], %323, %324 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %325 = ascvec.reinterpret %323 : <bf16> -> <ui16>
                    %326 = ascvec.and %325, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %327 = ascvec.reinterpret %324 : <bf16> -> <ui16>
                    %328 = ascvec.and %327, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %329 = ascvec.min %326, %328, %256 : <ui16>
                    %330 = ascvec.min %298, %329, %256 : <ui16>
                    %331 = arith.muli %303, %c16_i32 : i32
                    %332 = arith.index_cast %331 : i32 to index
                    ascvec.storealign %arg11[%332], %316, %mask : memref<?xui16, 1>, index, <ui16>
                    %333 = ascvec.reinterpret %316 : <ui16> -> <ui8>
                    %334 = ascvec.squeeze_store_reg %333, %266 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_14], %334 : memref<2048xui8, 1>, <ui8>
                    %335 = arith.addi %arg14, %c2_i32_12 : i32
                    %336 = arith.muli %335, %c256_i32 : i32
                    %337 = arith.index_cast %336 : i32 to index
                    %result0_20, %result1_21 = ascvec.loadalign_deintlv %arg10[%337] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %338 = ascvec.reinterpret %result0_20 : <bf16> -> <ui16>
                    %339 = ascvec.and %338, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %340 = ascvec.reinterpret %result1_21 : <bf16> -> <ui16>
                    %341 = ascvec.and %340, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %342 = ascvec.max %339, %341, %256 : <ui16>
                    %343 = ascvec.reduce_max_datablock %342, %256 : <ui16>
                    %344 = ascvec.shr %343, 7, %256 : <ui16> -> <ui16>
                    %345 = ascvec.cmp_ge_scalar %344, %c255_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %346 = ascvec.max_scalar %344, %c15_i16, %256 : <ui16>, i16
                    %347 = ascvec.add_scalar %346, %c-15_i16, %256 : <ui16>, i16
                    %348 = ascvec.duplicate_scalar %c255_i16, %345 merge %347 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %349 = ascvec.sub %284, %348, %256 : <ui16>
                    %350 = ascvec.cmp_ne_scalar %348, %c0_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %351 = ascvec.shl %349, 7, %350 : <ui16> -> <ui16>
                    %352 = ascvec.max %320, %348, %mask : <ui16>
                    %353 = ascvec.gather_reg %351, %260 : <ui16>, <ui16> -> <ui16>
                    %354 = ascvec.reinterpret %353 : <ui16> -> <bf16>
                    %355 = ascvec.mul %result0_20, %354, %256 : <bf16>
                    %356 = ascvec.mul %result1_21, %354, %256 : <bf16>
                    ascvec.storealign_intlv %arg10[%337], %355, %356 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %357 = ascvec.reinterpret %355 : <bf16> -> <ui16>
                    %358 = ascvec.and %357, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %359 = ascvec.reinterpret %356 : <bf16> -> <ui16>
                    %360 = ascvec.and %359, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %361 = ascvec.min %358, %360, %256 : <ui16>
                    %362 = ascvec.min %330, %361, %256 : <ui16>
                    %363 = arith.muli %335, %c16_i32 : i32
                    %364 = arith.index_cast %363 : i32 to index
                    ascvec.storealign %arg11[%364], %348, %mask : memref<?xui16, 1>, index, <ui16>
                    %365 = ascvec.reinterpret %348 : <ui16> -> <ui8>
                    %366 = ascvec.squeeze_store_reg %365, %266 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_14], %366 : memref<2048xui8, 1>, <ui8>
                    %367 = arith.addi %arg14, %c3_i32 : i32
                    %368 = arith.muli %367, %c256_i32 : i32
                    %369 = arith.index_cast %368 : i32 to index
                    %result0_22, %result1_23 = ascvec.loadalign_deintlv %arg10[%369] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %370 = ascvec.reinterpret %result0_22 : <bf16> -> <ui16>
                    %371 = ascvec.and %370, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %372 = ascvec.reinterpret %result1_23 : <bf16> -> <ui16>
                    %373 = ascvec.and %372, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %374 = ascvec.max %371, %373, %256 : <ui16>
                    %375 = ascvec.reduce_max_datablock %374, %256 : <ui16>
                    %376 = ascvec.shr %375, 7, %256 : <ui16> -> <ui16>
                    %377 = ascvec.cmp_ge_scalar %376, %c255_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %378 = ascvec.max_scalar %376, %c15_i16, %256 : <ui16>, i16
                    %379 = ascvec.add_scalar %378, %c-15_i16, %256 : <ui16>, i16
                    %380 = ascvec.duplicate_scalar %c255_i16, %377 merge %379 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %381 = ascvec.sub %284, %380, %256 : <ui16>
                    %382 = ascvec.cmp_ne_scalar %380, %c0_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %383 = ascvec.shl %381, 7, %382 : <ui16> -> <ui16>
                    %384 = ascvec.max %352, %380, %mask : <ui16>
                    %385 = ascvec.gather_reg %383, %260 : <ui16>, <ui16> -> <ui16>
                    %386 = ascvec.reinterpret %385 : <ui16> -> <bf16>
                    %387 = ascvec.mul %result0_22, %386, %256 : <bf16>
                    %388 = ascvec.mul %result1_23, %386, %256 : <bf16>
                    ascvec.storealign_intlv %arg10[%369], %387, %388 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %389 = ascvec.reinterpret %387 : <bf16> -> <ui16>
                    %390 = ascvec.and %389, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %391 = ascvec.reinterpret %388 : <bf16> -> <ui16>
                    %392 = ascvec.and %391, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %393 = ascvec.min %390, %392, %256 : <ui16>
                    %394 = ascvec.min %362, %393, %256 : <ui16>
                    %395 = arith.muli %367, %c16_i32 : i32
                    %396 = arith.index_cast %395 : i32 to index
                    ascvec.storealign %arg11[%396], %380, %mask : memref<?xui16, 1>, index, <ui16>
                    %397 = ascvec.reinterpret %380 : <ui16> -> <ui8>
                    %398 = ascvec.squeeze_store_reg %397, %266 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_14], %398 : memref<2048xui8, 1>, <ui8>
                    %399 = ascvec.reduce_min %394, %256 : <ui16>
                    %400 = ascvec.reduce_max %384, %mask : <ui16>
                    %401 = ascvec.cmp_eq_scalar %400, %c255_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %402 = ascvec.duplicate_scalar %c0_i16, %401 merge %399 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.storealign_1st %arg12[%300], %402 : memref<2048xui16, 1>, index, <ui16>
                    %403 = ascvec.min %arg15, %402, %mask_15 : <ui16>
                    scf.yield %403 : !ascvec.vreg<ui16>
                  }
                  %c0_17 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg14 = %c0_17 to %c1 step %c1 {
                    ascvec.store_unalign_post %arg9[%c0_14] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg12[%c16_9], %268 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %252 = memref.load %alloc_8[%c16] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %253 = cannir.scalar_storage_cast %252 : ui16 -> i16
                %254 = arith.extui %253 : i16 to i64
                %255 = arith.cmpi sge, %254, %c14464_i64 : i64
                scf.if %255 {
                  ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %256 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                  %257 = ascscalar.bitcast_memref %103 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%103, %256, %257, %alloc_4, %250 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_9 = arith.constant 1 : i32
                    %c0_i32_10 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c128_i32 = arith.constant 128 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c31_i16 = arith.constant 31 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %258 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %259 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %260 = ascvec.duplicate_scalar %c255_i16, %258 : i16 -> <ui16>
                    %261 = ascvec.duplicate_scalar %c-128_i8, %259 : i8 -> <ui8>
                    %262 = ascvec.duplicate_scalar %c31_i16, %258 : i16 -> <ui16>
                    scf.for %arg13 = %c0_i32_10 to %arg12 step %c1_i32_9  : i32 {
                      %263 = arith.muli %arg13, %c256_i32 : i32
                      %264 = arith.index_cast %263 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg8[%264] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %265 = arith.muli %arg13, %c128_i32 : i32
                      %266 = arith.index_cast %265 : i32 to index
                      %267 = ascvec.loadalign %arg9[%266] : memref<?xui8, 1>, index -> <ui8>
                      %268 = ascvec.unpack %267 {part = "lower"} : <ui8> -> <ui16>
                      %269 = ascvec.and %268, %262, %258 : <ui16>, <ui16> -> <ui16>
                      %270 = ascvec.shr %268, 3, %258 : <ui16> -> <ui16>
                      %271 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %272 = ascvec.add %271, %269, %258 : <ui16>
                      %273 = ascvec.shr %272, 5, %258 : <ui16> -> <ui16>
                      %274 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %275 = ascvec.add %274, %270, %258 : <ui16>
                      %276 = ascvec.shr %275, 5, %258 : <ui16> -> <ui16>
                      %277 = ascvec.and %273, %260, %258 : <ui16>, <ui16> -> <ui16>
                      %278 = ascvec.shl %276, 8, %258 : <ui16> -> <ui16>
                      %279 = ascvec.or %277, %278, %258 : <ui16>, <ui16> -> <ui16>
                      %280 = ascvec.reinterpret %279 : <ui16> -> <ui8>
                      %281 = ascvec.add_scalar %280, %c64_i8, %259 : <ui8>, i8
                      %282 = ascvec.min_scalar %281, %c123_i8, %259 : <ui8>, i8
                      %283 = arith.muli %arg13, %c512_i32 : i32
                      %284 = arith.index_cast %283 : i32 to index
                      %result0_11, %result1_12 = ascvec.loadalign_deintlv %arg10[%284] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %285 = ascvec.and %result1_12, %261, %259 : <ui8>, <ui8> -> <ui8>
                      %286 = ascvec.or %282, %285, %259 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg11[%264], %286, %259 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
                } else {
                  scf.for %arg8 = %c0_i64 to %199 step %c16_i64  : i64 {
                    %256 = arith.addi %arg8, %c16_i64 : i64
                    %257 = arith.cmpi sgt, %256, %199 : i64
                    %258 = arith.select %257, %199, %256 : i64
                    %259 = arith.muli %arg8, %c16_i64 : i64
                    ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %260 = arith.index_cast %259 : i64 to index
                    %261 = memref.load %alloc_8[%260] : memref<2048xui16, 1>
                    ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %262 = cannir.scalar_storage_cast %261 : ui16 -> i16
                    %263 = arith.extui %262 : i16 to i64
                    %264 = arith.addi %arg8, %c4_i64 : i64
                    %265 = scf.for %arg9 = %264 to %258 step %c4_i64 iter_args(%arg10 = %263) -> (i64)  : i64 {
                      %267 = arith.muli %arg9, %c16_i64 : i64
                      ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %268 = arith.index_cast %267 : i64 to index
                      %269 = memref.load %alloc_8[%268] : memref<2048xui16, 1>
                      ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %270 = cannir.scalar_storage_cast %269 : ui16 -> i16
                      %271 = arith.extui %270 : i16 to i64
                      %272 = arith.cmpi slt, %271, %arg10 : i64
                      %273 = arith.select %272, %271, %arg10 : i64
                      scf.yield %273 : i64
                    }
                    %266 = arith.cmpi sge, %265, %c14464_i64 : i64
                    scf.if %266 {
                      %267 = arith.trunci %arg8 : i64 to i32
                      %268 = arith.trunci %258 : i64 to i32
                      ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %269 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                      %270 = ascscalar.bitcast_memref %103 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                      ascvec.vf(%103, %269, %270, %alloc_4, %267, %268 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                      ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32, %arg14: i32):
                        %c512_i32 = arith.constant 512 : i32
                        %c1_i32_9 = arith.constant 1 : i32
                        %c123_i8 = arith.constant 123 : i8
                        %c64_i8 = arith.constant 64 : i8
                        %c128_i32 = arith.constant 128 : i32
                        %c256_i32 = arith.constant 256 : i32
                        %c31_i16 = arith.constant 31 : i16
                        %c-128_i8 = arith.constant -128 : i8
                        %c255_i16 = arith.constant 255 : i16
                        %271 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                        %272 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                        %273 = ascvec.duplicate_scalar %c255_i16, %271 : i16 -> <ui16>
                        %274 = ascvec.duplicate_scalar %c-128_i8, %272 : i8 -> <ui8>
                        %275 = ascvec.duplicate_scalar %c31_i16, %271 : i16 -> <ui16>
                        scf.for %arg15 = %arg13 to %arg14 step %c1_i32_9  : i32 {
                          %276 = arith.muli %arg15, %c256_i32 : i32
                          %277 = arith.index_cast %276 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg9[%277] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                          %278 = arith.muli %arg15, %c128_i32 : i32
                          %279 = arith.index_cast %278 : i32 to index
                          %280 = ascvec.loadalign %arg10[%279] : memref<?xui8, 1>, index -> <ui8>
                          %281 = ascvec.unpack %280 {part = "lower"} : <ui8> -> <ui16>
                          %282 = ascvec.and %281, %275, %271 : <ui16>, <ui16> -> <ui16>
                          %283 = ascvec.shr %281, 3, %271 : <ui16> -> <ui16>
                          %284 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                          %285 = ascvec.add %284, %282, %271 : <ui16>
                          %286 = ascvec.shr %285, 5, %271 : <ui16> -> <ui16>
                          %287 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                          %288 = ascvec.add %287, %283, %271 : <ui16>
                          %289 = ascvec.shr %288, 5, %271 : <ui16> -> <ui16>
                          %290 = ascvec.and %286, %273, %271 : <ui16>, <ui16> -> <ui16>
                          %291 = ascvec.shl %289, 8, %271 : <ui16> -> <ui16>
                          %292 = ascvec.or %290, %291, %271 : <ui16>, <ui16> -> <ui16>
                          %293 = ascvec.reinterpret %292 : <ui16> -> <ui8>
                          %294 = ascvec.add_scalar %293, %c64_i8, %272 : <ui8>, i8
                          %295 = ascvec.min_scalar %294, %c123_i8, %272 : <ui8>, i8
                          %296 = arith.muli %arg15, %c512_i32 : i32
                          %297 = arith.index_cast %296 : i32 to index
                          %result0_10, %result1_11 = ascvec.loadalign_deintlv %arg11[%297] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                          %298 = ascvec.and %result1_11, %274, %272 : <ui8>, <ui8> -> <ui8>
                          %299 = ascvec.or %295, %298, %272 : <ui8>, <ui8> -> <ui8>
                          ascvec.storealign %arg12[%277], %299, %272 : memref<32768xui8, 1>, index, <ui8>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
                    } else {
                      %267 = arith.cmpi sge, %265, %c13568_i64 : i64
                      scf.if %267 {
                        %268 = arith.trunci %arg8 : i64 to i32
                        %269 = arith.trunci %258 : i64 to i32
                        ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %270 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                        %271 = ascscalar.bitcast_memref %103 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%103, %270, %271, %alloc_4, %268, %269 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32, %arg14: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_9 = arith.constant 1 : i32
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
                          %272 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %273 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %274 = ascvec.duplicate_scalar %c255_i16, %272 : i16 -> <ui16>
                          %275 = ascvec.duplicate_scalar %c-128_i8, %273 : i8 -> <ui8>
                          %276 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %277 = ascvec.shr %276, 1, %273 : <ui8> -> <ui8>
                          %278 = ascvec.shr %277, 2, %273 : <ui8> -> <ui8>
                          %279 = ascvec.shl %278, 2, %273 : <ui8> -> <ui8>
                          %280 = ascvec.duplicate_scalar %c1_i8, %273 : i8 -> <ui8>
                          %281 = ascvec.and %276, %280, %273 : <ui8>, <ui8> -> <ui8>
                          %282 = ascvec.add_scalar %277, %c3_i8, %273 : <ui8>, i8
                          %283 = ascvec.add %282, %281, %273 : <ui8>
                          %284 = ascvec.duplicate_scalar %c3_i8, %273 : i8 -> <ui8>
                          %285 = ascvec.and %283, %284, %273 : <ui8>, <ui8> -> <ui8>
                          %286 = ascvec.or %279, %285, %273 : <ui8>, <ui8> -> <ui8>
                          %287 = ascvec.duplicate_scalar %c4095_i16, %272 : i16 -> <ui16>
                          %288 = ascvec.duplicate_scalar %c32767_i16, %272 : i16 -> <ui16>
                          %289 = ascvec.duplicate_scalar %cst, %272 : bf16 -> <bf16>
                          scf.for %arg15 = %arg13 to %arg14 step %c1_i32_9  : i32 {
                            %290 = arith.muli %arg15, %c256_i32 : i32
                            %291 = arith.index_cast %290 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg9[%291] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %292 = arith.muli %arg15, %c128_i32 : i32
                            %293 = arith.index_cast %292 : i32 to index
                            %294 = ascvec.loadalign %arg10[%293] : memref<?xui8, 1>, index -> <ui8>
                            %295 = ascvec.gather_reg %294, %286 : <ui8>, <ui8> -> <ui8>
                            %296 = ascvec.reinterpret %295 : <ui8> -> <ui16>
                            %297 = ascvec.shr %296, 1, %272 : <ui16> -> <ui16>
                            %298 = ascvec.and %297, %287, %272 : <ui16>, <ui16> -> <ui16>
                            %299 = ascvec.shr %296, 4, %272 : <ui16> -> <ui16>
                            %300 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %301 = ascvec.and %300, %288, %272 : <ui16>, <ui16> -> <ui16>
                            %302 = ascvec.cmp_lt_scalar %301, %c14464_i16, %272 : <ui16>, i16 -> !ascvec.mask
                            %303 = ascvec.shr %298, 7, %272 : <ui16> -> <ui16>
                            %304 = ascvec.add %301, %303, %272 : <ui16>
                            %305 = ascvec.shr %304, 5, %272 : <ui16> -> <ui16>
                            %306 = ascvec.reinterpret %301 : <ui16> -> <bf16>
                            %307 = ascvec.mul %306, %289, %302 : <bf16>
                            %308 = ascvec.cast %307, %272 {rounding = 4 : i32} : <bf16> -> <f16>
                            %309 = ascvec.cast %308, %272 {rounding = 4 : i32} : <f16> -> <i16>
                            %310 = ascvec.reinterpret %309 : <i16> -> <ui16>
                            %311 = ascvec.add %310, %298, %272 : <ui16>
                            %312 = ascvec.shr %311, 12, %272 : <ui16> -> <ui16>
                            %313 = ascvec.add_scalar %312, %c448_i16, %272 : <ui16>, i16
                            %314 = ascvec.select %313, %305, %302 : <ui16>
                            %315 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %316 = ascvec.and %315, %288, %272 : <ui16>, <ui16> -> <ui16>
                            %317 = ascvec.cmp_lt_scalar %316, %c14464_i16, %272 : <ui16>, i16 -> !ascvec.mask
                            %318 = ascvec.shr %299, 7, %272 : <ui16> -> <ui16>
                            %319 = ascvec.add %316, %318, %272 : <ui16>
                            %320 = ascvec.shr %319, 5, %272 : <ui16> -> <ui16>
                            %321 = ascvec.reinterpret %316 : <ui16> -> <bf16>
                            %322 = ascvec.mul %321, %289, %317 : <bf16>
                            %323 = ascvec.cast %322, %272 {rounding = 4 : i32} : <bf16> -> <f16>
                            %324 = ascvec.cast %323, %272 {rounding = 4 : i32} : <f16> -> <i16>
                            %325 = ascvec.reinterpret %324 : <i16> -> <ui16>
                            %326 = ascvec.add %325, %299, %272 : <ui16>
                            %327 = ascvec.shr %326, 12, %272 : <ui16> -> <ui16>
                            %328 = ascvec.add_scalar %327, %c448_i16, %272 : <ui16>, i16
                            %329 = ascvec.select %328, %320, %317 : <ui16>
                            %330 = ascvec.and %314, %274, %272 : <ui16>, <ui16> -> <ui16>
                            %331 = ascvec.shl %329, 8, %272 : <ui16> -> <ui16>
                            %332 = ascvec.or %330, %331, %272 : <ui16>, <ui16> -> <ui16>
                            %333 = ascvec.reinterpret %332 : <ui16> -> <ui8>
                            %334 = ascvec.add_scalar %333, %c64_i8, %273 : <ui8>, i8
                            %335 = ascvec.min_scalar %334, %c123_i8, %273 : <ui8>, i8
                            %336 = arith.muli %arg15, %c512_i32 : i32
                            %337 = arith.index_cast %336 : i32 to index
                            %result0_10, %result1_11 = ascvec.loadalign_deintlv %arg11[%337] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %338 = ascvec.and %result1_11, %275, %273 : <ui8>, <ui8> -> <ui8>
                            %339 = ascvec.or %335, %338, %273 : <ui8>, <ui8> -> <ui8>
                            ascvec.storealign %arg12[%291], %339, %273 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
                      } else {
                        %268 = arith.trunci %arg8 : i64 to i32
                        %269 = arith.trunci %258 : i64 to i32
                        ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %270 = ascscalar.bitcast_memref %103 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%103, %alloc_7, %270, %251, %alloc_4, %268, %269 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<8192xi32, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<?xui16, 1>, %arg13: memref<32768xui8, 1>, %arg14: i32, %arg15: i32):
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
                          %c4_i32_9 = arith.constant 4 : i32
                          %c2_i32_10 = arith.constant 2 : i32
                          %c1_i32_11 = arith.constant 1 : i32
                          %c7_i32 = arith.constant 7 : i32
                          %c32_i32 = arith.constant 32 : i32
                          %c0_i32_12 = arith.constant 0 : i32
                          %c32767_i16 = arith.constant 32767 : i16
                          %c0_13 = arith.constant 0 : index
                          %c256_i32 = arith.constant 256 : i32
                          %c3_i8 = arith.constant 3 : i8
                          %c-128_i8 = arith.constant -128 : i8
                          %c0_i8 = arith.constant 0 : i8
                          %c8_i32 = arith.constant 8 : i32
                          %271 = ascvec.create_mask : !ascvec.mask
                          %272 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %273 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                          %274 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %275 = ascvec.duplicate_scalar %c-128_i8, %273 : i8 -> <ui8>
                          %276 = ascvec.shr %274, 5, %273 : <ui8> -> <ui8>
                          %277 = ascvec.duplicate_scalar %c3_i8, %273 : i8 -> <ui8>
                          %278 = ascvec.and %274, %277, %273 : <ui8>, <ui8> -> <ui8>
                          %279 = ascvec.shl %278, 6, %273 : <ui8> -> <ui8>
                          %280 = ascvec.shr %274, 2, %273 : <ui8> -> <ui8>
                          %281 = ascvec.or %279, %280, %273 : <ui8>, <ui8> -> <ui8>
                          scf.for %arg16 = %arg14 to %arg15 step %c1_i32_11  : i32 {
                            %282 = arith.muli %arg16, %c256_i32 : i32
                            %283 = arith.index_cast %282 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg9[%283] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %284 = ascvec.duplicate_scalar %c32767_i16, %272 : i16 -> <ui16>
                            %285 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %286 = ascvec.and %285, %284, %272 : <ui16>, <ui16> -> <ui16>
                            %287 = ascvec.reinterpret %286 : <ui16> -> <bf16>
                            %288 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %289 = ascvec.and %288, %284, %272 : <ui16>, <ui16> -> <ui16>
                            %290 = ascvec.reinterpret %289 : <ui16> -> <bf16>
                            %291 = ascvec.cast %287, %272 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %292 = ascvec.cast %287, %272 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %293 = ascvec.cast %290, %272 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %294 = ascvec.cast %290, %272 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %295 = ascvec.arange %c0_i32_12 : i32 -> <ui32>
                            %296 = ascvec.shl %295, 2, %271 : <ui32> -> <ui32>
                            %297 = ascvec.add_scalar %296, %c0_i32_12, %271 : <ui32>, i32
                            %298 = ascvec.shr %297, 3, %271 : <ui32> -> <ui32>
                            %299 = arith.muli %arg16, %c32_i32 : i32
                            %300 = ascvec.add_scalar %298, %299, %271 : <ui32>, i32
                            %301 = ascvec.gather %arg10[%c0_13], %300, %271 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %302 = ascvec.reinterpret %301 : <i32> -> <ui32>
                            %303 = ascvec.duplicate_scalar %c7_i32, %271 : i32 -> <ui32>
                            %304 = ascvec.and %297, %303, %271 : <ui32>, <ui32> -> <ui32>
                            %305 = ascvec.shl %302, 11, %271 : <ui32> -> <ui32>
                            %306 = ascvec.shr %302, 21, %271 : <ui32> -> <ui32>
                            %307 = ascvec.or %305, %306, %271 : <ui32>, <ui32> -> <ui32>
                            %308 = ascvec.shr %307, 3, %271 : <ui32> -> <ui32>
                            %309 = ascvec.shl %307, 29, %271 : <ui32> -> <ui32>
                            %310 = ascvec.or %308, %309, %271 : <ui32>, <ui32> -> <ui32>
                            %311 = ascvec.duplicate_scalar %c1_i32_11, %271 : i32 -> <ui32>
                            %312 = ascvec.and %304, %311, %271 : <ui32>, <ui32> -> <ui32>
                            %313 = ascvec.cmp_ne_scalar %312, %c0_i32_12, %271 : <ui32>, i32 -> !ascvec.mask
                            %314 = ascvec.select %310, %307, %313 : <ui32>
                            %315 = ascvec.shr %314, 8, %271 : <ui32> -> <ui32>
                            %316 = ascvec.shl %314, 24, %271 : <ui32> -> <ui32>
                            %317 = ascvec.or %315, %316, %271 : <ui32>, <ui32> -> <ui32>
                            %318 = ascvec.duplicate_scalar %c2_i32_10, %271 : i32 -> <ui32>
                            %319 = ascvec.and %304, %318, %271 : <ui32>, <ui32> -> <ui32>
                            %320 = ascvec.cmp_ne_scalar %319, %c0_i32_12, %271 : <ui32>, i32 -> !ascvec.mask
                            %321 = ascvec.select %317, %314, %320 : <ui32>
                            %322 = ascvec.shr %321, 16, %271 : <ui32> -> <ui32>
                            %323 = ascvec.shl %321, 16, %271 : <ui32> -> <ui32>
                            %324 = ascvec.or %322, %323, %271 : <ui32>, <ui32> -> <ui32>
                            %325 = ascvec.duplicate_scalar %c4_i32_9, %271 : i32 -> <ui32>
                            %326 = ascvec.and %304, %325, %271 : <ui32>, <ui32> -> <ui32>
                            %327 = ascvec.cmp_ne_scalar %326, %c0_i32_12, %271 : <ui32>, i32 -> !ascvec.mask
                            %328 = ascvec.select %324, %321, %327 : <ui32>
                            %329 = ascvec.duplicate_scalar %c65535_i32, %271 : i32 -> <ui32>
                            %330 = ascvec.and %328, %329, %271 : <ui32>, <ui32> -> <ui32>
                            %331 = ascvec.add_scalar %296, %c1_i32_11, %271 : <ui32>, i32
                            %332 = ascvec.shr %331, 3, %271 : <ui32> -> <ui32>
                            %333 = ascvec.add_scalar %332, %299, %271 : <ui32>, i32
                            %334 = ascvec.gather %arg10[%c0_13], %333, %271 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %335 = ascvec.reinterpret %334 : <i32> -> <ui32>
                            %336 = ascvec.and %331, %303, %271 : <ui32>, <ui32> -> <ui32>
                            %337 = ascvec.shl %335, 11, %271 : <ui32> -> <ui32>
                            %338 = ascvec.shr %335, 21, %271 : <ui32> -> <ui32>
                            %339 = ascvec.or %337, %338, %271 : <ui32>, <ui32> -> <ui32>
                            %340 = ascvec.shr %339, 3, %271 : <ui32> -> <ui32>
                            %341 = ascvec.shl %339, 29, %271 : <ui32> -> <ui32>
                            %342 = ascvec.or %340, %341, %271 : <ui32>, <ui32> -> <ui32>
                            %343 = ascvec.and %336, %311, %271 : <ui32>, <ui32> -> <ui32>
                            %344 = ascvec.cmp_ne_scalar %343, %c0_i32_12, %271 : <ui32>, i32 -> !ascvec.mask
                            %345 = ascvec.select %342, %339, %344 : <ui32>
                            %346 = ascvec.shr %345, 8, %271 : <ui32> -> <ui32>
                            %347 = ascvec.shl %345, 24, %271 : <ui32> -> <ui32>
                            %348 = ascvec.or %346, %347, %271 : <ui32>, <ui32> -> <ui32>
                            %349 = ascvec.and %336, %318, %271 : <ui32>, <ui32> -> <ui32>
                            %350 = ascvec.cmp_ne_scalar %349, %c0_i32_12, %271 : <ui32>, i32 -> !ascvec.mask
                            %351 = ascvec.select %348, %345, %350 : <ui32>
                            %352 = ascvec.shr %351, 16, %271 : <ui32> -> <ui32>
                            %353 = ascvec.shl %351, 16, %271 : <ui32> -> <ui32>
                            %354 = ascvec.or %352, %353, %271 : <ui32>, <ui32> -> <ui32>
                            %355 = ascvec.and %336, %325, %271 : <ui32>, <ui32> -> <ui32>
                            %356 = ascvec.cmp_ne_scalar %355, %c0_i32_12, %271 : <ui32>, i32 -> !ascvec.mask
                            %357 = ascvec.select %354, %351, %356 : <ui32>
                            %358 = ascvec.and %357, %329, %271 : <ui32>, <ui32> -> <ui32>
                            %359 = ascvec.add_scalar %296, %c2_i32_10, %271 : <ui32>, i32
                            %360 = ascvec.shr %359, 3, %271 : <ui32> -> <ui32>
                            %361 = ascvec.add_scalar %360, %299, %271 : <ui32>, i32
                            %362 = ascvec.gather %arg10[%c0_13], %361, %271 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %363 = ascvec.reinterpret %362 : <i32> -> <ui32>
                            %364 = ascvec.and %359, %303, %271 : <ui32>, <ui32> -> <ui32>
                            %365 = ascvec.shl %363, 11, %271 : <ui32> -> <ui32>
                            %366 = ascvec.shr %363, 21, %271 : <ui32> -> <ui32>
                            %367 = ascvec.or %365, %366, %271 : <ui32>, <ui32> -> <ui32>
                            %368 = ascvec.shr %367, 3, %271 : <ui32> -> <ui32>
                            %369 = ascvec.shl %367, 29, %271 : <ui32> -> <ui32>
                            %370 = ascvec.or %368, %369, %271 : <ui32>, <ui32> -> <ui32>
                            %371 = ascvec.and %364, %311, %271 : <ui32>, <ui32> -> <ui32>
                            %372 = ascvec.cmp_ne_scalar %371, %c0_i32_12, %271 : <ui32>, i32 -> !ascvec.mask
                            %373 = ascvec.select %370, %367, %372 : <ui32>
                            %374 = ascvec.shr %373, 8, %271 : <ui32> -> <ui32>
                            %375 = ascvec.shl %373, 24, %271 : <ui32> -> <ui32>
                            %376 = ascvec.or %374, %375, %271 : <ui32>, <ui32> -> <ui32>
                            %377 = ascvec.and %364, %318, %271 : <ui32>, <ui32> -> <ui32>
                            %378 = ascvec.cmp_ne_scalar %377, %c0_i32_12, %271 : <ui32>, i32 -> !ascvec.mask
                            %379 = ascvec.select %376, %373, %378 : <ui32>
                            %380 = ascvec.shr %379, 16, %271 : <ui32> -> <ui32>
                            %381 = ascvec.shl %379, 16, %271 : <ui32> -> <ui32>
                            %382 = ascvec.or %380, %381, %271 : <ui32>, <ui32> -> <ui32>
                            %383 = ascvec.and %364, %325, %271 : <ui32>, <ui32> -> <ui32>
                            %384 = ascvec.cmp_ne_scalar %383, %c0_i32_12, %271 : <ui32>, i32 -> !ascvec.mask
                            %385 = ascvec.select %382, %379, %384 : <ui32>
                            %386 = ascvec.and %385, %329, %271 : <ui32>, <ui32> -> <ui32>
                            %387 = ascvec.add_scalar %296, %c3_i32, %271 : <ui32>, i32
                            %388 = ascvec.shr %387, 3, %271 : <ui32> -> <ui32>
                            %389 = ascvec.add_scalar %388, %299, %271 : <ui32>, i32
                            %390 = ascvec.gather %arg10[%c0_13], %389, %271 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %391 = ascvec.reinterpret %390 : <i32> -> <ui32>
                            %392 = ascvec.and %387, %303, %271 : <ui32>, <ui32> -> <ui32>
                            %393 = ascvec.shl %391, 11, %271 : <ui32> -> <ui32>
                            %394 = ascvec.shr %391, 21, %271 : <ui32> -> <ui32>
                            %395 = ascvec.or %393, %394, %271 : <ui32>, <ui32> -> <ui32>
                            %396 = ascvec.shr %395, 3, %271 : <ui32> -> <ui32>
                            %397 = ascvec.shl %395, 29, %271 : <ui32> -> <ui32>
                            %398 = ascvec.or %396, %397, %271 : <ui32>, <ui32> -> <ui32>
                            %399 = ascvec.and %392, %311, %271 : <ui32>, <ui32> -> <ui32>
                            %400 = ascvec.cmp_ne_scalar %399, %c0_i32_12, %271 : <ui32>, i32 -> !ascvec.mask
                            %401 = ascvec.select %398, %395, %400 : <ui32>
                            %402 = ascvec.shr %401, 8, %271 : <ui32> -> <ui32>
                            %403 = ascvec.shl %401, 24, %271 : <ui32> -> <ui32>
                            %404 = ascvec.or %402, %403, %271 : <ui32>, <ui32> -> <ui32>
                            %405 = ascvec.and %392, %318, %271 : <ui32>, <ui32> -> <ui32>
                            %406 = ascvec.cmp_ne_scalar %405, %c0_i32_12, %271 : <ui32>, i32 -> !ascvec.mask
                            %407 = ascvec.select %404, %401, %406 : <ui32>
                            %408 = ascvec.shr %407, 16, %271 : <ui32> -> <ui32>
                            %409 = ascvec.shl %407, 16, %271 : <ui32> -> <ui32>
                            %410 = ascvec.or %408, %409, %271 : <ui32>, <ui32> -> <ui32>
                            %411 = ascvec.and %392, %325, %271 : <ui32>, <ui32> -> <ui32>
                            %412 = ascvec.cmp_ne_scalar %411, %c0_i32_12, %271 : <ui32>, i32 -> !ascvec.mask
                            %413 = ascvec.select %410, %407, %412 : <ui32>
                            %414 = ascvec.and %413, %329, %271 : <ui32>, <ui32> -> <ui32>
                            %415 = ascvec.reinterpret %291 : <f32> -> <ui32>
                            %416 = ascvec.reinterpret %415 : <ui32> -> <f32>
                            %417 = ascvec.max_scalar %416, %cst, %271 : <f32>, f32
                            %418 = ascvec.add %416, %417, %271 : <f32>
                            %419 = ascvec.reinterpret %418 : <f32> -> <ui32>
                            %420 = ascvec.shr %419, 5, %271 : <ui32> -> <ui32>
                            %421 = ascvec.add %420, %330, %271 : <ui32>
                            %422 = ascvec.reinterpret %293 : <f32> -> <ui32>
                            %423 = ascvec.reinterpret %422 : <ui32> -> <f32>
                            %424 = ascvec.max_scalar %423, %cst, %271 : <f32>, f32
                            %425 = ascvec.add %423, %424, %271 : <f32>
                            %426 = ascvec.reinterpret %425 : <f32> -> <ui32>
                            %427 = ascvec.shr %426, 5, %271 : <ui32> -> <ui32>
                            %428 = ascvec.add %427, %358, %271 : <ui32>
                            %429 = ascvec.reinterpret %292 : <f32> -> <ui32>
                            %430 = ascvec.reinterpret %429 : <ui32> -> <f32>
                            %431 = ascvec.max_scalar %430, %cst, %271 : <f32>, f32
                            %432 = ascvec.add %430, %431, %271 : <f32>
                            %433 = ascvec.reinterpret %432 : <f32> -> <ui32>
                            %434 = ascvec.shr %433, 5, %271 : <ui32> -> <ui32>
                            %435 = ascvec.add %434, %386, %271 : <ui32>
                            %436 = ascvec.reinterpret %294 : <f32> -> <ui32>
                            %437 = ascvec.reinterpret %436 : <ui32> -> <f32>
                            %438 = ascvec.max_scalar %437, %cst, %271 : <f32>, f32
                            %439 = ascvec.add %437, %438, %271 : <f32>
                            %440 = ascvec.reinterpret %439 : <f32> -> <ui32>
                            %441 = ascvec.shr %440, 5, %271 : <ui32> -> <ui32>
                            %442 = ascvec.add %441, %414, %271 : <ui32>
                            %443 = ascvec.reinterpret %421 : <ui32> -> <ui16>
                            %444 = ascvec.reinterpret %428 : <ui32> -> <ui16>
                            %result0_14, %result1_15 = ascvec.deintlv %443, %444 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %445 = ascvec.reinterpret %435 : <ui32> -> <ui16>
                            %446 = ascvec.reinterpret %442 : <ui32> -> <ui16>
                            %result0_16, %result1_17 = ascvec.deintlv %445, %446 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %447 = ascvec.reinterpret %result1_15 : <ui16> -> <ui8>
                            %448 = ascvec.reinterpret %result1_17 : <ui16> -> <ui8>
                            %result0_18, %result1_19 = ascvec.deintlv %447, %448 : <ui8>, <ui8> -> <ui8>, <ui8>
                            %449 = ascvec.gather_reg %result0_18, %281 : <ui8>, <ui8> -> <ui8>
                            %450 = ascvec.add_scalar %449, %c60_i8, %273 : <ui8>, i8
                            %451 = ascvec.cmp_ne_scalar %450, %c-1_i8, %273 : <ui8>, i8 -> !ascvec.mask
                            %452 = ascvec.min_scalar %450, %c123_i8, %451 : <ui8>, i8
                            %453 = arith.muli %arg16, %c512_i32 : i32
                            %454 = arith.index_cast %453 : i32 to index
                            %result0_20, %result1_21 = ascvec.loadalign_deintlv %arg11[%454] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %455 = ascvec.and %result1_21, %275, %273 : <ui8>, <ui8> -> <ui8>
                            %456 = ascvec.or %452, %455, %273 : <ui8>, <ui8> -> <ui8>
                            %457 = ascvec.arange %c0_i16 : i16 -> <ui16>
                            %458 = arith.muli %arg16, %c16_i32 : i32
                            %459 = arith.trunci %458 : i32 to i16
                            %460 = ascvec.add_scalar %457, %459, %272 : <ui16>, i16
                            %461 = ascvec.gather %arg12[%c0_13], %460, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                            %462 = ascvec.pack %461 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                            %463 = ascvec.gather_reg %462, %276 : <ui8>, <ui8> -> <ui8>
                            %464 = ascvec.cmp_eq_scalar %463, %c-1_i8, %273 : <ui8>, i8 -> !ascvec.mask
                            %465 = ascvec.duplicate_scalar %c127_i8, %464 merge %456 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                            ascvec.storealign %arg13[%283], %465, %273 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
                      }
                    }
                  }
                }
              } else {
                %199 = arith.divsi %62, %c256_i64 : i64
                %200 = arith.addi %62, %c1023_i64 : i64
                %201 = arith.divsi %200, %c1024_i64 : i64
                %202 = arith.trunci %arg4 : i64 to i32
                %203 = arith.extui %202 : i32 to i64
                %204 = arith.subi %arg4, %203 : i64
                %205 = arith.remsi %204, %c4294967296_i64 : i64
                %206 = arith.cmpi ne, %205, %c0_i64 : i64
                %207 = arith.cmpi slt, %205, %c0_i64 : i64
                %208 = arith.andi %206, %207 : i1
                %209 = arith.divsi %204, %c4294967296_i64 : i64
                %210 = arith.extui %208 : i1 to i64
                %211 = arith.subi %209, %210 : i64
                %212 = arith.trunci %211 : i64 to i32
                %213 = arith.divui %104, %c32_i64 : i64
                %214 = arith.trunci %213 : i64 to i32
                %215 = arith.extui %214 : i32 to i64
                %216 = arith.subi %213, %215 : i64
                %217 = arith.remsi %216, %c4294967296_i64 : i64
                %218 = arith.cmpi ne, %217, %c0_i64 : i64
                %219 = arith.cmpi slt, %217, %c0_i64 : i64
                %220 = arith.andi %218, %219 : i1
                %221 = arith.divsi %216, %c4294967296_i64 : i64
                %222 = arith.extui %220 : i1 to i64
                %223 = arith.subi %221, %222 : i64
                %224 = arith.trunci %223 : i64 to i32
                %225 = arith.trunci %arg5 : i64 to i32
                %226 = arith.extui %225 : i32 to i64
                %227 = arith.subi %arg5, %226 : i64
                %228 = arith.remsi %227, %c4294967296_i64 : i64
                %229 = arith.cmpi ne, %228, %c0_i64 : i64
                %230 = arith.cmpi slt, %228, %c0_i64 : i64
                %231 = arith.andi %229, %230 : i1
                %232 = arith.divsi %227, %c4294967296_i64 : i64
                %233 = arith.extui %231 : i1 to i64
                %234 = arith.subi %232, %233 : i64
                %235 = arith.trunci %234 : i64 to i32
                %236 = arith.divui %225, %c4_i32 : i32
                %237 = arith.remui %235, %c4_i32 : i32
                %238 = arith.muli %237, %c1073741824_i32 : i32
                %239 = arith.addi %236, %238 : i32
                %240 = arith.divui %235, %c4_i32 : i32
                %241 = arith.muli %201, %c32_i64 : i64
                %242 = arith.divui %213, %c16384_i64 : i64
                %243 = arith.addi %213, %241 : i64
                %244 = arith.subi %243, %c1_i64 : i64
                %245 = arith.divui %244, %c16384_i64 : i64
                %246 = arith.cmpi eq, %242, %245 : i64
                scf.if %246 {
                  %256 = arith.trunci %242 : i64 to i32
                  %257 = arith.extui %256 : i32 to i64
                  %258 = arith.subi %242, %257 : i64
                  %259 = arith.remsi %258, %c4294967296_i64 : i64
                  %260 = arith.cmpi ne, %259, %c0_i64 : i64
                  %261 = arith.cmpi slt, %259, %c0_i64 : i64
                  %262 = arith.andi %260, %261 : i1
                  %263 = arith.divsi %258, %c4294967296_i64 : i64
                  %264 = arith.extui %262 : i1 to i64
                  %265 = arith.subi %263, %264 : i64
                  %266 = arith.trunci %265 : i64 to i32
                  %267 = arith.addi %239, %256 : i32
                  %268 = arith.addi %240, %266 : i32
                  %269 = arith.cmpi ult, %267, %239 : i32
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
                  %285 = arith.xori %271, %212 : i32
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
                  %299 = arith.addi %202, %c-1640531527_i32 : i32
                  %300 = arith.addi %212, %c-1150833019_i32 : i32
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
                  %316 = arith.xori %284, %202 : i32
                  %317 = arith.addi %202, %c1013904242_i32 : i32
                  %318 = arith.xori %288, %317 : i32
                  %319 = arith.addi %212, %c1993301258_i32 : i32
                  %320 = arith.xori %315, %319 : i32
                  ascsimt.vf(%214, %316, %300, %318, %320, %202, %212, %305, %alloc_7, %241 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: memref<8192xi32, 1>, %arg17: i64):
                    %c256_i64_9 = arith.constant 256 : i64
                    %c3_i64_10 = arith.constant 3 : i64
                    %c2_i64_11 = arith.constant 2 : i64
                    %c1_i64_12 = arith.constant 1 : i64
                    %c4_i64_13 = arith.constant 4 : i64
                    %c1_i32_14 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c3_i32 = arith.constant 3 : i32
                    %c-1150833019_i32_15 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_16 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %321 = ascsimt.thread_idx {ascsimt.scope = 8 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %322 = arith.extsi %321 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg18 = %322 to %arg17 step %c256_i64_9  : i64 {
                      %323 = arith.trunci %arg18 : i64 to i32
                      %324 = arith.addi %arg8, %323 : i32
                      %325 = arith.remui %324, %c16384_i32 : i32
                      %326 = arith.xori %325, %arg9 : i32
                      %327 = arith.muli %326, %c-766435501_i32 : i32
                      %328 = "ascsimt.bitcast"(%326) : (i32) -> ui32
                      %329 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %330 = ascsimt.math_binary %328, %329, "mulhi" : ui32, ui32 -> ui32
                      %331 = "ascsimt.bitcast"(%330) : (ui32) -> i32
                      %332 = arith.xori %331, %arg10 : i32
                      %333 = arith.muli %332, %c-845247145_i32 : i32
                      %334 = "ascsimt.bitcast"(%332) : (i32) -> ui32
                      %335 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %336 = ascsimt.math_binary %334, %335, "mulhi" : ui32, ui32 -> ui32
                      %337 = "ascsimt.bitcast"(%336) : (ui32) -> i32
                      %338 = arith.xori %337, %arg11 : i32
                      %339 = arith.xori %327, %arg12 : i32
                      %340:4 = scf.for %arg19 = %c3_i32 to %c10_i32 step %c1_i32_14 iter_args(%arg20 = %338, %arg21 = %333, %arg22 = %339, %arg23 = %arg15) -> (i32, i32, i32, i32)  : i32 {
                        %349 = arith.muli %arg20, %c-766435501_i32 : i32
                        %350 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                        %351 = ascsimt.math_binary %350, %329, "mulhi" : ui32, ui32 -> ui32
                        %352 = "ascsimt.bitcast"(%351) : (ui32) -> i32
                        %353 = arith.muli %arg22, %c-845247145_i32 : i32
                        %354 = "ascsimt.bitcast"(%arg22) : (i32) -> ui32
                        %355 = ascsimt.math_binary %354, %335, "mulhi" : ui32, ui32 -> ui32
                        %356 = "ascsimt.bitcast"(%355) : (ui32) -> i32
                        %357 = arith.muli %arg19, %c-1640531527_i32_16 : i32
                        %358 = arith.addi %arg13, %357 : i32
                        %359 = arith.muli %arg19, %c-1150833019_i32_15 : i32
                        %360 = arith.addi %arg14, %359 : i32
                        %361 = arith.xori %356, %arg21 : i32
                        %362 = arith.xori %361, %358 : i32
                        %363 = arith.xori %352, %arg23 : i32
                        %364 = arith.xori %363, %360 : i32
                        scf.yield %362, %353, %364, %349 : i32, i32, i32, i32
                      }
                      %341 = arith.muli %arg18, %c4_i64_13 : i64
                      %342 = arith.index_cast %341 : i64 to index
                      ascsimt.store %arg16[%342], %340#0 : memref<8192xi32, 1>, i32
                      %343 = arith.addi %341, %c1_i64_12 : i64
                      %344 = arith.index_cast %343 : i64 to index
                      ascsimt.store %arg16[%344], %340#1 : memref<8192xi32, 1>, i32
                      %345 = arith.addi %341, %c2_i64_11 : i64
                      %346 = arith.index_cast %345 : i64 to index
                      ascsimt.store %arg16[%346], %340#2 : memref<8192xi32, 1>, i32
                      %347 = arith.addi %341, %c3_i64_10 : i64
                      %348 = arith.index_cast %347 : i64 to index
                      ascsimt.store %arg16[%348], %340#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%214, %224, %239, %240, %202, %212, %alloc_7, %241 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: memref<8192xi32, 1>, %arg15: i64):
                    %c256_i64_9 = arith.constant 256 : i64
                    %c3_i64_10 = arith.constant 3 : i64
                    %c2_i64_11 = arith.constant 2 : i64
                    %c1_i64_12 = arith.constant 1 : i64
                    %c4_i64_13 = arith.constant 4 : i64
                    %c1_i32_14 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c0_i32_15 = arith.constant 0 : i32
                    %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c262144_i32 = arith.constant 262144 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %256 = ascsimt.thread_idx {ascsimt.scope = 9 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %257 = arith.extsi %256 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg16 = %257 to %arg15 step %c256_i64_9  : i64 {
                      %258 = arith.trunci %arg16 : i64 to i32
                      %259 = arith.addi %arg8, %258 : i32
                      %260 = arith.cmpi ult, %259, %arg8 : i32
                      %261 = arith.extui %260 : i1 to i32
                      %262 = arith.addi %arg9, %261 : i32
                      %263 = arith.divui %259, %c16384_i32 : i32
                      %264 = arith.remui %262, %c16384_i32 : i32
                      %265 = arith.muli %264, %c262144_i32 : i32
                      %266 = arith.addi %263, %265 : i32
                      %267 = arith.divui %262, %c16384_i32 : i32
                      %268 = arith.addi %arg10, %266 : i32
                      %269 = arith.addi %arg11, %267 : i32
                      %270 = arith.cmpi ult, %268, %arg10 : i32
                      %271 = arith.extui %270 : i1 to i32
                      %272 = arith.addi %269, %271 : i32
                      %273 = arith.remui %259, %c16384_i32 : i32
                      %274:4 = scf.for %arg17 = %c0_i32_15 to %c10_i32 step %c1_i32_14 iter_args(%arg18 = %c0_i32_15, %arg19 = %273, %arg20 = %268, %arg21 = %272) -> (i32, i32, i32, i32)  : i32 {
                        %283 = arith.muli %arg18, %c-766435501_i32 : i32
                        %284 = "ascsimt.bitcast"(%arg18) : (i32) -> ui32
                        %285 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %286 = ascsimt.math_binary %284, %285, "mulhi" : ui32, ui32 -> ui32
                        %287 = "ascsimt.bitcast"(%286) : (ui32) -> i32
                        %288 = arith.muli %arg20, %c-845247145_i32 : i32
                        %289 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                        %290 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %291 = ascsimt.math_binary %289, %290, "mulhi" : ui32, ui32 -> ui32
                        %292 = "ascsimt.bitcast"(%291) : (ui32) -> i32
                        %293 = arith.muli %arg17, %c-1640531527_i32_17 : i32
                        %294 = arith.addi %arg12, %293 : i32
                        %295 = arith.muli %arg17, %c-1150833019_i32_16 : i32
                        %296 = arith.addi %arg13, %295 : i32
                        %297 = arith.xori %292, %arg19 : i32
                        %298 = arith.xori %297, %294 : i32
                        %299 = arith.xori %287, %arg21 : i32
                        %300 = arith.xori %299, %296 : i32
                        scf.yield %298, %288, %300, %283 : i32, i32, i32, i32
                      }
                      %275 = arith.muli %arg16, %c4_i64_13 : i64
                      %276 = arith.index_cast %275 : i64 to index
                      ascsimt.store %arg14[%276], %274#0 : memref<8192xi32, 1>, i32
                      %277 = arith.addi %275, %c1_i64_12 : i64
                      %278 = arith.index_cast %277 : i64 to index
                      ascsimt.store %arg14[%278], %274#1 : memref<8192xi32, 1>, i32
                      %279 = arith.addi %275, %c2_i64_11 : i64
                      %280 = arith.index_cast %279 : i64 to index
                      ascsimt.store %arg14[%280], %274#2 : memref<8192xi32, 1>, i32
                      %281 = arith.addi %275, %c3_i64_10 : i64
                      %282 = arith.index_cast %281 : i64 to index
                      ascsimt.store %arg14[%282], %274#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %247 = arith.trunci %44 : i64 to i32
                %248 = arith.divsi %247, %c2_i32 : i32
                %249 = arith.trunci %248 : i32 to i8
                %250 = arith.trunci %199 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %251 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%249, %alloc_5, %103, %251, %250, %alloc_8 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, i32, memref<2048xui16, 1>) {
                ^bb0(%arg8: i8, %arg9: memref<2048xui8, 1>, %arg10: memref<32768xbf16, 1>, %arg11: memref<?xui16, 1>, %arg12: i32, %arg13: memref<2048xui16, 1>):
                  %c1_i32_9 = arith.constant 1 : i32
                  %c0_i32_10 = arith.constant 0 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c0_11 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %256 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %257 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %258 = ascvec.duplicate_scalar %c32767_i16, %256 : i16 -> <ui16>
                  %259 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %260 = ascvec.shr %259, 4, %256 : <ui16> -> <ui16>
                  %261 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %262 = ascvec.duplicate_scalar %c1_i8, %257 : i8 -> <ui8>
                  %263 = ascvec.and %261, %262, %257 : <ui8>, <ui8> -> <ui8>
                  %264 = ascvec.cmp_lt_scalar %261, %c16_i8, %257 : <ui8>, i8 -> !ascvec.mask
                  %265 = ascvec.cmp_lt_scalar %263, %arg8, %257 : <ui8>, i8 -> !ascvec.mask
                  %266 = ascvec.mask_and %264, %265, %257
                  ascvec.store_unalign_begin %arg9 : memref<2048xui8, 1>
                  %267 = ascvec.duplicate_scalar %c-1_i16, %256 : i16 -> <ui16>
                  %268 = ascvec.duplicate_scalar %c0_i16, %256 : i16 -> <ui16>
                  %269:2 = scf.for %arg14 = %c0_i32_10 to %arg12 step %c1_i32_9 iter_args(%arg15 = %268, %arg16 = %267) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i32 {
                    %270 = arith.muli %arg14, %c256_i32 : i32
                    %271 = arith.index_cast %270 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg10[%271] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %272 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %273 = ascvec.and %272, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %274 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %275 = ascvec.and %274, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %276 = ascvec.max %273, %275, %256 : <ui16>
                    %277 = ascvec.reduce_max_datablock %276, %256 : <ui16>
                    %278 = ascvec.shr %277, 7, %256 : <ui16> -> <ui16>
                    %279 = ascvec.cmp_ge_scalar %278, %c255_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %280 = ascvec.max_scalar %278, %c15_i16, %256 : <ui16>, i16
                    %281 = ascvec.add_scalar %280, %c-15_i16, %256 : <ui16>, i16
                    %282 = ascvec.duplicate_scalar %c255_i16, %279 merge %281 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %283 = ascvec.duplicate_scalar %c254_i16, %256 : i16 -> <ui16>
                    %284 = ascvec.sub %283, %282, %256 : <ui16>
                    %285 = ascvec.cmp_ne_scalar %282, %c0_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %286 = ascvec.shl %284, 7, %285 : <ui16> -> <ui16>
                    %287 = ascvec.max %arg15, %282, %mask : <ui16>
                    %288 = ascvec.gather_reg %286, %260 : <ui16>, <ui16> -> <ui16>
                    %289 = ascvec.reinterpret %288 : <ui16> -> <bf16>
                    %290 = ascvec.mul %result0, %289, %256 : <bf16>
                    %291 = ascvec.mul %result1, %289, %256 : <bf16>
                    ascvec.storealign_intlv %arg10[%271], %290, %291 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %292 = ascvec.reinterpret %290 : <bf16> -> <ui16>
                    %293 = ascvec.and %292, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %294 = ascvec.reinterpret %291 : <bf16> -> <ui16>
                    %295 = ascvec.and %294, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %296 = ascvec.min %293, %295, %256 : <ui16>
                    %297 = ascvec.min %arg16, %296, %256 : <ui16>
                    %298 = arith.muli %arg14, %c16_i32 : i32
                    %299 = arith.index_cast %298 : i32 to index
                    ascvec.storealign %arg11[%299], %282, %mask : memref<?xui16, 1>, index, <ui16>
                    %300 = ascvec.reinterpret %282 : <ui16> -> <ui8>
                    %301 = ascvec.squeeze_store_reg %300, %266 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_11], %301 : memref<2048xui8, 1>, <ui8>
                    scf.yield %287, %297 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                  }
                  %c0_12 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg14 = %c0_12 to %c1 step %c1 {
                    %270 = ascvec.reduce_min %269#1, %256 : <ui16>
                    %271 = ascvec.reduce_max %269#0, %mask : <ui16>
                    %272 = ascvec.cmp_eq_scalar %271, %c255_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %273 = ascvec.duplicate_scalar %c0_i16, %272 merge %270 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.store_unalign_post %arg9[%c0_11] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg13[%c0_11], %273 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %252 = memref.load %alloc_8[%c0] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %253 = cannir.scalar_storage_cast %252 : ui16 -> i16
                %254 = arith.extui %253 : i16 to i64
                %255 = arith.cmpi sge, %254, %c14464_i64 : i64
                scf.if %255 {
                  ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %256 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                  %257 = ascscalar.bitcast_memref %103 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%103, %256, %257, %alloc_4, %250 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_9 = arith.constant 1 : i32
                    %c0_i32_10 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c128_i32 = arith.constant 128 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c31_i16 = arith.constant 31 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %258 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %259 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %260 = ascvec.duplicate_scalar %c255_i16, %258 : i16 -> <ui16>
                    %261 = ascvec.duplicate_scalar %c-128_i8, %259 : i8 -> <ui8>
                    %262 = ascvec.duplicate_scalar %c31_i16, %258 : i16 -> <ui16>
                    scf.for %arg13 = %c0_i32_10 to %arg12 step %c1_i32_9  : i32 {
                      %263 = arith.muli %arg13, %c256_i32 : i32
                      %264 = arith.index_cast %263 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg8[%264] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %265 = arith.muli %arg13, %c128_i32 : i32
                      %266 = arith.index_cast %265 : i32 to index
                      %267 = ascvec.loadalign %arg9[%266] : memref<?xui8, 1>, index -> <ui8>
                      %268 = ascvec.unpack %267 {part = "lower"} : <ui8> -> <ui16>
                      %269 = ascvec.and %268, %262, %258 : <ui16>, <ui16> -> <ui16>
                      %270 = ascvec.shr %268, 3, %258 : <ui16> -> <ui16>
                      %271 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %272 = ascvec.add %271, %269, %258 : <ui16>
                      %273 = ascvec.shr %272, 5, %258 : <ui16> -> <ui16>
                      %274 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %275 = ascvec.add %274, %270, %258 : <ui16>
                      %276 = ascvec.shr %275, 5, %258 : <ui16> -> <ui16>
                      %277 = ascvec.and %273, %260, %258 : <ui16>, <ui16> -> <ui16>
                      %278 = ascvec.shl %276, 8, %258 : <ui16> -> <ui16>
                      %279 = ascvec.or %277, %278, %258 : <ui16>, <ui16> -> <ui16>
                      %280 = ascvec.reinterpret %279 : <ui16> -> <ui8>
                      %281 = ascvec.add_scalar %280, %c64_i8, %259 : <ui8>, i8
                      %282 = ascvec.min_scalar %281, %c123_i8, %259 : <ui8>, i8
                      %283 = arith.muli %arg13, %c512_i32 : i32
                      %284 = arith.index_cast %283 : i32 to index
                      %result0_11, %result1_12 = ascvec.loadalign_deintlv %arg10[%284] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %285 = ascvec.and %result1_12, %261, %259 : <ui8>, <ui8> -> <ui8>
                      %286 = ascvec.or %282, %285, %259 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg11[%264], %286, %259 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
                } else {
                  %256 = arith.cmpi sge, %254, %c13568_i64 : i64
                  scf.if %256 {
                    ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %257 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                    %258 = ascscalar.bitcast_memref %103 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                    ascvec.vf(%103, %257, %258, %alloc_4, %250 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                    ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32):
                      %c512_i32 = arith.constant 512 : i32
                      %c1_i32_9 = arith.constant 1 : i32
                      %c0_i32_10 = arith.constant 0 : i32
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
                      %259 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                      %260 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                      %261 = ascvec.duplicate_scalar %c255_i16, %259 : i16 -> <ui16>
                      %262 = ascvec.duplicate_scalar %c-128_i8, %260 : i8 -> <ui8>
                      %263 = ascvec.arange %c0_i8 : i8 -> <ui8>
                      %264 = ascvec.shr %263, 1, %260 : <ui8> -> <ui8>
                      %265 = ascvec.shr %264, 2, %260 : <ui8> -> <ui8>
                      %266 = ascvec.shl %265, 2, %260 : <ui8> -> <ui8>
                      %267 = ascvec.duplicate_scalar %c1_i8, %260 : i8 -> <ui8>
                      %268 = ascvec.and %263, %267, %260 : <ui8>, <ui8> -> <ui8>
                      %269 = ascvec.add_scalar %264, %c3_i8, %260 : <ui8>, i8
                      %270 = ascvec.add %269, %268, %260 : <ui8>
                      %271 = ascvec.duplicate_scalar %c3_i8, %260 : i8 -> <ui8>
                      %272 = ascvec.and %270, %271, %260 : <ui8>, <ui8> -> <ui8>
                      %273 = ascvec.or %266, %272, %260 : <ui8>, <ui8> -> <ui8>
                      %274 = ascvec.duplicate_scalar %c4095_i16, %259 : i16 -> <ui16>
                      %275 = ascvec.duplicate_scalar %c32767_i16, %259 : i16 -> <ui16>
                      %276 = ascvec.duplicate_scalar %cst, %259 : bf16 -> <bf16>
                      scf.for %arg13 = %c0_i32_10 to %arg12 step %c1_i32_9  : i32 {
                        %277 = arith.muli %arg13, %c256_i32 : i32
                        %278 = arith.index_cast %277 : i32 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg8[%278] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                        %279 = arith.muli %arg13, %c128_i32 : i32
                        %280 = arith.index_cast %279 : i32 to index
                        %281 = ascvec.loadalign %arg9[%280] : memref<?xui8, 1>, index -> <ui8>
                        %282 = ascvec.gather_reg %281, %273 : <ui8>, <ui8> -> <ui8>
                        %283 = ascvec.reinterpret %282 : <ui8> -> <ui16>
                        %284 = ascvec.shr %283, 1, %259 : <ui16> -> <ui16>
                        %285 = ascvec.and %284, %274, %259 : <ui16>, <ui16> -> <ui16>
                        %286 = ascvec.shr %283, 4, %259 : <ui16> -> <ui16>
                        %287 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                        %288 = ascvec.and %287, %275, %259 : <ui16>, <ui16> -> <ui16>
                        %289 = ascvec.cmp_lt_scalar %288, %c14464_i16, %259 : <ui16>, i16 -> !ascvec.mask
                        %290 = ascvec.shr %285, 7, %259 : <ui16> -> <ui16>
                        %291 = ascvec.add %288, %290, %259 : <ui16>
                        %292 = ascvec.shr %291, 5, %259 : <ui16> -> <ui16>
                        %293 = ascvec.reinterpret %288 : <ui16> -> <bf16>
                        %294 = ascvec.mul %293, %276, %289 : <bf16>
                        %295 = ascvec.cast %294, %259 {rounding = 4 : i32} : <bf16> -> <f16>
                        %296 = ascvec.cast %295, %259 {rounding = 4 : i32} : <f16> -> <i16>
                        %297 = ascvec.reinterpret %296 : <i16> -> <ui16>
                        %298 = ascvec.add %297, %285, %259 : <ui16>
                        %299 = ascvec.shr %298, 12, %259 : <ui16> -> <ui16>
                        %300 = ascvec.add_scalar %299, %c448_i16, %259 : <ui16>, i16
                        %301 = ascvec.select %300, %292, %289 : <ui16>
                        %302 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                        %303 = ascvec.and %302, %275, %259 : <ui16>, <ui16> -> <ui16>
                        %304 = ascvec.cmp_lt_scalar %303, %c14464_i16, %259 : <ui16>, i16 -> !ascvec.mask
                        %305 = ascvec.shr %286, 7, %259 : <ui16> -> <ui16>
                        %306 = ascvec.add %303, %305, %259 : <ui16>
                        %307 = ascvec.shr %306, 5, %259 : <ui16> -> <ui16>
                        %308 = ascvec.reinterpret %303 : <ui16> -> <bf16>
                        %309 = ascvec.mul %308, %276, %304 : <bf16>
                        %310 = ascvec.cast %309, %259 {rounding = 4 : i32} : <bf16> -> <f16>
                        %311 = ascvec.cast %310, %259 {rounding = 4 : i32} : <f16> -> <i16>
                        %312 = ascvec.reinterpret %311 : <i16> -> <ui16>
                        %313 = ascvec.add %312, %286, %259 : <ui16>
                        %314 = ascvec.shr %313, 12, %259 : <ui16> -> <ui16>
                        %315 = ascvec.add_scalar %314, %c448_i16, %259 : <ui16>, i16
                        %316 = ascvec.select %315, %307, %304 : <ui16>
                        %317 = ascvec.and %301, %261, %259 : <ui16>, <ui16> -> <ui16>
                        %318 = ascvec.shl %316, 8, %259 : <ui16> -> <ui16>
                        %319 = ascvec.or %317, %318, %259 : <ui16>, <ui16> -> <ui16>
                        %320 = ascvec.reinterpret %319 : <ui16> -> <ui8>
                        %321 = ascvec.add_scalar %320, %c64_i8, %260 : <ui8>, i8
                        %322 = ascvec.min_scalar %321, %c123_i8, %260 : <ui8>, i8
                        %323 = arith.muli %arg13, %c512_i32 : i32
                        %324 = arith.index_cast %323 : i32 to index
                        %result0_11, %result1_12 = ascvec.loadalign_deintlv %arg10[%324] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                        %325 = ascvec.and %result1_12, %262, %260 : <ui8>, <ui8> -> <ui8>
                        %326 = ascvec.or %322, %325, %260 : <ui8>, <ui8> -> <ui8>
                        ascvec.storealign %arg11[%278], %326, %260 : memref<32768xui8, 1>, index, <ui8>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
                  } else {
                    ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %257 = ascscalar.bitcast_memref %103 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                    ascvec.vf(%103, %alloc_7, %257, %251, %alloc_4, %250 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32) {
                    ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<8192xi32, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<?xui16, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32):
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
                      %c4_i32_9 = arith.constant 4 : i32
                      %c2_i32_10 = arith.constant 2 : i32
                      %c1_i32_11 = arith.constant 1 : i32
                      %c7_i32 = arith.constant 7 : i32
                      %c32_i32 = arith.constant 32 : i32
                      %c0_i32_12 = arith.constant 0 : i32
                      %c32767_i16 = arith.constant 32767 : i16
                      %c0_13 = arith.constant 0 : index
                      %c256_i32 = arith.constant 256 : i32
                      %c3_i8 = arith.constant 3 : i8
                      %c-128_i8 = arith.constant -128 : i8
                      %c0_i8 = arith.constant 0 : i8
                      %c8_i32 = arith.constant 8 : i32
                      %258 = ascvec.create_mask : !ascvec.mask
                      %259 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                      %260 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                      %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                      %261 = ascvec.arange %c0_i8 : i8 -> <ui8>
                      %262 = ascvec.duplicate_scalar %c-128_i8, %260 : i8 -> <ui8>
                      %263 = ascvec.shr %261, 5, %260 : <ui8> -> <ui8>
                      %264 = ascvec.duplicate_scalar %c3_i8, %260 : i8 -> <ui8>
                      %265 = ascvec.and %261, %264, %260 : <ui8>, <ui8> -> <ui8>
                      %266 = ascvec.shl %265, 6, %260 : <ui8> -> <ui8>
                      %267 = ascvec.shr %261, 2, %260 : <ui8> -> <ui8>
                      %268 = ascvec.or %266, %267, %260 : <ui8>, <ui8> -> <ui8>
                      scf.for %arg14 = %c0_i32_12 to %arg13 step %c1_i32_11  : i32 {
                        %269 = arith.muli %arg14, %c256_i32 : i32
                        %270 = arith.index_cast %269 : i32 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg8[%270] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                        %271 = ascvec.duplicate_scalar %c32767_i16, %259 : i16 -> <ui16>
                        %272 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                        %273 = ascvec.and %272, %271, %259 : <ui16>, <ui16> -> <ui16>
                        %274 = ascvec.reinterpret %273 : <ui16> -> <bf16>
                        %275 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                        %276 = ascvec.and %275, %271, %259 : <ui16>, <ui16> -> <ui16>
                        %277 = ascvec.reinterpret %276 : <ui16> -> <bf16>
                        %278 = ascvec.cast %274, %259 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %279 = ascvec.cast %274, %259 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %280 = ascvec.cast %277, %259 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %281 = ascvec.cast %277, %259 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %282 = ascvec.arange %c0_i32_12 : i32 -> <ui32>
                        %283 = ascvec.shl %282, 2, %258 : <ui32> -> <ui32>
                        %284 = ascvec.add_scalar %283, %c0_i32_12, %258 : <ui32>, i32
                        %285 = ascvec.shr %284, 3, %258 : <ui32> -> <ui32>
                        %286 = arith.muli %arg14, %c32_i32 : i32
                        %287 = ascvec.add_scalar %285, %286, %258 : <ui32>, i32
                        %288 = ascvec.gather %arg9[%c0_13], %287, %258 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %289 = ascvec.reinterpret %288 : <i32> -> <ui32>
                        %290 = ascvec.duplicate_scalar %c7_i32, %258 : i32 -> <ui32>
                        %291 = ascvec.and %284, %290, %258 : <ui32>, <ui32> -> <ui32>
                        %292 = ascvec.shl %289, 11, %258 : <ui32> -> <ui32>
                        %293 = ascvec.shr %289, 21, %258 : <ui32> -> <ui32>
                        %294 = ascvec.or %292, %293, %258 : <ui32>, <ui32> -> <ui32>
                        %295 = ascvec.shr %294, 3, %258 : <ui32> -> <ui32>
                        %296 = ascvec.shl %294, 29, %258 : <ui32> -> <ui32>
                        %297 = ascvec.or %295, %296, %258 : <ui32>, <ui32> -> <ui32>
                        %298 = ascvec.duplicate_scalar %c1_i32_11, %258 : i32 -> <ui32>
                        %299 = ascvec.and %291, %298, %258 : <ui32>, <ui32> -> <ui32>
                        %300 = ascvec.cmp_ne_scalar %299, %c0_i32_12, %258 : <ui32>, i32 -> !ascvec.mask
                        %301 = ascvec.select %297, %294, %300 : <ui32>
                        %302 = ascvec.shr %301, 8, %258 : <ui32> -> <ui32>
                        %303 = ascvec.shl %301, 24, %258 : <ui32> -> <ui32>
                        %304 = ascvec.or %302, %303, %258 : <ui32>, <ui32> -> <ui32>
                        %305 = ascvec.duplicate_scalar %c2_i32_10, %258 : i32 -> <ui32>
                        %306 = ascvec.and %291, %305, %258 : <ui32>, <ui32> -> <ui32>
                        %307 = ascvec.cmp_ne_scalar %306, %c0_i32_12, %258 : <ui32>, i32 -> !ascvec.mask
                        %308 = ascvec.select %304, %301, %307 : <ui32>
                        %309 = ascvec.shr %308, 16, %258 : <ui32> -> <ui32>
                        %310 = ascvec.shl %308, 16, %258 : <ui32> -> <ui32>
                        %311 = ascvec.or %309, %310, %258 : <ui32>, <ui32> -> <ui32>
                        %312 = ascvec.duplicate_scalar %c4_i32_9, %258 : i32 -> <ui32>
                        %313 = ascvec.and %291, %312, %258 : <ui32>, <ui32> -> <ui32>
                        %314 = ascvec.cmp_ne_scalar %313, %c0_i32_12, %258 : <ui32>, i32 -> !ascvec.mask
                        %315 = ascvec.select %311, %308, %314 : <ui32>
                        %316 = ascvec.duplicate_scalar %c65535_i32, %258 : i32 -> <ui32>
                        %317 = ascvec.and %315, %316, %258 : <ui32>, <ui32> -> <ui32>
                        %318 = ascvec.add_scalar %283, %c1_i32_11, %258 : <ui32>, i32
                        %319 = ascvec.shr %318, 3, %258 : <ui32> -> <ui32>
                        %320 = ascvec.add_scalar %319, %286, %258 : <ui32>, i32
                        %321 = ascvec.gather %arg9[%c0_13], %320, %258 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %322 = ascvec.reinterpret %321 : <i32> -> <ui32>
                        %323 = ascvec.and %318, %290, %258 : <ui32>, <ui32> -> <ui32>
                        %324 = ascvec.shl %322, 11, %258 : <ui32> -> <ui32>
                        %325 = ascvec.shr %322, 21, %258 : <ui32> -> <ui32>
                        %326 = ascvec.or %324, %325, %258 : <ui32>, <ui32> -> <ui32>
                        %327 = ascvec.shr %326, 3, %258 : <ui32> -> <ui32>
                        %328 = ascvec.shl %326, 29, %258 : <ui32> -> <ui32>
                        %329 = ascvec.or %327, %328, %258 : <ui32>, <ui32> -> <ui32>
                        %330 = ascvec.and %323, %298, %258 : <ui32>, <ui32> -> <ui32>
                        %331 = ascvec.cmp_ne_scalar %330, %c0_i32_12, %258 : <ui32>, i32 -> !ascvec.mask
                        %332 = ascvec.select %329, %326, %331 : <ui32>
                        %333 = ascvec.shr %332, 8, %258 : <ui32> -> <ui32>
                        %334 = ascvec.shl %332, 24, %258 : <ui32> -> <ui32>
                        %335 = ascvec.or %333, %334, %258 : <ui32>, <ui32> -> <ui32>
                        %336 = ascvec.and %323, %305, %258 : <ui32>, <ui32> -> <ui32>
                        %337 = ascvec.cmp_ne_scalar %336, %c0_i32_12, %258 : <ui32>, i32 -> !ascvec.mask
                        %338 = ascvec.select %335, %332, %337 : <ui32>
                        %339 = ascvec.shr %338, 16, %258 : <ui32> -> <ui32>
                        %340 = ascvec.shl %338, 16, %258 : <ui32> -> <ui32>
                        %341 = ascvec.or %339, %340, %258 : <ui32>, <ui32> -> <ui32>
                        %342 = ascvec.and %323, %312, %258 : <ui32>, <ui32> -> <ui32>
                        %343 = ascvec.cmp_ne_scalar %342, %c0_i32_12, %258 : <ui32>, i32 -> !ascvec.mask
                        %344 = ascvec.select %341, %338, %343 : <ui32>
                        %345 = ascvec.and %344, %316, %258 : <ui32>, <ui32> -> <ui32>
                        %346 = ascvec.add_scalar %283, %c2_i32_10, %258 : <ui32>, i32
                        %347 = ascvec.shr %346, 3, %258 : <ui32> -> <ui32>
                        %348 = ascvec.add_scalar %347, %286, %258 : <ui32>, i32
                        %349 = ascvec.gather %arg9[%c0_13], %348, %258 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %350 = ascvec.reinterpret %349 : <i32> -> <ui32>
                        %351 = ascvec.and %346, %290, %258 : <ui32>, <ui32> -> <ui32>
                        %352 = ascvec.shl %350, 11, %258 : <ui32> -> <ui32>
                        %353 = ascvec.shr %350, 21, %258 : <ui32> -> <ui32>
                        %354 = ascvec.or %352, %353, %258 : <ui32>, <ui32> -> <ui32>
                        %355 = ascvec.shr %354, 3, %258 : <ui32> -> <ui32>
                        %356 = ascvec.shl %354, 29, %258 : <ui32> -> <ui32>
                        %357 = ascvec.or %355, %356, %258 : <ui32>, <ui32> -> <ui32>
                        %358 = ascvec.and %351, %298, %258 : <ui32>, <ui32> -> <ui32>
                        %359 = ascvec.cmp_ne_scalar %358, %c0_i32_12, %258 : <ui32>, i32 -> !ascvec.mask
                        %360 = ascvec.select %357, %354, %359 : <ui32>
                        %361 = ascvec.shr %360, 8, %258 : <ui32> -> <ui32>
                        %362 = ascvec.shl %360, 24, %258 : <ui32> -> <ui32>
                        %363 = ascvec.or %361, %362, %258 : <ui32>, <ui32> -> <ui32>
                        %364 = ascvec.and %351, %305, %258 : <ui32>, <ui32> -> <ui32>
                        %365 = ascvec.cmp_ne_scalar %364, %c0_i32_12, %258 : <ui32>, i32 -> !ascvec.mask
                        %366 = ascvec.select %363, %360, %365 : <ui32>
                        %367 = ascvec.shr %366, 16, %258 : <ui32> -> <ui32>
                        %368 = ascvec.shl %366, 16, %258 : <ui32> -> <ui32>
                        %369 = ascvec.or %367, %368, %258 : <ui32>, <ui32> -> <ui32>
                        %370 = ascvec.and %351, %312, %258 : <ui32>, <ui32> -> <ui32>
                        %371 = ascvec.cmp_ne_scalar %370, %c0_i32_12, %258 : <ui32>, i32 -> !ascvec.mask
                        %372 = ascvec.select %369, %366, %371 : <ui32>
                        %373 = ascvec.and %372, %316, %258 : <ui32>, <ui32> -> <ui32>
                        %374 = ascvec.add_scalar %283, %c3_i32, %258 : <ui32>, i32
                        %375 = ascvec.shr %374, 3, %258 : <ui32> -> <ui32>
                        %376 = ascvec.add_scalar %375, %286, %258 : <ui32>, i32
                        %377 = ascvec.gather %arg9[%c0_13], %376, %258 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %378 = ascvec.reinterpret %377 : <i32> -> <ui32>
                        %379 = ascvec.and %374, %290, %258 : <ui32>, <ui32> -> <ui32>
                        %380 = ascvec.shl %378, 11, %258 : <ui32> -> <ui32>
                        %381 = ascvec.shr %378, 21, %258 : <ui32> -> <ui32>
                        %382 = ascvec.or %380, %381, %258 : <ui32>, <ui32> -> <ui32>
                        %383 = ascvec.shr %382, 3, %258 : <ui32> -> <ui32>
                        %384 = ascvec.shl %382, 29, %258 : <ui32> -> <ui32>
                        %385 = ascvec.or %383, %384, %258 : <ui32>, <ui32> -> <ui32>
                        %386 = ascvec.and %379, %298, %258 : <ui32>, <ui32> -> <ui32>
                        %387 = ascvec.cmp_ne_scalar %386, %c0_i32_12, %258 : <ui32>, i32 -> !ascvec.mask
                        %388 = ascvec.select %385, %382, %387 : <ui32>
                        %389 = ascvec.shr %388, 8, %258 : <ui32> -> <ui32>
                        %390 = ascvec.shl %388, 24, %258 : <ui32> -> <ui32>
                        %391 = ascvec.or %389, %390, %258 : <ui32>, <ui32> -> <ui32>
                        %392 = ascvec.and %379, %305, %258 : <ui32>, <ui32> -> <ui32>
                        %393 = ascvec.cmp_ne_scalar %392, %c0_i32_12, %258 : <ui32>, i32 -> !ascvec.mask
                        %394 = ascvec.select %391, %388, %393 : <ui32>
                        %395 = ascvec.shr %394, 16, %258 : <ui32> -> <ui32>
                        %396 = ascvec.shl %394, 16, %258 : <ui32> -> <ui32>
                        %397 = ascvec.or %395, %396, %258 : <ui32>, <ui32> -> <ui32>
                        %398 = ascvec.and %379, %312, %258 : <ui32>, <ui32> -> <ui32>
                        %399 = ascvec.cmp_ne_scalar %398, %c0_i32_12, %258 : <ui32>, i32 -> !ascvec.mask
                        %400 = ascvec.select %397, %394, %399 : <ui32>
                        %401 = ascvec.and %400, %316, %258 : <ui32>, <ui32> -> <ui32>
                        %402 = ascvec.reinterpret %278 : <f32> -> <ui32>
                        %403 = ascvec.reinterpret %402 : <ui32> -> <f32>
                        %404 = ascvec.max_scalar %403, %cst, %258 : <f32>, f32
                        %405 = ascvec.add %403, %404, %258 : <f32>
                        %406 = ascvec.reinterpret %405 : <f32> -> <ui32>
                        %407 = ascvec.shr %406, 5, %258 : <ui32> -> <ui32>
                        %408 = ascvec.add %407, %317, %258 : <ui32>
                        %409 = ascvec.reinterpret %280 : <f32> -> <ui32>
                        %410 = ascvec.reinterpret %409 : <ui32> -> <f32>
                        %411 = ascvec.max_scalar %410, %cst, %258 : <f32>, f32
                        %412 = ascvec.add %410, %411, %258 : <f32>
                        %413 = ascvec.reinterpret %412 : <f32> -> <ui32>
                        %414 = ascvec.shr %413, 5, %258 : <ui32> -> <ui32>
                        %415 = ascvec.add %414, %345, %258 : <ui32>
                        %416 = ascvec.reinterpret %279 : <f32> -> <ui32>
                        %417 = ascvec.reinterpret %416 : <ui32> -> <f32>
                        %418 = ascvec.max_scalar %417, %cst, %258 : <f32>, f32
                        %419 = ascvec.add %417, %418, %258 : <f32>
                        %420 = ascvec.reinterpret %419 : <f32> -> <ui32>
                        %421 = ascvec.shr %420, 5, %258 : <ui32> -> <ui32>
                        %422 = ascvec.add %421, %373, %258 : <ui32>
                        %423 = ascvec.reinterpret %281 : <f32> -> <ui32>
                        %424 = ascvec.reinterpret %423 : <ui32> -> <f32>
                        %425 = ascvec.max_scalar %424, %cst, %258 : <f32>, f32
                        %426 = ascvec.add %424, %425, %258 : <f32>
                        %427 = ascvec.reinterpret %426 : <f32> -> <ui32>
                        %428 = ascvec.shr %427, 5, %258 : <ui32> -> <ui32>
                        %429 = ascvec.add %428, %401, %258 : <ui32>
                        %430 = ascvec.reinterpret %408 : <ui32> -> <ui16>
                        %431 = ascvec.reinterpret %415 : <ui32> -> <ui16>
                        %result0_14, %result1_15 = ascvec.deintlv %430, %431 : <ui16>, <ui16> -> <ui16>, <ui16>
                        %432 = ascvec.reinterpret %422 : <ui32> -> <ui16>
                        %433 = ascvec.reinterpret %429 : <ui32> -> <ui16>
                        %result0_16, %result1_17 = ascvec.deintlv %432, %433 : <ui16>, <ui16> -> <ui16>, <ui16>
                        %434 = ascvec.reinterpret %result1_15 : <ui16> -> <ui8>
                        %435 = ascvec.reinterpret %result1_17 : <ui16> -> <ui8>
                        %result0_18, %result1_19 = ascvec.deintlv %434, %435 : <ui8>, <ui8> -> <ui8>, <ui8>
                        %436 = ascvec.gather_reg %result0_18, %268 : <ui8>, <ui8> -> <ui8>
                        %437 = ascvec.add_scalar %436, %c60_i8, %260 : <ui8>, i8
                        %438 = ascvec.cmp_ne_scalar %437, %c-1_i8, %260 : <ui8>, i8 -> !ascvec.mask
                        %439 = ascvec.min_scalar %437, %c123_i8, %438 : <ui8>, i8
                        %440 = arith.muli %arg14, %c512_i32 : i32
                        %441 = arith.index_cast %440 : i32 to index
                        %result0_20, %result1_21 = ascvec.loadalign_deintlv %arg10[%441] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                        %442 = ascvec.and %result1_21, %262, %260 : <ui8>, <ui8> -> <ui8>
                        %443 = ascvec.or %439, %442, %260 : <ui8>, <ui8> -> <ui8>
                        %444 = ascvec.arange %c0_i16 : i16 -> <ui16>
                        %445 = arith.muli %arg14, %c16_i32 : i32
                        %446 = arith.trunci %445 : i32 to i16
                        %447 = ascvec.add_scalar %444, %446, %259 : <ui16>, i16
                        %448 = ascvec.gather %arg11[%c0_13], %447, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %449 = ascvec.pack %448 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                        %450 = ascvec.gather_reg %449, %263 : <ui8>, <ui8> -> <ui8>
                        %451 = ascvec.cmp_eq_scalar %450, %c-1_i8, %260 : <ui8>, i8 -> !ascvec.mask
                        %452 = ascvec.duplicate_scalar %c127_i8, %451 merge %443 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                        ascvec.storealign %arg12[%270], %452, %260 : memref<32768xui8, 1>, index, <ui8>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
                  }
                }
              }
            } else {
              %195 = arith.cmpi sge, %62, %c1024_i64 : i64
              %196 = arith.remsi %62, %c1024_i64 : i64
              %197 = arith.cmpi eq, %196, %c0_i64 : i64
              %198 = arith.select %195, %197, %false : i1
              scf.if %198 {
                %199 = arith.divsi %62, %c256_i64 : i64
                %200 = arith.addi %62, %c1023_i64 : i64
                %201 = arith.divsi %200, %c1024_i64 : i64
                %202 = arith.trunci %arg4 : i64 to i32
                %203 = arith.extui %202 : i32 to i64
                %204 = arith.subi %arg4, %203 : i64
                %205 = arith.remsi %204, %c4294967296_i64 : i64
                %206 = arith.cmpi ne, %205, %c0_i64 : i64
                %207 = arith.cmpi slt, %205, %c0_i64 : i64
                %208 = arith.andi %206, %207 : i1
                %209 = arith.divsi %204, %c4294967296_i64 : i64
                %210 = arith.extui %208 : i1 to i64
                %211 = arith.subi %209, %210 : i64
                %212 = arith.trunci %211 : i64 to i32
                %213 = arith.divui %104, %c16_i64 : i64
                %214 = arith.trunci %213 : i64 to i32
                %215 = arith.extui %214 : i32 to i64
                %216 = arith.subi %213, %215 : i64
                %217 = arith.remsi %216, %c4294967296_i64 : i64
                %218 = arith.cmpi ne, %217, %c0_i64 : i64
                %219 = arith.cmpi slt, %217, %c0_i64 : i64
                %220 = arith.andi %218, %219 : i1
                %221 = arith.divsi %216, %c4294967296_i64 : i64
                %222 = arith.extui %220 : i1 to i64
                %223 = arith.subi %221, %222 : i64
                %224 = arith.trunci %223 : i64 to i32
                %225 = arith.trunci %arg5 : i64 to i32
                %226 = arith.extui %225 : i32 to i64
                %227 = arith.subi %arg5, %226 : i64
                %228 = arith.remsi %227, %c4294967296_i64 : i64
                %229 = arith.cmpi ne, %228, %c0_i64 : i64
                %230 = arith.cmpi slt, %228, %c0_i64 : i64
                %231 = arith.andi %229, %230 : i1
                %232 = arith.divsi %227, %c4294967296_i64 : i64
                %233 = arith.extui %231 : i1 to i64
                %234 = arith.subi %232, %233 : i64
                %235 = arith.trunci %234 : i64 to i32
                %236 = arith.divui %225, %c4_i32 : i32
                %237 = arith.remui %235, %c4_i32 : i32
                %238 = arith.muli %237, %c1073741824_i32 : i32
                %239 = arith.addi %236, %238 : i32
                %240 = arith.divui %235, %c4_i32 : i32
                %241 = arith.muli %201, %c64_i64 : i64
                %242 = arith.divui %213, %c16384_i64 : i64
                %243 = arith.addi %213, %241 : i64
                %244 = arith.subi %243, %c1_i64 : i64
                %245 = arith.divui %244, %c16384_i64 : i64
                %246 = arith.cmpi eq, %242, %245 : i64
                scf.if %246 {
                  %256 = arith.trunci %242 : i64 to i32
                  %257 = arith.extui %256 : i32 to i64
                  %258 = arith.subi %242, %257 : i64
                  %259 = arith.remsi %258, %c4294967296_i64 : i64
                  %260 = arith.cmpi ne, %259, %c0_i64 : i64
                  %261 = arith.cmpi slt, %259, %c0_i64 : i64
                  %262 = arith.andi %260, %261 : i1
                  %263 = arith.divsi %258, %c4294967296_i64 : i64
                  %264 = arith.extui %262 : i1 to i64
                  %265 = arith.subi %263, %264 : i64
                  %266 = arith.trunci %265 : i64 to i32
                  %267 = arith.addi %239, %256 : i32
                  %268 = arith.addi %240, %266 : i32
                  %269 = arith.cmpi ult, %267, %239 : i32
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
                  %285 = arith.xori %271, %212 : i32
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
                  %299 = arith.addi %202, %c-1640531527_i32 : i32
                  %300 = arith.addi %212, %c-1150833019_i32 : i32
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
                  %316 = arith.xori %284, %202 : i32
                  %317 = arith.addi %202, %c1013904242_i32 : i32
                  %318 = arith.xori %288, %317 : i32
                  %319 = arith.addi %212, %c1993301258_i32 : i32
                  %320 = arith.xori %315, %319 : i32
                  ascsimt.vf(%214, %316, %300, %318, %320, %202, %212, %305, %alloc_7, %241 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: memref<8192xi32, 1>, %arg17: i64):
                    %c256_i64_9 = arith.constant 256 : i64
                    %c3_i64_10 = arith.constant 3 : i64
                    %c2_i64_11 = arith.constant 2 : i64
                    %c1_i64_12 = arith.constant 1 : i64
                    %c4_i64_13 = arith.constant 4 : i64
                    %c1_i32_14 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c3_i32 = arith.constant 3 : i32
                    %c-1150833019_i32_15 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_16 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %321 = ascsimt.thread_idx {ascsimt.scope = 14 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %322 = arith.extsi %321 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg18 = %322 to %arg17 step %c256_i64_9  : i64 {
                      %323 = arith.trunci %arg18 : i64 to i32
                      %324 = arith.addi %arg8, %323 : i32
                      %325 = arith.remui %324, %c16384_i32 : i32
                      %326 = arith.xori %325, %arg9 : i32
                      %327 = arith.muli %326, %c-766435501_i32 : i32
                      %328 = "ascsimt.bitcast"(%326) : (i32) -> ui32
                      %329 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %330 = ascsimt.math_binary %328, %329, "mulhi" : ui32, ui32 -> ui32
                      %331 = "ascsimt.bitcast"(%330) : (ui32) -> i32
                      %332 = arith.xori %331, %arg10 : i32
                      %333 = arith.muli %332, %c-845247145_i32 : i32
                      %334 = "ascsimt.bitcast"(%332) : (i32) -> ui32
                      %335 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %336 = ascsimt.math_binary %334, %335, "mulhi" : ui32, ui32 -> ui32
                      %337 = "ascsimt.bitcast"(%336) : (ui32) -> i32
                      %338 = arith.xori %337, %arg11 : i32
                      %339 = arith.xori %327, %arg12 : i32
                      %340:4 = scf.for %arg19 = %c3_i32 to %c10_i32 step %c1_i32_14 iter_args(%arg20 = %338, %arg21 = %333, %arg22 = %339, %arg23 = %arg15) -> (i32, i32, i32, i32)  : i32 {
                        %349 = arith.muli %arg20, %c-766435501_i32 : i32
                        %350 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                        %351 = ascsimt.math_binary %350, %329, "mulhi" : ui32, ui32 -> ui32
                        %352 = "ascsimt.bitcast"(%351) : (ui32) -> i32
                        %353 = arith.muli %arg22, %c-845247145_i32 : i32
                        %354 = "ascsimt.bitcast"(%arg22) : (i32) -> ui32
                        %355 = ascsimt.math_binary %354, %335, "mulhi" : ui32, ui32 -> ui32
                        %356 = "ascsimt.bitcast"(%355) : (ui32) -> i32
                        %357 = arith.muli %arg19, %c-1640531527_i32_16 : i32
                        %358 = arith.addi %arg13, %357 : i32
                        %359 = arith.muli %arg19, %c-1150833019_i32_15 : i32
                        %360 = arith.addi %arg14, %359 : i32
                        %361 = arith.xori %356, %arg21 : i32
                        %362 = arith.xori %361, %358 : i32
                        %363 = arith.xori %352, %arg23 : i32
                        %364 = arith.xori %363, %360 : i32
                        scf.yield %362, %353, %364, %349 : i32, i32, i32, i32
                      }
                      %341 = arith.muli %arg18, %c4_i64_13 : i64
                      %342 = arith.index_cast %341 : i64 to index
                      ascsimt.store %arg16[%342], %340#0 : memref<8192xi32, 1>, i32
                      %343 = arith.addi %341, %c1_i64_12 : i64
                      %344 = arith.index_cast %343 : i64 to index
                      ascsimt.store %arg16[%344], %340#1 : memref<8192xi32, 1>, i32
                      %345 = arith.addi %341, %c2_i64_11 : i64
                      %346 = arith.index_cast %345 : i64 to index
                      ascsimt.store %arg16[%346], %340#2 : memref<8192xi32, 1>, i32
                      %347 = arith.addi %341, %c3_i64_10 : i64
                      %348 = arith.index_cast %347 : i64 to index
                      ascsimt.store %arg16[%348], %340#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%214, %224, %239, %240, %202, %212, %alloc_7, %241 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: memref<8192xi32, 1>, %arg15: i64):
                    %c256_i64_9 = arith.constant 256 : i64
                    %c3_i64_10 = arith.constant 3 : i64
                    %c2_i64_11 = arith.constant 2 : i64
                    %c1_i64_12 = arith.constant 1 : i64
                    %c4_i64_13 = arith.constant 4 : i64
                    %c1_i32_14 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c0_i32_15 = arith.constant 0 : i32
                    %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c262144_i32 = arith.constant 262144 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %256 = ascsimt.thread_idx {ascsimt.scope = 15 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %257 = arith.extsi %256 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg16 = %257 to %arg15 step %c256_i64_9  : i64 {
                      %258 = arith.trunci %arg16 : i64 to i32
                      %259 = arith.addi %arg8, %258 : i32
                      %260 = arith.cmpi ult, %259, %arg8 : i32
                      %261 = arith.extui %260 : i1 to i32
                      %262 = arith.addi %arg9, %261 : i32
                      %263 = arith.divui %259, %c16384_i32 : i32
                      %264 = arith.remui %262, %c16384_i32 : i32
                      %265 = arith.muli %264, %c262144_i32 : i32
                      %266 = arith.addi %263, %265 : i32
                      %267 = arith.divui %262, %c16384_i32 : i32
                      %268 = arith.addi %arg10, %266 : i32
                      %269 = arith.addi %arg11, %267 : i32
                      %270 = arith.cmpi ult, %268, %arg10 : i32
                      %271 = arith.extui %270 : i1 to i32
                      %272 = arith.addi %269, %271 : i32
                      %273 = arith.remui %259, %c16384_i32 : i32
                      %274:4 = scf.for %arg17 = %c0_i32_15 to %c10_i32 step %c1_i32_14 iter_args(%arg18 = %c0_i32_15, %arg19 = %273, %arg20 = %268, %arg21 = %272) -> (i32, i32, i32, i32)  : i32 {
                        %283 = arith.muli %arg18, %c-766435501_i32 : i32
                        %284 = "ascsimt.bitcast"(%arg18) : (i32) -> ui32
                        %285 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %286 = ascsimt.math_binary %284, %285, "mulhi" : ui32, ui32 -> ui32
                        %287 = "ascsimt.bitcast"(%286) : (ui32) -> i32
                        %288 = arith.muli %arg20, %c-845247145_i32 : i32
                        %289 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                        %290 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %291 = ascsimt.math_binary %289, %290, "mulhi" : ui32, ui32 -> ui32
                        %292 = "ascsimt.bitcast"(%291) : (ui32) -> i32
                        %293 = arith.muli %arg17, %c-1640531527_i32_17 : i32
                        %294 = arith.addi %arg12, %293 : i32
                        %295 = arith.muli %arg17, %c-1150833019_i32_16 : i32
                        %296 = arith.addi %arg13, %295 : i32
                        %297 = arith.xori %292, %arg19 : i32
                        %298 = arith.xori %297, %294 : i32
                        %299 = arith.xori %287, %arg21 : i32
                        %300 = arith.xori %299, %296 : i32
                        scf.yield %298, %288, %300, %283 : i32, i32, i32, i32
                      }
                      %275 = arith.muli %arg16, %c4_i64_13 : i64
                      %276 = arith.index_cast %275 : i64 to index
                      ascsimt.store %arg14[%276], %274#0 : memref<8192xi32, 1>, i32
                      %277 = arith.addi %275, %c1_i64_12 : i64
                      %278 = arith.index_cast %277 : i64 to index
                      ascsimt.store %arg14[%278], %274#1 : memref<8192xi32, 1>, i32
                      %279 = arith.addi %275, %c2_i64_11 : i64
                      %280 = arith.index_cast %279 : i64 to index
                      ascsimt.store %arg14[%280], %274#2 : memref<8192xi32, 1>, i32
                      %281 = arith.addi %275, %c3_i64_10 : i64
                      %282 = arith.index_cast %281 : i64 to index
                      ascsimt.store %arg14[%282], %274#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %247 = arith.trunci %44 : i64 to i32
                %248 = arith.divsi %247, %c2_i32 : i32
                %249 = arith.trunci %248 : i32 to i8
                %250 = arith.trunci %199 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %251 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%249, %alloc_5, %103, %251, %alloc_8, %250 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, memref<2048xui16, 1>, i32) {
                ^bb0(%arg8: i8, %arg9: memref<2048xui8, 1>, %arg10: memref<32768xbf16, 1>, %arg11: memref<?xui16, 1>, %arg12: memref<2048xui16, 1>, %arg13: i32):
                  %c16_9 = arith.constant 16 : index
                  %c4_i32_10 = arith.constant 4 : i32
                  %c0_i32_11 = arith.constant 0 : i32
                  %c3_i32 = arith.constant 3 : i32
                  %c2_i32_12 = arith.constant 2 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c1_i32_13 = arith.constant 1 : i32
                  %c0_14 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %256 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %257 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %258 = ascvec.duplicate_scalar %c32767_i16, %256 : i16 -> <ui16>
                  %259 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %260 = ascvec.shr %259, 4, %256 : <ui16> -> <ui16>
                  %261 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %262 = ascvec.duplicate_scalar %c1_i8, %257 : i8 -> <ui8>
                  %263 = ascvec.and %261, %262, %257 : <ui8>, <ui8> -> <ui8>
                  %264 = ascvec.cmp_lt_scalar %261, %c16_i8, %257 : <ui8>, i8 -> !ascvec.mask
                  %265 = ascvec.cmp_lt_scalar %263, %arg8, %257 : <ui8>, i8 -> !ascvec.mask
                  %266 = ascvec.mask_and %264, %265, %257
                  ascvec.store_unalign_begin %arg9 : memref<2048xui8, 1>
                  %mask_15, %remaining_out_16 = ascvec.update_mask %c1_i32_13 elem_bits 16 : !ascvec.mask
                  %267 = ascvec.duplicate_scalar %c-1_i16, %mask_15 : i16 -> <ui16>
                  %268 = scf.for %arg14 = %c0_i32_11 to %arg13 step %c4_i32_10 iter_args(%arg15 = %267) -> (!ascvec.vreg<ui16>)  : i32 {
                    %269 = ascvec.duplicate_scalar %c-1_i16, %256 : i16 -> <ui16>
                    %270 = ascvec.duplicate_scalar %c0_i16, %256 : i16 -> <ui16>
                    %271 = arith.muli %arg14, %c256_i32 : i32
                    %272 = arith.index_cast %271 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg10[%272] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %273 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %274 = ascvec.and %273, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %275 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %276 = ascvec.and %275, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %277 = ascvec.max %274, %276, %256 : <ui16>
                    %278 = ascvec.reduce_max_datablock %277, %256 : <ui16>
                    %279 = ascvec.shr %278, 7, %256 : <ui16> -> <ui16>
                    %280 = ascvec.cmp_ge_scalar %279, %c255_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %281 = ascvec.max_scalar %279, %c15_i16, %256 : <ui16>, i16
                    %282 = ascvec.add_scalar %281, %c-15_i16, %256 : <ui16>, i16
                    %283 = ascvec.duplicate_scalar %c255_i16, %280 merge %282 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %284 = ascvec.duplicate_scalar %c254_i16, %256 : i16 -> <ui16>
                    %285 = ascvec.sub %284, %283, %256 : <ui16>
                    %286 = ascvec.cmp_ne_scalar %283, %c0_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %287 = ascvec.shl %285, 7, %286 : <ui16> -> <ui16>
                    %288 = ascvec.max %270, %283, %mask : <ui16>
                    %289 = ascvec.gather_reg %287, %260 : <ui16>, <ui16> -> <ui16>
                    %290 = ascvec.reinterpret %289 : <ui16> -> <bf16>
                    %291 = ascvec.mul %result0, %290, %256 : <bf16>
                    %292 = ascvec.mul %result1, %290, %256 : <bf16>
                    ascvec.storealign_intlv %arg10[%272], %291, %292 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %293 = ascvec.reinterpret %291 : <bf16> -> <ui16>
                    %294 = ascvec.and %293, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %295 = ascvec.reinterpret %292 : <bf16> -> <ui16>
                    %296 = ascvec.and %295, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %297 = ascvec.min %294, %296, %256 : <ui16>
                    %298 = ascvec.min %269, %297, %256 : <ui16>
                    %299 = arith.muli %arg14, %c16_i32 : i32
                    %300 = arith.index_cast %299 : i32 to index
                    ascvec.storealign %arg11[%300], %283, %mask : memref<?xui16, 1>, index, <ui16>
                    %301 = ascvec.reinterpret %283 : <ui16> -> <ui8>
                    %302 = ascvec.squeeze_store_reg %301, %266 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_14], %302 : memref<2048xui8, 1>, <ui8>
                    %303 = arith.addi %arg14, %c1_i32_13 : i32
                    %304 = arith.muli %303, %c256_i32 : i32
                    %305 = arith.index_cast %304 : i32 to index
                    %result0_18, %result1_19 = ascvec.loadalign_deintlv %arg10[%305] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %306 = ascvec.reinterpret %result0_18 : <bf16> -> <ui16>
                    %307 = ascvec.and %306, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %308 = ascvec.reinterpret %result1_19 : <bf16> -> <ui16>
                    %309 = ascvec.and %308, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %310 = ascvec.max %307, %309, %256 : <ui16>
                    %311 = ascvec.reduce_max_datablock %310, %256 : <ui16>
                    %312 = ascvec.shr %311, 7, %256 : <ui16> -> <ui16>
                    %313 = ascvec.cmp_ge_scalar %312, %c255_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %314 = ascvec.max_scalar %312, %c15_i16, %256 : <ui16>, i16
                    %315 = ascvec.add_scalar %314, %c-15_i16, %256 : <ui16>, i16
                    %316 = ascvec.duplicate_scalar %c255_i16, %313 merge %315 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %317 = ascvec.sub %284, %316, %256 : <ui16>
                    %318 = ascvec.cmp_ne_scalar %316, %c0_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %319 = ascvec.shl %317, 7, %318 : <ui16> -> <ui16>
                    %320 = ascvec.max %288, %316, %mask : <ui16>
                    %321 = ascvec.gather_reg %319, %260 : <ui16>, <ui16> -> <ui16>
                    %322 = ascvec.reinterpret %321 : <ui16> -> <bf16>
                    %323 = ascvec.mul %result0_18, %322, %256 : <bf16>
                    %324 = ascvec.mul %result1_19, %322, %256 : <bf16>
                    ascvec.storealign_intlv %arg10[%305], %323, %324 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %325 = ascvec.reinterpret %323 : <bf16> -> <ui16>
                    %326 = ascvec.and %325, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %327 = ascvec.reinterpret %324 : <bf16> -> <ui16>
                    %328 = ascvec.and %327, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %329 = ascvec.min %326, %328, %256 : <ui16>
                    %330 = ascvec.min %298, %329, %256 : <ui16>
                    %331 = arith.muli %303, %c16_i32 : i32
                    %332 = arith.index_cast %331 : i32 to index
                    ascvec.storealign %arg11[%332], %316, %mask : memref<?xui16, 1>, index, <ui16>
                    %333 = ascvec.reinterpret %316 : <ui16> -> <ui8>
                    %334 = ascvec.squeeze_store_reg %333, %266 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_14], %334 : memref<2048xui8, 1>, <ui8>
                    %335 = arith.addi %arg14, %c2_i32_12 : i32
                    %336 = arith.muli %335, %c256_i32 : i32
                    %337 = arith.index_cast %336 : i32 to index
                    %result0_20, %result1_21 = ascvec.loadalign_deintlv %arg10[%337] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %338 = ascvec.reinterpret %result0_20 : <bf16> -> <ui16>
                    %339 = ascvec.and %338, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %340 = ascvec.reinterpret %result1_21 : <bf16> -> <ui16>
                    %341 = ascvec.and %340, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %342 = ascvec.max %339, %341, %256 : <ui16>
                    %343 = ascvec.reduce_max_datablock %342, %256 : <ui16>
                    %344 = ascvec.shr %343, 7, %256 : <ui16> -> <ui16>
                    %345 = ascvec.cmp_ge_scalar %344, %c255_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %346 = ascvec.max_scalar %344, %c15_i16, %256 : <ui16>, i16
                    %347 = ascvec.add_scalar %346, %c-15_i16, %256 : <ui16>, i16
                    %348 = ascvec.duplicate_scalar %c255_i16, %345 merge %347 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %349 = ascvec.sub %284, %348, %256 : <ui16>
                    %350 = ascvec.cmp_ne_scalar %348, %c0_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %351 = ascvec.shl %349, 7, %350 : <ui16> -> <ui16>
                    %352 = ascvec.max %320, %348, %mask : <ui16>
                    %353 = ascvec.gather_reg %351, %260 : <ui16>, <ui16> -> <ui16>
                    %354 = ascvec.reinterpret %353 : <ui16> -> <bf16>
                    %355 = ascvec.mul %result0_20, %354, %256 : <bf16>
                    %356 = ascvec.mul %result1_21, %354, %256 : <bf16>
                    ascvec.storealign_intlv %arg10[%337], %355, %356 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %357 = ascvec.reinterpret %355 : <bf16> -> <ui16>
                    %358 = ascvec.and %357, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %359 = ascvec.reinterpret %356 : <bf16> -> <ui16>
                    %360 = ascvec.and %359, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %361 = ascvec.min %358, %360, %256 : <ui16>
                    %362 = ascvec.min %330, %361, %256 : <ui16>
                    %363 = arith.muli %335, %c16_i32 : i32
                    %364 = arith.index_cast %363 : i32 to index
                    ascvec.storealign %arg11[%364], %348, %mask : memref<?xui16, 1>, index, <ui16>
                    %365 = ascvec.reinterpret %348 : <ui16> -> <ui8>
                    %366 = ascvec.squeeze_store_reg %365, %266 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_14], %366 : memref<2048xui8, 1>, <ui8>
                    %367 = arith.addi %arg14, %c3_i32 : i32
                    %368 = arith.muli %367, %c256_i32 : i32
                    %369 = arith.index_cast %368 : i32 to index
                    %result0_22, %result1_23 = ascvec.loadalign_deintlv %arg10[%369] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %370 = ascvec.reinterpret %result0_22 : <bf16> -> <ui16>
                    %371 = ascvec.and %370, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %372 = ascvec.reinterpret %result1_23 : <bf16> -> <ui16>
                    %373 = ascvec.and %372, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %374 = ascvec.max %371, %373, %256 : <ui16>
                    %375 = ascvec.reduce_max_datablock %374, %256 : <ui16>
                    %376 = ascvec.shr %375, 7, %256 : <ui16> -> <ui16>
                    %377 = ascvec.cmp_ge_scalar %376, %c255_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %378 = ascvec.max_scalar %376, %c15_i16, %256 : <ui16>, i16
                    %379 = ascvec.add_scalar %378, %c-15_i16, %256 : <ui16>, i16
                    %380 = ascvec.duplicate_scalar %c255_i16, %377 merge %379 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %381 = ascvec.sub %284, %380, %256 : <ui16>
                    %382 = ascvec.cmp_ne_scalar %380, %c0_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %383 = ascvec.shl %381, 7, %382 : <ui16> -> <ui16>
                    %384 = ascvec.max %352, %380, %mask : <ui16>
                    %385 = ascvec.gather_reg %383, %260 : <ui16>, <ui16> -> <ui16>
                    %386 = ascvec.reinterpret %385 : <ui16> -> <bf16>
                    %387 = ascvec.mul %result0_22, %386, %256 : <bf16>
                    %388 = ascvec.mul %result1_23, %386, %256 : <bf16>
                    ascvec.storealign_intlv %arg10[%369], %387, %388 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %389 = ascvec.reinterpret %387 : <bf16> -> <ui16>
                    %390 = ascvec.and %389, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %391 = ascvec.reinterpret %388 : <bf16> -> <ui16>
                    %392 = ascvec.and %391, %258, %256 : <ui16>, <ui16> -> <ui16>
                    %393 = ascvec.min %390, %392, %256 : <ui16>
                    %394 = ascvec.min %362, %393, %256 : <ui16>
                    %395 = arith.muli %367, %c16_i32 : i32
                    %396 = arith.index_cast %395 : i32 to index
                    ascvec.storealign %arg11[%396], %380, %mask : memref<?xui16, 1>, index, <ui16>
                    %397 = ascvec.reinterpret %380 : <ui16> -> <ui8>
                    %398 = ascvec.squeeze_store_reg %397, %266 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_14], %398 : memref<2048xui8, 1>, <ui8>
                    %399 = ascvec.reduce_min %394, %256 : <ui16>
                    %400 = ascvec.reduce_max %384, %mask : <ui16>
                    %401 = ascvec.cmp_eq_scalar %400, %c255_i16, %256 : <ui16>, i16 -> !ascvec.mask
                    %402 = ascvec.duplicate_scalar %c0_i16, %401 merge %399 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.storealign_1st %arg12[%300], %402 : memref<2048xui16, 1>, index, <ui16>
                    %403 = ascvec.min %arg15, %402, %mask_15 : <ui16>
                    scf.yield %403 : !ascvec.vreg<ui16>
                  }
                  %c0_17 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg14 = %c0_17 to %c1 step %c1 {
                    ascvec.store_unalign_post %arg9[%c0_14] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg12[%c16_9], %268 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %252 = memref.load %alloc_8[%c16] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %253 = cannir.scalar_storage_cast %252 : ui16 -> i16
                %254 = arith.extui %253 : i16 to i64
                %255 = arith.cmpi sge, %254, %c14464_i64 : i64
                scf.if %255 {
                  ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %256 = ascscalar.bitcast_memref %103 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%103, %alloc_7, %256, %alloc_4, %250 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<8192xi32, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_9 = arith.constant 1 : i32
                    %c0_i32_10 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c64_i32 = arith.constant 64 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c1_i16 = arith.constant 1 : i16
                    %c0_i16 = arith.constant 0 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %257 = ascvec.create_mask : !ascvec.mask
                    %258 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %259 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %260 = ascvec.duplicate_scalar %c255_i16, %258 : i16 -> <ui16>
                    %261 = ascvec.duplicate_scalar %c-128_i8, %259 : i8 -> <ui8>
                    %262 = ascvec.arange %c0_i16 : i16 -> <ui16>
                    %263 = ascvec.duplicate_scalar %c1_i16, %258 : i16 -> <ui16>
                    %264 = ascvec.xor %262, %263, %258 : <ui16>, <ui16> -> <ui16>
                    scf.for %arg13 = %c0_i32_10 to %arg12 step %c1_i32_9  : i32 {
                      %265 = arith.muli %arg13, %c256_i32 : i32
                      %266 = arith.index_cast %265 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg8[%266] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %267 = arith.muli %arg13, %c64_i32 : i32
                      %268 = arith.index_cast %267 : i32 to index
                      %269 = ascvec.loadalign %arg9[%268] : memref<8192xi32, 1>, index -> <i32>
                      %270 = ascvec.reinterpret %269 : <i32> -> <ui16>
                      %271 = ascvec.reinterpret %270 : <ui16> -> <ui32>
                      %272 = ascvec.shl %271, 8, %257 : <ui32> -> <ui32>
                      %273 = ascvec.shr %271, 24, %257 : <ui32> -> <ui32>
                      %274 = ascvec.or %272, %273, %257 : <ui32>, <ui32> -> <ui32>
                      %275 = ascvec.reinterpret %274 : <ui32> -> <ui16>
                      %276 = ascvec.gather_reg %270, %264 : <ui16>, <ui16> -> <ui16>
                      %277 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %278 = ascvec.shr %276, 11, %258 : <ui16> -> <ui16>
                      %279 = ascvec.add %277, %278, %258 : <ui16>
                      %280 = ascvec.shr %279, 5, %258 : <ui16> -> <ui16>
                      %281 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %282 = ascvec.shr %275, 11, %258 : <ui16> -> <ui16>
                      %283 = ascvec.add %281, %282, %258 : <ui16>
                      %284 = ascvec.shr %283, 5, %258 : <ui16> -> <ui16>
                      %285 = ascvec.and %280, %260, %258 : <ui16>, <ui16> -> <ui16>
                      %286 = ascvec.shl %284, 8, %258 : <ui16> -> <ui16>
                      %287 = ascvec.or %285, %286, %258 : <ui16>, <ui16> -> <ui16>
                      %288 = ascvec.reinterpret %287 : <ui16> -> <ui8>
                      %289 = ascvec.add_scalar %288, %c64_i8, %259 : <ui8>, i8
                      %290 = ascvec.min_scalar %289, %c123_i8, %259 : <ui8>, i8
                      %291 = arith.muli %arg13, %c512_i32 : i32
                      %292 = arith.index_cast %291 : i32 to index
                      %result0_11, %result1_12 = ascvec.loadalign_deintlv %arg10[%292] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %293 = ascvec.and %result1_12, %261, %259 : <ui8>, <ui8> -> <ui8>
                      %294 = ascvec.or %290, %293, %259 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg11[%266], %294, %259 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
                } else {
                  scf.for %arg8 = %c0_i64 to %199 step %c16_i64  : i64 {
                    %256 = arith.addi %arg8, %c16_i64 : i64
                    %257 = arith.cmpi sgt, %256, %199 : i64
                    %258 = arith.select %257, %199, %256 : i64
                    %259 = arith.muli %arg8, %c16_i64 : i64
                    ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %260 = arith.index_cast %259 : i64 to index
                    %261 = memref.load %alloc_8[%260] : memref<2048xui16, 1>
                    ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %262 = cannir.scalar_storage_cast %261 : ui16 -> i16
                    %263 = arith.extui %262 : i16 to i64
                    %264 = arith.addi %arg8, %c4_i64 : i64
                    %265 = scf.for %arg9 = %264 to %258 step %c4_i64 iter_args(%arg10 = %263) -> (i64)  : i64 {
                      %267 = arith.muli %arg9, %c16_i64 : i64
                      ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %268 = arith.index_cast %267 : i64 to index
                      %269 = memref.load %alloc_8[%268] : memref<2048xui16, 1>
                      ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %270 = cannir.scalar_storage_cast %269 : ui16 -> i16
                      %271 = arith.extui %270 : i16 to i64
                      %272 = arith.cmpi slt, %271, %arg10 : i64
                      %273 = arith.select %272, %271, %arg10 : i64
                      scf.yield %273 : i64
                    }
                    %266 = arith.cmpi sge, %265, %c14464_i64 : i64
                    scf.if %266 {
                      %267 = arith.trunci %arg8 : i64 to i32
                      %268 = arith.trunci %258 : i64 to i32
                      ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %269 = ascscalar.bitcast_memref %103 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                      ascvec.vf(%103, %alloc_7, %269, %alloc_4, %267, %268 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                      ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<8192xi32, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32, %arg14: i32):
                        %c512_i32 = arith.constant 512 : i32
                        %c1_i32_9 = arith.constant 1 : i32
                        %c123_i8 = arith.constant 123 : i8
                        %c64_i8 = arith.constant 64 : i8
                        %c64_i32 = arith.constant 64 : i32
                        %c256_i32 = arith.constant 256 : i32
                        %c1_i16 = arith.constant 1 : i16
                        %c0_i16 = arith.constant 0 : i16
                        %c-128_i8 = arith.constant -128 : i8
                        %c255_i16 = arith.constant 255 : i16
                        %270 = ascvec.create_mask : !ascvec.mask
                        %271 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                        %272 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                        %273 = ascvec.duplicate_scalar %c255_i16, %271 : i16 -> <ui16>
                        %274 = ascvec.duplicate_scalar %c-128_i8, %272 : i8 -> <ui8>
                        %275 = ascvec.arange %c0_i16 : i16 -> <ui16>
                        %276 = ascvec.duplicate_scalar %c1_i16, %271 : i16 -> <ui16>
                        %277 = ascvec.xor %275, %276, %271 : <ui16>, <ui16> -> <ui16>
                        scf.for %arg15 = %arg13 to %arg14 step %c1_i32_9  : i32 {
                          %278 = arith.muli %arg15, %c256_i32 : i32
                          %279 = arith.index_cast %278 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg9[%279] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                          %280 = arith.muli %arg15, %c64_i32 : i32
                          %281 = arith.index_cast %280 : i32 to index
                          %282 = ascvec.loadalign %arg10[%281] : memref<8192xi32, 1>, index -> <i32>
                          %283 = ascvec.reinterpret %282 : <i32> -> <ui16>
                          %284 = ascvec.reinterpret %283 : <ui16> -> <ui32>
                          %285 = ascvec.shl %284, 8, %270 : <ui32> -> <ui32>
                          %286 = ascvec.shr %284, 24, %270 : <ui32> -> <ui32>
                          %287 = ascvec.or %285, %286, %270 : <ui32>, <ui32> -> <ui32>
                          %288 = ascvec.reinterpret %287 : <ui32> -> <ui16>
                          %289 = ascvec.gather_reg %283, %277 : <ui16>, <ui16> -> <ui16>
                          %290 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                          %291 = ascvec.shr %289, 11, %271 : <ui16> -> <ui16>
                          %292 = ascvec.add %290, %291, %271 : <ui16>
                          %293 = ascvec.shr %292, 5, %271 : <ui16> -> <ui16>
                          %294 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                          %295 = ascvec.shr %288, 11, %271 : <ui16> -> <ui16>
                          %296 = ascvec.add %294, %295, %271 : <ui16>
                          %297 = ascvec.shr %296, 5, %271 : <ui16> -> <ui16>
                          %298 = ascvec.and %293, %273, %271 : <ui16>, <ui16> -> <ui16>
                          %299 = ascvec.shl %297, 8, %271 : <ui16> -> <ui16>
                          %300 = ascvec.or %298, %299, %271 : <ui16>, <ui16> -> <ui16>
                          %301 = ascvec.reinterpret %300 : <ui16> -> <ui8>
                          %302 = ascvec.add_scalar %301, %c64_i8, %272 : <ui8>, i8
                          %303 = ascvec.min_scalar %302, %c123_i8, %272 : <ui8>, i8
                          %304 = arith.muli %arg15, %c512_i32 : i32
                          %305 = arith.index_cast %304 : i32 to index
                          %result0_10, %result1_11 = ascvec.loadalign_deintlv %arg11[%305] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                          %306 = ascvec.and %result1_11, %274, %272 : <ui8>, <ui8> -> <ui8>
                          %307 = ascvec.or %303, %306, %272 : <ui8>, <ui8> -> <ui8>
                          ascvec.storealign %arg12[%279], %307, %272 : memref<32768xui8, 1>, index, <ui8>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
                    } else {
                      %267 = arith.cmpi sge, %265, %c13568_i64 : i64
                      scf.if %267 {
                        %268 = arith.trunci %arg8 : i64 to i32
                        %269 = arith.trunci %258 : i64 to i32
                        ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %270 = ascscalar.bitcast_memref %103 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%103, %alloc_7, %270, %alloc_4, %268, %269 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<8192xi32, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32, %arg14: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_9 = arith.constant 1 : i32
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
                          %271 = ascvec.create_mask : !ascvec.mask
                          %272 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %273 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %274 = ascvec.duplicate_scalar %c255_i16, %272 : i16 -> <ui16>
                          %275 = ascvec.duplicate_scalar %c-128_i8, %273 : i8 -> <ui8>
                          %276 = ascvec.arange %c0_i16 : i16 -> <ui16>
                          %277 = ascvec.duplicate_scalar %c1_i16, %272 : i16 -> <ui16>
                          %278 = ascvec.xor %276, %277, %272 : <ui16>, <ui16> -> <ui16>
                          %279 = ascvec.duplicate_scalar %c32767_i16, %272 : i16 -> <ui16>
                          %280 = ascvec.duplicate_scalar %cst, %272 : bf16 -> <bf16>
                          scf.for %arg15 = %arg13 to %arg14 step %c1_i32_9  : i32 {
                            %281 = arith.muli %arg15, %c256_i32 : i32
                            %282 = arith.index_cast %281 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg9[%282] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %283 = arith.muli %arg15, %c64_i32 : i32
                            %284 = arith.index_cast %283 : i32 to index
                            %285 = ascvec.loadalign %arg10[%284] : memref<8192xi32, 1>, index -> <i32>
                            %286 = ascvec.reinterpret %285 : <i32> -> <ui16>
                            %287 = ascvec.reinterpret %286 : <ui16> -> <ui32>
                            %288 = ascvec.shl %287, 8, %271 : <ui32> -> <ui32>
                            %289 = ascvec.shr %287, 24, %271 : <ui32> -> <ui32>
                            %290 = ascvec.or %288, %289, %271 : <ui32>, <ui32> -> <ui32>
                            %291 = ascvec.reinterpret %290 : <ui32> -> <ui16>
                            %292 = ascvec.gather_reg %286, %278 : <ui16>, <ui16> -> <ui16>
                            %293 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %294 = ascvec.and %293, %279, %272 : <ui16>, <ui16> -> <ui16>
                            %295 = ascvec.cmp_lt_scalar %294, %c14464_i16, %272 : <ui16>, i16 -> !ascvec.mask
                            %296 = ascvec.shr %292, 11, %272 : <ui16> -> <ui16>
                            %297 = ascvec.add %294, %296, %272 : <ui16>
                            %298 = ascvec.shr %297, 5, %272 : <ui16> -> <ui16>
                            %299 = ascvec.reinterpret %294 : <ui16> -> <bf16>
                            %300 = ascvec.mul %299, %280, %295 : <bf16>
                            %301 = ascvec.cast %300, %272 {rounding = 4 : i32} : <bf16> -> <f16>
                            %302 = ascvec.cast %301, %272 {rounding = 4 : i32} : <f16> -> <i16>
                            %303 = ascvec.reinterpret %302 : <i16> -> <ui16>
                            %304 = ascvec.shr %292, 4, %272 : <ui16> -> <ui16>
                            %305 = ascvec.add %303, %304, %272 : <ui16>
                            %306 = ascvec.shr %305, 12, %272 : <ui16> -> <ui16>
                            %307 = ascvec.add_scalar %306, %c448_i16, %272 : <ui16>, i16
                            %308 = ascvec.select %307, %298, %295 : <ui16>
                            %309 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %310 = ascvec.and %309, %279, %272 : <ui16>, <ui16> -> <ui16>
                            %311 = ascvec.cmp_lt_scalar %310, %c14464_i16, %272 : <ui16>, i16 -> !ascvec.mask
                            %312 = ascvec.shr %291, 11, %272 : <ui16> -> <ui16>
                            %313 = ascvec.add %310, %312, %272 : <ui16>
                            %314 = ascvec.shr %313, 5, %272 : <ui16> -> <ui16>
                            %315 = ascvec.reinterpret %310 : <ui16> -> <bf16>
                            %316 = ascvec.mul %315, %280, %311 : <bf16>
                            %317 = ascvec.cast %316, %272 {rounding = 4 : i32} : <bf16> -> <f16>
                            %318 = ascvec.cast %317, %272 {rounding = 4 : i32} : <f16> -> <i16>
                            %319 = ascvec.reinterpret %318 : <i16> -> <ui16>
                            %320 = ascvec.shr %291, 4, %272 : <ui16> -> <ui16>
                            %321 = ascvec.add %319, %320, %272 : <ui16>
                            %322 = ascvec.shr %321, 12, %272 : <ui16> -> <ui16>
                            %323 = ascvec.add_scalar %322, %c448_i16, %272 : <ui16>, i16
                            %324 = ascvec.select %323, %314, %311 : <ui16>
                            %325 = ascvec.and %308, %274, %272 : <ui16>, <ui16> -> <ui16>
                            %326 = ascvec.shl %324, 8, %272 : <ui16> -> <ui16>
                            %327 = ascvec.or %325, %326, %272 : <ui16>, <ui16> -> <ui16>
                            %328 = ascvec.reinterpret %327 : <ui16> -> <ui8>
                            %329 = ascvec.add_scalar %328, %c64_i8, %273 : <ui8>, i8
                            %330 = ascvec.min_scalar %329, %c123_i8, %273 : <ui8>, i8
                            %331 = arith.muli %arg15, %c512_i32 : i32
                            %332 = arith.index_cast %331 : i32 to index
                            %result0_10, %result1_11 = ascvec.loadalign_deintlv %arg11[%332] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %333 = ascvec.and %result1_11, %275, %273 : <ui8>, <ui8> -> <ui8>
                            %334 = ascvec.or %330, %333, %273 : <ui8>, <ui8> -> <ui8>
                            ascvec.storealign %arg12[%282], %334, %273 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
                      } else {
                        %268 = arith.trunci %arg8 : i64 to i32
                        %269 = arith.trunci %258 : i64 to i32
                        ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %270 = ascscalar.bitcast_memref %103 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%103, %alloc_7, %270, %251, %alloc_4, %268, %269 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<8192xi32, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<?xui16, 1>, %arg13: memref<32768xui8, 1>, %arg14: i32, %arg15: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_9 = arith.constant 1 : i32
                          %c127_i8 = arith.constant 127 : i8
                          %c16_i32 = arith.constant 16 : i32
                          %c0_i16 = arith.constant 0 : i16
                          %c123_i8 = arith.constant 123 : i8
                          %c-1_i8 = arith.constant -1 : i8
                          %c60_i8 = arith.constant 60 : i8
                          %cst = arith.constant 6.10351526E-5 : f32
                          %c64_i32 = arith.constant 64 : i32
                          %c32767_i16 = arith.constant 32767 : i16
                          %c0_10 = arith.constant 0 : index
                          %c256_i32 = arith.constant 256 : i32
                          %c65535_i32 = arith.constant 65535 : i32
                          %c3_i8 = arith.constant 3 : i8
                          %c-128_i8 = arith.constant -128 : i8
                          %c0_i8 = arith.constant 0 : i8
                          %c8_i32 = arith.constant 8 : i32
                          %271 = ascvec.create_mask : !ascvec.mask
                          %272 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %273 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                          %274 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %275 = ascvec.duplicate_scalar %c-128_i8, %273 : i8 -> <ui8>
                          %276 = ascvec.shr %274, 5, %273 : <ui8> -> <ui8>
                          %277 = ascvec.duplicate_scalar %c3_i8, %273 : i8 -> <ui8>
                          %278 = ascvec.and %274, %277, %273 : <ui8>, <ui8> -> <ui8>
                          %279 = ascvec.shl %278, 6, %273 : <ui8> -> <ui8>
                          %280 = ascvec.shr %274, 2, %273 : <ui8> -> <ui8>
                          %281 = ascvec.or %279, %280, %273 : <ui8>, <ui8> -> <ui8>
                          %282 = ascvec.duplicate_scalar %c65535_i32, %271 : i32 -> <ui32>
                          scf.for %arg16 = %arg14 to %arg15 step %c1_i32_9  : i32 {
                            %283 = arith.muli %arg16, %c256_i32 : i32
                            %284 = arith.index_cast %283 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg9[%284] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %285 = ascvec.duplicate_scalar %c32767_i16, %272 : i16 -> <ui16>
                            %286 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %287 = ascvec.and %286, %285, %272 : <ui16>, <ui16> -> <ui16>
                            %288 = ascvec.reinterpret %287 : <ui16> -> <bf16>
                            %289 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %290 = ascvec.and %289, %285, %272 : <ui16>, <ui16> -> <ui16>
                            %291 = ascvec.reinterpret %290 : <ui16> -> <bf16>
                            %292 = ascvec.cast %288, %272 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %293 = ascvec.cast %288, %272 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %294 = ascvec.cast %291, %272 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %295 = ascvec.cast %291, %272 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %296 = arith.muli %arg16, %c64_i32 : i32
                            %297 = arith.index_cast %296 : i32 to index
                            %298 = ascvec.loadalign %arg10[%297] : memref<8192xi32, 1>, index -> <i32>
                            %299 = ascvec.reinterpret %298 : <i32> -> <ui32>
                            %300 = ascvec.shl %299, 8, %271 : <ui32> -> <ui32>
                            %301 = ascvec.shr %299, 24, %271 : <ui32> -> <ui32>
                            %302 = ascvec.or %300, %301, %271 : <ui32>, <ui32> -> <ui32>
                            %303 = ascvec.shr %299, 16, %271 : <ui32> -> <ui32>
                            %304 = ascvec.and %302, %282, %271 : <ui32>, <ui32> -> <ui32>
                            %305 = ascvec.and %299, %282, %271 : <ui32>, <ui32> -> <ui32>
                            %306 = ascvec.shr %302, 16, %271 : <ui32> -> <ui32>
                            %307 = ascvec.reinterpret %292 : <f32> -> <ui32>
                            %308 = ascvec.reinterpret %307 : <ui32> -> <f32>
                            %309 = ascvec.max_scalar %308, %cst, %271 : <f32>, f32
                            %310 = ascvec.add %308, %309, %271 : <f32>
                            %311 = ascvec.reinterpret %310 : <f32> -> <ui32>
                            %312 = ascvec.shr %311, 5, %271 : <ui32> -> <ui32>
                            %313 = ascvec.add %312, %303, %271 : <ui32>
                            %314 = ascvec.reinterpret %294 : <f32> -> <ui32>
                            %315 = ascvec.reinterpret %314 : <ui32> -> <f32>
                            %316 = ascvec.max_scalar %315, %cst, %271 : <f32>, f32
                            %317 = ascvec.add %315, %316, %271 : <f32>
                            %318 = ascvec.reinterpret %317 : <f32> -> <ui32>
                            %319 = ascvec.shr %318, 5, %271 : <ui32> -> <ui32>
                            %320 = ascvec.add %319, %304, %271 : <ui32>
                            %321 = ascvec.reinterpret %293 : <f32> -> <ui32>
                            %322 = ascvec.reinterpret %321 : <ui32> -> <f32>
                            %323 = ascvec.max_scalar %322, %cst, %271 : <f32>, f32
                            %324 = ascvec.add %322, %323, %271 : <f32>
                            %325 = ascvec.reinterpret %324 : <f32> -> <ui32>
                            %326 = ascvec.shr %325, 5, %271 : <ui32> -> <ui32>
                            %327 = ascvec.add %326, %305, %271 : <ui32>
                            %328 = ascvec.reinterpret %295 : <f32> -> <ui32>
                            %329 = ascvec.reinterpret %328 : <ui32> -> <f32>
                            %330 = ascvec.max_scalar %329, %cst, %271 : <f32>, f32
                            %331 = ascvec.add %329, %330, %271 : <f32>
                            %332 = ascvec.reinterpret %331 : <f32> -> <ui32>
                            %333 = ascvec.shr %332, 5, %271 : <ui32> -> <ui32>
                            %334 = ascvec.add %333, %306, %271 : <ui32>
                            %335 = ascvec.reinterpret %313 : <ui32> -> <ui16>
                            %336 = ascvec.reinterpret %320 : <ui32> -> <ui16>
                            %result0_11, %result1_12 = ascvec.deintlv %335, %336 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %337 = ascvec.reinterpret %327 : <ui32> -> <ui16>
                            %338 = ascvec.reinterpret %334 : <ui32> -> <ui16>
                            %result0_13, %result1_14 = ascvec.deintlv %337, %338 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %339 = ascvec.reinterpret %result1_12 : <ui16> -> <ui8>
                            %340 = ascvec.reinterpret %result1_14 : <ui16> -> <ui8>
                            %result0_15, %result1_16 = ascvec.deintlv %339, %340 : <ui8>, <ui8> -> <ui8>, <ui8>
                            %341 = ascvec.gather_reg %result0_15, %281 : <ui8>, <ui8> -> <ui8>
                            %342 = ascvec.add_scalar %341, %c60_i8, %273 : <ui8>, i8
                            %343 = ascvec.cmp_ne_scalar %342, %c-1_i8, %273 : <ui8>, i8 -> !ascvec.mask
                            %344 = ascvec.min_scalar %342, %c123_i8, %343 : <ui8>, i8
                            %345 = arith.muli %arg16, %c512_i32 : i32
                            %346 = arith.index_cast %345 : i32 to index
                            %result0_17, %result1_18 = ascvec.loadalign_deintlv %arg11[%346] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %347 = ascvec.and %result1_18, %275, %273 : <ui8>, <ui8> -> <ui8>
                            %348 = ascvec.or %344, %347, %273 : <ui8>, <ui8> -> <ui8>
                            %349 = ascvec.arange %c0_i16 : i16 -> <ui16>
                            %350 = arith.muli %arg16, %c16_i32 : i32
                            %351 = arith.trunci %350 : i32 to i16
                            %352 = ascvec.add_scalar %349, %351, %272 : <ui16>, i16
                            %353 = ascvec.gather %arg12[%c0_10], %352, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                            %354 = ascvec.pack %353 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                            %355 = ascvec.gather_reg %354, %276 : <ui8>, <ui8> -> <ui8>
                            %356 = ascvec.cmp_eq_scalar %355, %c-1_i8, %273 : <ui8>, i8 -> !ascvec.mask
                            %357 = ascvec.duplicate_scalar %c127_i8, %356 merge %348 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                            ascvec.storealign %arg13[%284], %357, %273 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
                      }
                    }
                  }
                }
              } else {
                %199 = arith.divsi %62, %c256_i64 : i64
                %200 = arith.addi %62, %c1023_i64 : i64
                %201 = arith.divsi %200, %c1024_i64 : i64
                %202 = arith.trunci %arg4 : i64 to i32
                %203 = arith.extui %202 : i32 to i64
                %204 = arith.subi %arg4, %203 : i64
                %205 = arith.remsi %204, %c4294967296_i64 : i64
                %206 = arith.cmpi ne, %205, %c0_i64 : i64
                %207 = arith.cmpi slt, %205, %c0_i64 : i64
                %208 = arith.andi %206, %207 : i1
                %209 = arith.divsi %204, %c4294967296_i64 : i64
                %210 = arith.extui %208 : i1 to i64
                %211 = arith.subi %209, %210 : i64
                %212 = arith.trunci %211 : i64 to i32
                %213 = arith.divui %104, %c16_i64 : i64
                %214 = arith.trunci %213 : i64 to i32
                %215 = arith.extui %214 : i32 to i64
                %216 = arith.subi %213, %215 : i64
                %217 = arith.remsi %216, %c4294967296_i64 : i64
                %218 = arith.cmpi ne, %217, %c0_i64 : i64
                %219 = arith.cmpi slt, %217, %c0_i64 : i64
                %220 = arith.andi %218, %219 : i1
                %221 = arith.divsi %216, %c4294967296_i64 : i64
                %222 = arith.extui %220 : i1 to i64
                %223 = arith.subi %221, %222 : i64
                %224 = arith.trunci %223 : i64 to i32
                %225 = arith.trunci %arg5 : i64 to i32
                %226 = arith.extui %225 : i32 to i64
                %227 = arith.subi %arg5, %226 : i64
                %228 = arith.remsi %227, %c4294967296_i64 : i64
                %229 = arith.cmpi ne, %228, %c0_i64 : i64
                %230 = arith.cmpi slt, %228, %c0_i64 : i64
                %231 = arith.andi %229, %230 : i1
                %232 = arith.divsi %227, %c4294967296_i64 : i64
                %233 = arith.extui %231 : i1 to i64
                %234 = arith.subi %232, %233 : i64
                %235 = arith.trunci %234 : i64 to i32
                %236 = arith.divui %225, %c4_i32 : i32
                %237 = arith.remui %235, %c4_i32 : i32
                %238 = arith.muli %237, %c1073741824_i32 : i32
                %239 = arith.addi %236, %238 : i32
                %240 = arith.divui %235, %c4_i32 : i32
                %241 = arith.muli %201, %c64_i64 : i64
                %242 = arith.divui %213, %c16384_i64 : i64
                %243 = arith.addi %213, %241 : i64
                %244 = arith.subi %243, %c1_i64 : i64
                %245 = arith.divui %244, %c16384_i64 : i64
                %246 = arith.cmpi eq, %242, %245 : i64
                scf.if %246 {
                  %252 = arith.trunci %242 : i64 to i32
                  %253 = arith.extui %252 : i32 to i64
                  %254 = arith.subi %242, %253 : i64
                  %255 = arith.remsi %254, %c4294967296_i64 : i64
                  %256 = arith.cmpi ne, %255, %c0_i64 : i64
                  %257 = arith.cmpi slt, %255, %c0_i64 : i64
                  %258 = arith.andi %256, %257 : i1
                  %259 = arith.divsi %254, %c4294967296_i64 : i64
                  %260 = arith.extui %258 : i1 to i64
                  %261 = arith.subi %259, %260 : i64
                  %262 = arith.trunci %261 : i64 to i32
                  %263 = arith.addi %239, %252 : i32
                  %264 = arith.addi %240, %262 : i32
                  %265 = arith.cmpi ult, %263, %239 : i32
                  %266 = arith.extui %265 : i1 to i32
                  %267 = arith.addi %264, %266 : i32
                  %268 = arith.extui %263 : i32 to i64
                  %269 = arith.muli %268, %c3449720151_i64 : i64
                  %270 = arith.trunci %269 : i64 to i32
                  %271 = arith.extui %270 : i32 to i64
                  %272 = arith.subi %269, %271 : i64
                  %273 = arith.remsi %272, %c4294967296_i64 : i64
                  %274 = arith.cmpi ne, %273, %c0_i64 : i64
                  %275 = arith.cmpi slt, %273, %c0_i64 : i64
                  %276 = arith.andi %274, %275 : i1
                  %277 = arith.divsi %272, %c4294967296_i64 : i64
                  %278 = arith.extui %276 : i1 to i64
                  %279 = arith.subi %277, %278 : i64
                  %280 = arith.trunci %279 : i64 to i32
                  %281 = arith.xori %267, %212 : i32
                  %282 = arith.extui %281 : i32 to i64
                  %283 = arith.muli %282, %c3449720151_i64 : i64
                  %284 = arith.trunci %283 : i64 to i32
                  %285 = arith.extui %284 : i32 to i64
                  %286 = arith.subi %283, %285 : i64
                  %287 = arith.remsi %286, %c4294967296_i64 : i64
                  %288 = arith.cmpi ne, %287, %c0_i64 : i64
                  %289 = arith.cmpi slt, %287, %c0_i64 : i64
                  %290 = arith.andi %288, %289 : i1
                  %291 = arith.divsi %286, %c4294967296_i64 : i64
                  %292 = arith.extui %290 : i1 to i64
                  %293 = arith.subi %291, %292 : i64
                  %294 = arith.trunci %293 : i64 to i32
                  %295 = arith.addi %202, %c-1640531527_i32 : i32
                  %296 = arith.addi %212, %c-1150833019_i32 : i32
                  %297 = arith.xori %294, %270 : i32
                  %298 = arith.xori %297, %295 : i32
                  %299 = arith.extui %298 : i32 to i64
                  %300 = arith.muli %299, %c3528531795_i64 : i64
                  %301 = arith.trunci %300 : i64 to i32
                  %302 = arith.extui %301 : i32 to i64
                  %303 = arith.subi %300, %302 : i64
                  %304 = arith.remsi %303, %c4294967296_i64 : i64
                  %305 = arith.cmpi ne, %304, %c0_i64 : i64
                  %306 = arith.cmpi slt, %304, %c0_i64 : i64
                  %307 = arith.andi %305, %306 : i1
                  %308 = arith.divsi %303, %c4294967296_i64 : i64
                  %309 = arith.extui %307 : i1 to i64
                  %310 = arith.subi %308, %309 : i64
                  %311 = arith.trunci %310 : i64 to i32
                  %312 = arith.xori %280, %202 : i32
                  %313 = arith.addi %202, %c1013904242_i32 : i32
                  %314 = arith.xori %284, %313 : i32
                  %315 = arith.addi %212, %c1993301258_i32 : i32
                  %316 = arith.xori %311, %315 : i32
                  ascsimt.vf(%214, %312, %296, %314, %316, %202, %212, %301, %alloc_7, %241 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: memref<8192xi32, 1>, %arg17: i64):
                    %c256_i64_9 = arith.constant 256 : i64
                    %c3_i64_10 = arith.constant 3 : i64
                    %c2_i64_11 = arith.constant 2 : i64
                    %c1_i64_12 = arith.constant 1 : i64
                    %c4_i64_13 = arith.constant 4 : i64
                    %c1_i32_14 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c3_i32 = arith.constant 3 : i32
                    %c-1150833019_i32_15 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_16 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %317 = ascsimt.thread_idx {ascsimt.scope = 21 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %318 = arith.extsi %317 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg18 = %318 to %arg17 step %c256_i64_9  : i64 {
                      %319 = arith.trunci %arg18 : i64 to i32
                      %320 = arith.addi %arg8, %319 : i32
                      %321 = arith.remui %320, %c16384_i32 : i32
                      %322 = arith.xori %321, %arg9 : i32
                      %323 = arith.muli %322, %c-766435501_i32 : i32
                      %324 = "ascsimt.bitcast"(%322) : (i32) -> ui32
                      %325 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %326 = ascsimt.math_binary %324, %325, "mulhi" : ui32, ui32 -> ui32
                      %327 = "ascsimt.bitcast"(%326) : (ui32) -> i32
                      %328 = arith.xori %327, %arg10 : i32
                      %329 = arith.muli %328, %c-845247145_i32 : i32
                      %330 = "ascsimt.bitcast"(%328) : (i32) -> ui32
                      %331 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %332 = ascsimt.math_binary %330, %331, "mulhi" : ui32, ui32 -> ui32
                      %333 = "ascsimt.bitcast"(%332) : (ui32) -> i32
                      %334 = arith.xori %333, %arg11 : i32
                      %335 = arith.xori %323, %arg12 : i32
                      %336:4 = scf.for %arg19 = %c3_i32 to %c10_i32 step %c1_i32_14 iter_args(%arg20 = %334, %arg21 = %329, %arg22 = %335, %arg23 = %arg15) -> (i32, i32, i32, i32)  : i32 {
                        %345 = arith.muli %arg20, %c-766435501_i32 : i32
                        %346 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                        %347 = ascsimt.math_binary %346, %325, "mulhi" : ui32, ui32 -> ui32
                        %348 = "ascsimt.bitcast"(%347) : (ui32) -> i32
                        %349 = arith.muli %arg22, %c-845247145_i32 : i32
                        %350 = "ascsimt.bitcast"(%arg22) : (i32) -> ui32
                        %351 = ascsimt.math_binary %350, %331, "mulhi" : ui32, ui32 -> ui32
                        %352 = "ascsimt.bitcast"(%351) : (ui32) -> i32
                        %353 = arith.muli %arg19, %c-1640531527_i32_16 : i32
                        %354 = arith.addi %arg13, %353 : i32
                        %355 = arith.muli %arg19, %c-1150833019_i32_15 : i32
                        %356 = arith.addi %arg14, %355 : i32
                        %357 = arith.xori %352, %arg21 : i32
                        %358 = arith.xori %357, %354 : i32
                        %359 = arith.xori %348, %arg23 : i32
                        %360 = arith.xori %359, %356 : i32
                        scf.yield %358, %349, %360, %345 : i32, i32, i32, i32
                      }
                      %337 = arith.muli %arg18, %c4_i64_13 : i64
                      %338 = arith.index_cast %337 : i64 to index
                      ascsimt.store %arg16[%338], %336#0 : memref<8192xi32, 1>, i32
                      %339 = arith.addi %337, %c1_i64_12 : i64
                      %340 = arith.index_cast %339 : i64 to index
                      ascsimt.store %arg16[%340], %336#1 : memref<8192xi32, 1>, i32
                      %341 = arith.addi %337, %c2_i64_11 : i64
                      %342 = arith.index_cast %341 : i64 to index
                      ascsimt.store %arg16[%342], %336#2 : memref<8192xi32, 1>, i32
                      %343 = arith.addi %337, %c3_i64_10 : i64
                      %344 = arith.index_cast %343 : i64 to index
                      ascsimt.store %arg16[%344], %336#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%214, %224, %239, %240, %202, %212, %alloc_7, %241 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: memref<8192xi32, 1>, %arg15: i64):
                    %c256_i64_9 = arith.constant 256 : i64
                    %c3_i64_10 = arith.constant 3 : i64
                    %c2_i64_11 = arith.constant 2 : i64
                    %c1_i64_12 = arith.constant 1 : i64
                    %c4_i64_13 = arith.constant 4 : i64
                    %c1_i32_14 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c0_i32_15 = arith.constant 0 : i32
                    %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c262144_i32 = arith.constant 262144 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %252 = ascsimt.thread_idx {ascsimt.scope = 22 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %253 = arith.extsi %252 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg16 = %253 to %arg15 step %c256_i64_9  : i64 {
                      %254 = arith.trunci %arg16 : i64 to i32
                      %255 = arith.addi %arg8, %254 : i32
                      %256 = arith.cmpi ult, %255, %arg8 : i32
                      %257 = arith.extui %256 : i1 to i32
                      %258 = arith.addi %arg9, %257 : i32
                      %259 = arith.divui %255, %c16384_i32 : i32
                      %260 = arith.remui %258, %c16384_i32 : i32
                      %261 = arith.muli %260, %c262144_i32 : i32
                      %262 = arith.addi %259, %261 : i32
                      %263 = arith.divui %258, %c16384_i32 : i32
                      %264 = arith.addi %arg10, %262 : i32
                      %265 = arith.addi %arg11, %263 : i32
                      %266 = arith.cmpi ult, %264, %arg10 : i32
                      %267 = arith.extui %266 : i1 to i32
                      %268 = arith.addi %265, %267 : i32
                      %269 = arith.remui %255, %c16384_i32 : i32
                      %270:4 = scf.for %arg17 = %c0_i32_15 to %c10_i32 step %c1_i32_14 iter_args(%arg18 = %c0_i32_15, %arg19 = %269, %arg20 = %264, %arg21 = %268) -> (i32, i32, i32, i32)  : i32 {
                        %279 = arith.muli %arg18, %c-766435501_i32 : i32
                        %280 = "ascsimt.bitcast"(%arg18) : (i32) -> ui32
                        %281 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %282 = ascsimt.math_binary %280, %281, "mulhi" : ui32, ui32 -> ui32
                        %283 = "ascsimt.bitcast"(%282) : (ui32) -> i32
                        %284 = arith.muli %arg20, %c-845247145_i32 : i32
                        %285 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                        %286 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %287 = ascsimt.math_binary %285, %286, "mulhi" : ui32, ui32 -> ui32
                        %288 = "ascsimt.bitcast"(%287) : (ui32) -> i32
                        %289 = arith.muli %arg17, %c-1640531527_i32_17 : i32
                        %290 = arith.addi %arg12, %289 : i32
                        %291 = arith.muli %arg17, %c-1150833019_i32_16 : i32
                        %292 = arith.addi %arg13, %291 : i32
                        %293 = arith.xori %288, %arg19 : i32
                        %294 = arith.xori %293, %290 : i32
                        %295 = arith.xori %283, %arg21 : i32
                        %296 = arith.xori %295, %292 : i32
                        scf.yield %294, %284, %296, %279 : i32, i32, i32, i32
                      }
                      %271 = arith.muli %arg16, %c4_i64_13 : i64
                      %272 = arith.index_cast %271 : i64 to index
                      ascsimt.store %arg14[%272], %270#0 : memref<8192xi32, 1>, i32
                      %273 = arith.addi %271, %c1_i64_12 : i64
                      %274 = arith.index_cast %273 : i64 to index
                      ascsimt.store %arg14[%274], %270#1 : memref<8192xi32, 1>, i32
                      %275 = arith.addi %271, %c2_i64_11 : i64
                      %276 = arith.index_cast %275 : i64 to index
                      ascsimt.store %arg14[%276], %270#2 : memref<8192xi32, 1>, i32
                      %277 = arith.addi %271, %c3_i64_10 : i64
                      %278 = arith.index_cast %277 : i64 to index
                      ascsimt.store %arg14[%278], %270#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %247 = arith.trunci %44 : i64 to i32
                %248 = arith.divsi %247, %c2_i32 : i32
                %249 = arith.trunci %248 : i32 to i8
                %250 = arith.trunci %199 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                %251 = ascscalar.bitcast_memref %103 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                ascvec.vf(%249, %alloc_5, %103, %251, %alloc_7, %alloc_4, %250 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui8, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, i32) {
                ^bb0(%arg8: i8, %arg9: memref<2048xui8, 1>, %arg10: memref<32768xbf16, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<8192xi32, 1>, %arg13: memref<32768xui8, 1>, %arg14: i32):
                  %c512_i32 = arith.constant 512 : i32
                  %c1_i32_9 = arith.constant 1 : i32
                  %c2139062143_i32 = arith.constant 2139062143 : i32
                  %c-8323200_i32 = arith.constant -8323200 : i32
                  %c123_i8 = arith.constant 123 : i8
                  %c-1_i8 = arith.constant -1 : i8
                  %c60_i8 = arith.constant 60 : i8
                  %cst = arith.constant 6.10351526E-5 : f32
                  %c64_i32 = arith.constant 64 : i32
                  %c0_i16 = arith.constant 0 : i16
                  %c0_i32_10 = arith.constant 0 : i32
                  %c254_i32 = arith.constant 254 : i32
                  %c-15_i32 = arith.constant -15 : i32
                  %c15_i32 = arith.constant 15 : i32
                  %c255_i32 = arith.constant 255 : i32
                  %c32767_i16 = arith.constant 32767 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c0_11 = arith.constant 0 : index
                  %c32_i8 = arith.constant 32 : i8
                  %c65535_i32 = arith.constant 65535 : i32
                  %c3_i8 = arith.constant 3 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c-128_i8 = arith.constant -128 : i8
                  %252 = ascvec.create_mask : !ascvec.mask
                  %253 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %254 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %255 = ascvec.duplicate_scalar %c-128_i8, %254 : i8 -> <ui8>
                  %256 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %257 = ascvec.duplicate_scalar %c3_i8, %254 : i8 -> <ui8>
                  %258 = ascvec.and %256, %257, %254 : <ui8>, <ui8> -> <ui8>
                  %259 = ascvec.shl %258, 6, %254 : <ui8> -> <ui8>
                  %260 = ascvec.shr %256, 2, %254 : <ui8> -> <ui8>
                  %261 = ascvec.or %259, %260, %254 : <ui8>, <ui8> -> <ui8>
                  %262 = ascvec.duplicate_scalar %c65535_i32, %252 : i32 -> <ui32>
                  %263 = ascvec.cmp_lt_scalar %256, %c32_i8, %254 : <ui8>, i8 -> !ascvec.mask
                  %264 = ascvec.cmp_lt_scalar %258, %arg8, %254 : <ui8>, i8 -> !ascvec.mask
                  %265 = ascvec.mask_and %263, %264, %254
                  ascvec.store_unalign_begin %arg9 : memref<2048xui8, 1>
                  scf.for %arg15 = %c0_i32_10 to %arg14 step %c1_i32_9  : i32 {
                    %266 = arith.muli %arg15, %c256_i32 : i32
                    %267 = arith.index_cast %266 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg10[%267] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %268 = ascvec.duplicate_scalar %c32767_i16, %253 : i16 -> <ui16>
                    %269 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %270 = ascvec.and %269, %268, %253 : <ui16>, <ui16> -> <ui16>
                    %271 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %272 = ascvec.and %271, %268, %253 : <ui16>, <ui16> -> <ui16>
                    %273 = ascvec.max %270, %272, %253 : <ui16>
                    %274 = ascvec.reduce_max_datablock %273, %253 : <ui16>
                    %275 = ascvec.unpack %274 {part = "lower"} : <ui16> -> <ui32>
                    %276 = ascvec.shl %275, 16, %252 : <ui32> -> <ui32>
                    %277 = arith.muli %arg15, %c512_i32 : i32
                    %278 = arith.index_cast %277 : i32 to index
                    %result0_13, %result1_14 = ascvec.loadalign_deintlv %arg11[%278] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                    %279 = ascvec.and %result1_14, %255, %254 : <ui8>, <ui8> -> <ui8>
                    %280 = ascvec.reinterpret %279 : <ui8> -> <ui32>
                    %281 = ascvec.shr %276, 23, %252 : <ui32> -> <ui32>
                    %282 = ascvec.cmp_ge_scalar %281, %c255_i32, %252 : <ui32>, i32 -> !ascvec.mask
                    %283 = ascvec.max_scalar %281, %c15_i32, %252 : <ui32>, i32
                    %284 = ascvec.add_scalar %283, %c-15_i32, %252 : <ui32>, i32
                    %285 = ascvec.duplicate_scalar %c255_i32, %282 merge %284 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                    %286 = ascvec.duplicate_scalar %c254_i32, %252 : i32 -> <ui32>
                    %287 = ascvec.sub %286, %285, %252 : <ui32>
                    %288 = ascvec.cmp_ne_scalar %285, %c0_i32_10, %252 : <ui32>, i32 -> !ascvec.mask
                    %289 = ascvec.shl %287, 23, %288 : <ui32> -> <ui32>
                    %290 = ascvec.reinterpret %289 : <ui32> -> <f32>
                    %291 = ascvec.reinterpret %290 : <f32> -> <ui32>
                    %292 = ascvec.shr %291, 16, %252 : <ui32> -> <ui32>
                    %293 = ascvec.pack %292 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                    %294 = ascvec.arange %c0_i16 : i16 -> <ui16>
                    %295 = ascvec.shr %294, 4, %253 : <ui16> -> <ui16>
                    %296 = ascvec.gather_reg %293, %295 : <ui16>, <ui16> -> <ui16>
                    %297 = ascvec.reinterpret %296 : <ui16> -> <bf16>
                    %298 = ascvec.reinterpret %270 : <ui16> -> <bf16>
                    %299 = ascvec.mul %298, %297, %253 : <bf16>
                    %300 = ascvec.reinterpret %272 : <ui16> -> <bf16>
                    %301 = ascvec.mul %300, %297, %253 : <bf16>
                    %302 = ascvec.cast %299, %253 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %303 = ascvec.cast %299, %253 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %304 = ascvec.cast %301, %253 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %305 = ascvec.cast %301, %253 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %306 = ascvec.reinterpret %297 : <bf16> -> <ui32>
                    %307 = arith.muli %arg15, %c64_i32 : i32
                    %308 = arith.index_cast %307 : i32 to index
                    %309 = ascvec.loadalign %arg12[%308] : memref<8192xi32, 1>, index -> <i32>
                    %310 = ascvec.reinterpret %309 : <i32> -> <ui32>
                    %311 = ascvec.shl %310, 8, %252 : <ui32> -> <ui32>
                    %312 = ascvec.shr %310, 24, %252 : <ui32> -> <ui32>
                    %313 = ascvec.or %311, %312, %252 : <ui32>, <ui32> -> <ui32>
                    %314 = ascvec.shr %310, 16, %252 : <ui32> -> <ui32>
                    %315 = ascvec.and %313, %262, %252 : <ui32>, <ui32> -> <ui32>
                    %316 = ascvec.and %310, %262, %252 : <ui32>, <ui32> -> <ui32>
                    %317 = ascvec.shr %313, 16, %252 : <ui32> -> <ui32>
                    %318 = ascvec.reinterpret %302 : <f32> -> <ui32>
                    %319 = ascvec.reinterpret %318 : <ui32> -> <f32>
                    %320 = ascvec.max_scalar %319, %cst, %252 : <f32>, f32
                    %321 = ascvec.add %319, %320, %252 : <f32>
                    %322 = ascvec.reinterpret %321 : <f32> -> <ui32>
                    %323 = ascvec.shr %322, 5, %252 : <ui32> -> <ui32>
                    %324 = ascvec.add %323, %314, %252 : <ui32>
                    %325 = ascvec.reinterpret %304 : <f32> -> <ui32>
                    %326 = ascvec.reinterpret %325 : <ui32> -> <f32>
                    %327 = ascvec.max_scalar %326, %cst, %252 : <f32>, f32
                    %328 = ascvec.add %326, %327, %252 : <f32>
                    %329 = ascvec.reinterpret %328 : <f32> -> <ui32>
                    %330 = ascvec.shr %329, 5, %252 : <ui32> -> <ui32>
                    %331 = ascvec.add %330, %315, %252 : <ui32>
                    %332 = ascvec.reinterpret %303 : <f32> -> <ui32>
                    %333 = ascvec.reinterpret %332 : <ui32> -> <f32>
                    %334 = ascvec.max_scalar %333, %cst, %252 : <f32>, f32
                    %335 = ascvec.add %333, %334, %252 : <f32>
                    %336 = ascvec.reinterpret %335 : <f32> -> <ui32>
                    %337 = ascvec.shr %336, 5, %252 : <ui32> -> <ui32>
                    %338 = ascvec.add %337, %316, %252 : <ui32>
                    %339 = ascvec.reinterpret %305 : <f32> -> <ui32>
                    %340 = ascvec.reinterpret %339 : <ui32> -> <f32>
                    %341 = ascvec.max_scalar %340, %cst, %252 : <f32>, f32
                    %342 = ascvec.add %340, %341, %252 : <f32>
                    %343 = ascvec.reinterpret %342 : <f32> -> <ui32>
                    %344 = ascvec.shr %343, 5, %252 : <ui32> -> <ui32>
                    %345 = ascvec.add %344, %317, %252 : <ui32>
                    %346 = ascvec.reinterpret %324 : <ui32> -> <ui16>
                    %347 = ascvec.reinterpret %331 : <ui32> -> <ui16>
                    %result0_15, %result1_16 = ascvec.deintlv %346, %347 : <ui16>, <ui16> -> <ui16>, <ui16>
                    %348 = ascvec.reinterpret %338 : <ui32> -> <ui16>
                    %349 = ascvec.reinterpret %345 : <ui32> -> <ui16>
                    %result0_17, %result1_18 = ascvec.deintlv %348, %349 : <ui16>, <ui16> -> <ui16>, <ui16>
                    %350 = ascvec.reinterpret %result1_16 : <ui16> -> <ui8>
                    %351 = ascvec.reinterpret %result1_18 : <ui16> -> <ui8>
                    %result0_19, %result1_20 = ascvec.deintlv %350, %351 : <ui8>, <ui8> -> <ui8>, <ui8>
                    %352 = ascvec.gather_reg %result0_19, %261 : <ui8>, <ui8> -> <ui8>
                    %353 = ascvec.reinterpret %352 : <ui8> -> <ui32>
                    %354 = ascvec.reinterpret %353 : <ui32> -> <ui8>
                    %355 = ascvec.add_scalar %354, %c60_i8, %254 : <ui8>, i8
                    %356 = ascvec.cmp_ne_scalar %355, %c-1_i8, %254 : <ui8>, i8 -> !ascvec.mask
                    %357 = ascvec.min_scalar %355, %c123_i8, %356 : <ui8>, i8
                    %358 = ascvec.reinterpret %357 : <ui8> -> <ui32>
                    %359 = ascvec.or %358, %280, %252 : <ui32>, <ui32> -> <ui32>
                    %360 = ascvec.reinterpret %306 : <ui32> -> <ui32>
                    %361 = ascvec.cmp_eq_scalar %360, %c-8323200_i32, %252 : <ui32>, i32 -> !ascvec.mask
                    %362 = ascvec.duplicate_scalar %c2139062143_i32, %361 merge %359 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                    %363 = ascvec.reinterpret %362 : <ui32> -> <ui8>
                    ascvec.storealign %arg13[%267], %363, %254 : memref<32768xui8, 1>, index, <ui8>
                    %364 = ascvec.reinterpret %285 : <ui32> -> <ui8>
                    %365 = ascvec.squeeze_store_reg %364, %265 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_11], %365 : memref<2048xui8, 1>, <ui8>
                  }
                  %c0_12 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg15 = %c0_12 to %c1 step %c1 {
                    ascvec.store_unalign_post %arg9[%c0_11] : memref<2048xui8, 1>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              }
            }
          } else {
            %188 = arith.remsi %arg3, %c256_i64 : i64
            %189 = arith.cmpi ne, %188, %c0_i64 : i64
            %190 = arith.cmpi slt, %188, %c0_i64 : i64
            %191 = arith.andi %189, %190 : i1
            %192 = arith.select %191, %c256_i64, %c0_i64 : i64
            %193 = arith.addi %188, %192 overflow<nsw> : i64
            %194 = arith.cmpi eq, %193, %c0_i64 : i64
            scf.if %194 {
              %195 = arith.addi %62, %c63_i64 : i64
              %196 = arith.remsi %195, %c64_i64 : i64
              %197 = arith.cmpi ne, %196, %c0_i64 : i64
              %198 = arith.cmpi slt, %196, %c0_i64 : i64
              %199 = arith.andi %197, %198 : i1
              %200 = arith.divsi %195, %c64_i64 : i64
              %201 = arith.extui %199 : i1 to i64
              %202 = arith.subi %200, %201 : i64
              %203 = arith.addi %62, %c1023_i64 : i64
              %204 = arith.remsi %203, %c1024_i64 : i64
              %205 = arith.cmpi ne, %204, %c0_i64 : i64
              %206 = arith.cmpi slt, %204, %c0_i64 : i64
              %207 = arith.andi %205, %206 : i1
              %208 = arith.divsi %203, %c1024_i64 : i64
              %209 = arith.extui %207 : i1 to i64
              %210 = arith.subi %208, %209 : i64
              %211 = arith.trunci %arg4 : i64 to i32
              %212 = arith.extui %211 : i32 to i64
              %213 = arith.subi %arg4, %212 : i64
              %214 = arith.remsi %213, %c4294967296_i64 : i64
              %215 = arith.cmpi ne, %214, %c0_i64 : i64
              %216 = arith.cmpi slt, %214, %c0_i64 : i64
              %217 = arith.andi %215, %216 : i1
              %218 = arith.divsi %213, %c4294967296_i64 : i64
              %219 = arith.extui %217 : i1 to i64
              %220 = arith.subi %218, %219 : i64
              %221 = arith.trunci %220 : i64 to i32
              %222 = arith.divui %104, %c32_i64 : i64
              %223 = arith.trunci %222 : i64 to i32
              %224 = arith.extui %223 : i32 to i64
              %225 = arith.subi %222, %224 : i64
              %226 = arith.remsi %225, %c4294967296_i64 : i64
              %227 = arith.cmpi ne, %226, %c0_i64 : i64
              %228 = arith.cmpi slt, %226, %c0_i64 : i64
              %229 = arith.andi %227, %228 : i1
              %230 = arith.divsi %225, %c4294967296_i64 : i64
              %231 = arith.extui %229 : i1 to i64
              %232 = arith.subi %230, %231 : i64
              %233 = arith.trunci %232 : i64 to i32
              %234 = arith.trunci %arg5 : i64 to i32
              %235 = arith.extui %234 : i32 to i64
              %236 = arith.subi %arg5, %235 : i64
              %237 = arith.remsi %236, %c4294967296_i64 : i64
              %238 = arith.cmpi ne, %237, %c0_i64 : i64
              %239 = arith.cmpi slt, %237, %c0_i64 : i64
              %240 = arith.andi %238, %239 : i1
              %241 = arith.divsi %236, %c4294967296_i64 : i64
              %242 = arith.extui %240 : i1 to i64
              %243 = arith.subi %241, %242 : i64
              %244 = arith.trunci %243 : i64 to i32
              %245 = arith.divui %234, %c4_i32 : i32
              %246 = arith.remui %244, %c4_i32 : i32
              %247 = arith.muli %246, %c1073741824_i32 : i32
              %248 = arith.addi %245, %247 : i32
              %249 = arith.divui %244, %c4_i32 : i32
              %250 = arith.muli %210, %c32_i64 : i64
              %251 = arith.divui %222, %c16384_i64 : i64
              %252 = arith.addi %222, %250 : i64
              %253 = arith.subi %252, %c1_i64 : i64
              %254 = arith.divui %253, %c16384_i64 : i64
              %255 = arith.cmpi eq, %251, %254 : i64
              scf.if %255 {
                %261 = arith.trunci %251 : i64 to i32
                %262 = arith.extui %261 : i32 to i64
                %263 = arith.subi %251, %262 : i64
                %264 = arith.remsi %263, %c4294967296_i64 : i64
                %265 = arith.cmpi ne, %264, %c0_i64 : i64
                %266 = arith.cmpi slt, %264, %c0_i64 : i64
                %267 = arith.andi %265, %266 : i1
                %268 = arith.divsi %263, %c4294967296_i64 : i64
                %269 = arith.extui %267 : i1 to i64
                %270 = arith.subi %268, %269 : i64
                %271 = arith.trunci %270 : i64 to i32
                %272 = arith.addi %248, %261 : i32
                %273 = arith.addi %249, %271 : i32
                %274 = arith.cmpi ult, %272, %248 : i32
                %275 = arith.extui %274 : i1 to i32
                %276 = arith.addi %273, %275 : i32
                %277 = arith.extui %272 : i32 to i64
                %278 = arith.muli %277, %c3449720151_i64 : i64
                %279 = arith.trunci %278 : i64 to i32
                %280 = arith.extui %279 : i32 to i64
                %281 = arith.subi %278, %280 : i64
                %282 = arith.remsi %281, %c4294967296_i64 : i64
                %283 = arith.cmpi ne, %282, %c0_i64 : i64
                %284 = arith.cmpi slt, %282, %c0_i64 : i64
                %285 = arith.andi %283, %284 : i1
                %286 = arith.divsi %281, %c4294967296_i64 : i64
                %287 = arith.extui %285 : i1 to i64
                %288 = arith.subi %286, %287 : i64
                %289 = arith.trunci %288 : i64 to i32
                %290 = arith.xori %276, %221 : i32
                %291 = arith.extui %290 : i32 to i64
                %292 = arith.muli %291, %c3449720151_i64 : i64
                %293 = arith.trunci %292 : i64 to i32
                %294 = arith.extui %293 : i32 to i64
                %295 = arith.subi %292, %294 : i64
                %296 = arith.remsi %295, %c4294967296_i64 : i64
                %297 = arith.cmpi ne, %296, %c0_i64 : i64
                %298 = arith.cmpi slt, %296, %c0_i64 : i64
                %299 = arith.andi %297, %298 : i1
                %300 = arith.divsi %295, %c4294967296_i64 : i64
                %301 = arith.extui %299 : i1 to i64
                %302 = arith.subi %300, %301 : i64
                %303 = arith.trunci %302 : i64 to i32
                %304 = arith.addi %211, %c-1640531527_i32 : i32
                %305 = arith.addi %221, %c-1150833019_i32 : i32
                %306 = arith.xori %303, %279 : i32
                %307 = arith.xori %306, %304 : i32
                %308 = arith.extui %307 : i32 to i64
                %309 = arith.muli %308, %c3528531795_i64 : i64
                %310 = arith.trunci %309 : i64 to i32
                %311 = arith.extui %310 : i32 to i64
                %312 = arith.subi %309, %311 : i64
                %313 = arith.remsi %312, %c4294967296_i64 : i64
                %314 = arith.cmpi ne, %313, %c0_i64 : i64
                %315 = arith.cmpi slt, %313, %c0_i64 : i64
                %316 = arith.andi %314, %315 : i1
                %317 = arith.divsi %312, %c4294967296_i64 : i64
                %318 = arith.extui %316 : i1 to i64
                %319 = arith.subi %317, %318 : i64
                %320 = arith.trunci %319 : i64 to i32
                %321 = arith.xori %289, %211 : i32
                %322 = arith.addi %211, %c1013904242_i32 : i32
                %323 = arith.xori %293, %322 : i32
                %324 = arith.addi %221, %c1993301258_i32 : i32
                %325 = arith.xori %320, %324 : i32
                ascsimt.vf(%223, %321, %305, %323, %325, %211, %221, %310, %alloc_7, %250 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: memref<8192xi32, 1>, %arg17: i64):
                  %c256_i64_9 = arith.constant 256 : i64
                  %c3_i64_10 = arith.constant 3 : i64
                  %c2_i64_11 = arith.constant 2 : i64
                  %c1_i64_12 = arith.constant 1 : i64
                  %c4_i64_13 = arith.constant 4 : i64
                  %c1_i32_14 = arith.constant 1 : i32
                  %c10_i32 = arith.constant 10 : i32
                  %c3_i32 = arith.constant 3 : i32
                  %c-1150833019_i32_15 = arith.constant -1150833019 : i32
                  %c-1640531527_i32_16 = arith.constant -1640531527 : i32
                  %c-845247145_i32 = arith.constant -845247145 : i32
                  %c-766435501_i32 = arith.constant -766435501 : i32
                  %c16384_i32 = arith.constant 16384 : i32
                  %326 = ascsimt.thread_idx {ascsimt.scope = 24 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                  %327 = arith.extsi %326 {cannir.vf_threads = 256 : i64} : i32 to i64
                  scf.for %arg18 = %327 to %arg17 step %c256_i64_9  : i64 {
                    %328 = arith.trunci %arg18 : i64 to i32
                    %329 = arith.addi %arg8, %328 : i32
                    %330 = arith.remui %329, %c16384_i32 : i32
                    %331 = arith.xori %330, %arg9 : i32
                    %332 = arith.muli %331, %c-766435501_i32 : i32
                    %333 = "ascsimt.bitcast"(%331) : (i32) -> ui32
                    %334 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                    %335 = ascsimt.math_binary %333, %334, "mulhi" : ui32, ui32 -> ui32
                    %336 = "ascsimt.bitcast"(%335) : (ui32) -> i32
                    %337 = arith.xori %336, %arg10 : i32
                    %338 = arith.muli %337, %c-845247145_i32 : i32
                    %339 = "ascsimt.bitcast"(%337) : (i32) -> ui32
                    %340 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                    %341 = ascsimt.math_binary %339, %340, "mulhi" : ui32, ui32 -> ui32
                    %342 = "ascsimt.bitcast"(%341) : (ui32) -> i32
                    %343 = arith.xori %342, %arg11 : i32
                    %344 = arith.xori %332, %arg12 : i32
                    %345:4 = scf.for %arg19 = %c3_i32 to %c10_i32 step %c1_i32_14 iter_args(%arg20 = %343, %arg21 = %338, %arg22 = %344, %arg23 = %arg15) -> (i32, i32, i32, i32)  : i32 {
                      %354 = arith.muli %arg20, %c-766435501_i32 : i32
                      %355 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                      %356 = ascsimt.math_binary %355, %334, "mulhi" : ui32, ui32 -> ui32
                      %357 = "ascsimt.bitcast"(%356) : (ui32) -> i32
                      %358 = arith.muli %arg22, %c-845247145_i32 : i32
                      %359 = "ascsimt.bitcast"(%arg22) : (i32) -> ui32
                      %360 = ascsimt.math_binary %359, %340, "mulhi" : ui32, ui32 -> ui32
                      %361 = "ascsimt.bitcast"(%360) : (ui32) -> i32
                      %362 = arith.muli %arg19, %c-1640531527_i32_16 : i32
                      %363 = arith.addi %arg13, %362 : i32
                      %364 = arith.muli %arg19, %c-1150833019_i32_15 : i32
                      %365 = arith.addi %arg14, %364 : i32
                      %366 = arith.xori %361, %arg21 : i32
                      %367 = arith.xori %366, %363 : i32
                      %368 = arith.xori %357, %arg23 : i32
                      %369 = arith.xori %368, %365 : i32
                      scf.yield %367, %358, %369, %354 : i32, i32, i32, i32
                    }
                    %346 = arith.muli %arg18, %c4_i64_13 : i64
                    %347 = arith.index_cast %346 : i64 to index
                    ascsimt.store %arg16[%347], %345#0 : memref<8192xi32, 1>, i32
                    %348 = arith.addi %346, %c1_i64_12 : i64
                    %349 = arith.index_cast %348 : i64 to index
                    ascsimt.store %arg16[%349], %345#1 : memref<8192xi32, 1>, i32
                    %350 = arith.addi %346, %c2_i64_11 : i64
                    %351 = arith.index_cast %350 : i64 to index
                    ascsimt.store %arg16[%351], %345#2 : memref<8192xi32, 1>, i32
                    %352 = arith.addi %346, %c3_i64_10 : i64
                    %353 = arith.index_cast %352 : i64 to index
                    ascsimt.store %arg16[%353], %345#3 : memref<8192xi32, 1>, i32
                  } {cannir.vf_threads = 256 : i64}
                } {threads = 256 : i64}
              } else {
                ascsimt.vf(%223, %233, %248, %249, %211, %221, %alloc_7, %250 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: memref<8192xi32, 1>, %arg15: i64):
                  %c256_i64_9 = arith.constant 256 : i64
                  %c3_i64_10 = arith.constant 3 : i64
                  %c2_i64_11 = arith.constant 2 : i64
                  %c1_i64_12 = arith.constant 1 : i64
                  %c4_i64_13 = arith.constant 4 : i64
                  %c1_i32_14 = arith.constant 1 : i32
                  %c10_i32 = arith.constant 10 : i32
                  %c0_i32_15 = arith.constant 0 : i32
                  %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                  %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                  %c-845247145_i32 = arith.constant -845247145 : i32
                  %c-766435501_i32 = arith.constant -766435501 : i32
                  %c262144_i32 = arith.constant 262144 : i32
                  %c16384_i32 = arith.constant 16384 : i32
                  %261 = ascsimt.thread_idx {ascsimt.scope = 25 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                  %262 = arith.extsi %261 {cannir.vf_threads = 256 : i64} : i32 to i64
                  scf.for %arg16 = %262 to %arg15 step %c256_i64_9  : i64 {
                    %263 = arith.trunci %arg16 : i64 to i32
                    %264 = arith.addi %arg8, %263 : i32
                    %265 = arith.cmpi ult, %264, %arg8 : i32
                    %266 = arith.extui %265 : i1 to i32
                    %267 = arith.addi %arg9, %266 : i32
                    %268 = arith.divui %264, %c16384_i32 : i32
                    %269 = arith.remui %267, %c16384_i32 : i32
                    %270 = arith.muli %269, %c262144_i32 : i32
                    %271 = arith.addi %268, %270 : i32
                    %272 = arith.divui %267, %c16384_i32 : i32
                    %273 = arith.addi %arg10, %271 : i32
                    %274 = arith.addi %arg11, %272 : i32
                    %275 = arith.cmpi ult, %273, %arg10 : i32
                    %276 = arith.extui %275 : i1 to i32
                    %277 = arith.addi %274, %276 : i32
                    %278 = arith.remui %264, %c16384_i32 : i32
                    %279:4 = scf.for %arg17 = %c0_i32_15 to %c10_i32 step %c1_i32_14 iter_args(%arg18 = %c0_i32_15, %arg19 = %278, %arg20 = %273, %arg21 = %277) -> (i32, i32, i32, i32)  : i32 {
                      %288 = arith.muli %arg18, %c-766435501_i32 : i32
                      %289 = "ascsimt.bitcast"(%arg18) : (i32) -> ui32
                      %290 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %291 = ascsimt.math_binary %289, %290, "mulhi" : ui32, ui32 -> ui32
                      %292 = "ascsimt.bitcast"(%291) : (ui32) -> i32
                      %293 = arith.muli %arg20, %c-845247145_i32 : i32
                      %294 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                      %295 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %296 = ascsimt.math_binary %294, %295, "mulhi" : ui32, ui32 -> ui32
                      %297 = "ascsimt.bitcast"(%296) : (ui32) -> i32
                      %298 = arith.muli %arg17, %c-1640531527_i32_17 : i32
                      %299 = arith.addi %arg12, %298 : i32
                      %300 = arith.muli %arg17, %c-1150833019_i32_16 : i32
                      %301 = arith.addi %arg13, %300 : i32
                      %302 = arith.xori %297, %arg19 : i32
                      %303 = arith.xori %302, %299 : i32
                      %304 = arith.xori %292, %arg21 : i32
                      %305 = arith.xori %304, %301 : i32
                      scf.yield %303, %293, %305, %288 : i32, i32, i32, i32
                    }
                    %280 = arith.muli %arg16, %c4_i64_13 : i64
                    %281 = arith.index_cast %280 : i64 to index
                    ascsimt.store %arg14[%281], %279#0 : memref<8192xi32, 1>, i32
                    %282 = arith.addi %280, %c1_i64_12 : i64
                    %283 = arith.index_cast %282 : i64 to index
                    ascsimt.store %arg14[%283], %279#1 : memref<8192xi32, 1>, i32
                    %284 = arith.addi %280, %c2_i64_11 : i64
                    %285 = arith.index_cast %284 : i64 to index
                    ascsimt.store %arg14[%285], %279#2 : memref<8192xi32, 1>, i32
                    %286 = arith.addi %280, %c3_i64_10 : i64
                    %287 = arith.index_cast %286 : i64 to index
                    ascsimt.store %arg14[%287], %279#3 : memref<8192xi32, 1>, i32
                  } {cannir.vf_threads = 256 : i64}
                } {threads = 256 : i64}
              }
              %256 = arith.trunci %202 : i64 to i32
              %257 = arith.trunci %62 : i64 to i32
              %258 = arith.trunci %44 : i64 to i32
              %259 = arith.muli %256, %c2_i32 : i32
              %260 = arith.cmpi eq, %258, %c4_i32 : i32
              ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%62, %103, %alloc_7, %alloc_4, %alloc_6, %256, %257, %259, %260, %alloc_5 : i64, memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, memref<1024xui32, 1>, i32, i32, i32, i1, memref<2048xui8, 1>) {
              ^bb0(%arg8: i64, %arg9: memref<32768xbf16, 1>, %arg10: memref<8192xi32, 1>, %arg11: memref<32768xui8, 1>, %arg12: memref<1024xui32, 1>, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i1, %arg17: memref<2048xui8, 1>):
                %c2139095040_i32 = arith.constant 2139095040 : i32
                %c127_i32 = arith.constant 127 : i32
                %c123_i32 = arith.constant 123 : i32
                %c947912704_i32 = arith.constant 947912704 : i32
                %cst = arith.constant 4.2949673E+9 : f32
                %c29360128_i32 = arith.constant 29360128 : i32
                %c128_i32 = arith.constant 128 : i32
                %c65535_i32 = arith.constant 65535 : i32
                %c2_i32_9 = arith.constant 2 : i32
                %c7_i32 = arith.constant 7 : i32
                %c1_i32_10 = arith.constant 1 : i32
                %c4_i32_11 = arith.constant 4 : i32
                %c2139160576_i32 = arith.constant 2139160576 : i32
                %c254_i32 = arith.constant 254 : i32
                %c-15_i32 = arith.constant -15 : i32
                %c15_i32 = arith.constant 15 : i32
                %c255_i32 = arith.constant 255 : i32
                %c2147483647_i32 = arith.constant 2147483647 : i32
                %cst_12 = arith.constant 0.000000e+00 : f32
                %c0_13 = arith.constant 0 : index
                %c64_i32 = arith.constant 64 : i32
                %c32_i32 = arith.constant 32 : i32
                %c0_i32_14 = arith.constant 0 : i32
                %261 = ascvec.create_mask : !ascvec.mask
                %262 = ascvec.arange %c0_i32_14 : i32 -> <ui32>
                %263 = ascvec.cmp_lt_scalar %262, %c32_i32, %261 : <ui32>, i32 -> !ascvec.mask
                %264 = ascvec.cmp_ge_scalar %262, %c32_i32, %261 : <ui32>, i32 -> !ascvec.mask
                scf.for %arg18 = %c0_i32_14 to %arg13 step %c1_i32_10  : i32 {
                  %266 = arith.muli %arg18, %c64_i32 : i32
                  %267 = arith.trunci %arg8 : i64 to i32
                  %268 = arith.subi %267, %266 : i32
                  %mask, %remaining_out = ascvec.update_mask %268 : !ascvec.mask
                  %269 = arith.index_cast %266 : i32 to index
                  %270 = ascvec.loadalign_unpack %arg9[%269] {mode = "b16_to_b32"} : memref<32768xbf16, 1>, index -> <bf16>
                  %271 = ascvec.cast %270, %261 {rounding = 6 : i32} : <bf16> -> <f32>
                  %272 = ascvec.duplicate_scalar %cst_12, %261 : f32 -> <f32>
                  %273 = ascvec.select %271, %272, %mask : <f32>
                  %274 = ascvec.reinterpret %273 : <f32> -> <ui32>
                  %275 = ascvec.duplicate_scalar %c2147483647_i32, %261 : i32 -> <ui32>
                  %276 = ascvec.and %274, %275, %261 : <ui32>, <ui32> -> <ui32>
                  %277 = ascvec.reduce_max %276, %263 : <ui32>
                  %278 = ascvec.duplicate %277, %261 : <ui32> -> <ui32>
                  %279 = ascvec.reduce_max %276, %264 : <ui32>
                  %280 = ascvec.duplicate %279, %261 : <ui32> -> <ui32>
                  %281 = ascvec.select %278, %280, %263 : <ui32>
                  %282 = ascvec.shr %281, 23, %261 : <ui32> -> <ui32>
                  %283 = ascvec.cmp_ge_scalar %282, %c255_i32, %261 : <ui32>, i32 -> !ascvec.mask
                  %284 = ascvec.max_scalar %282, %c15_i32, %261 : <ui32>, i32
                  %285 = ascvec.add_scalar %284, %c-15_i32, %261 : <ui32>, i32
                  %286 = ascvec.duplicate_scalar %c255_i32, %283 merge %285 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %287 = ascvec.duplicate_scalar %c254_i32, %261 : i32 -> <ui32>
                  %288 = ascvec.sub %287, %286, %261 : <ui32>
                  %289 = ascvec.cmp_ne_scalar %286, %c0_i32_14, %261 : <ui32>, i32 -> !ascvec.mask
                  %290 = ascvec.shl %288, 23, %289 : <ui32> -> <ui32>
                  %291 = ascvec.duplicate_scalar %c2139160576_i32, %283 merge %290 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %292 = ascvec.reinterpret %291 : <ui32> -> <f32>
                  %293 = ascvec.mul %273, %292, %261 : <f32>
                  %294 = ascvec.cast %293, %261 : <f32> -> <bf16>
                  %295 = ascvec.cast %294, %261 {rounding = 6 : i32} : <bf16> -> <f32>
                  %296 = arith.remsi %arg18, %c4_i32_11 : i32
                  %297 = arith.cmpi ne, %296, %c0_i32_14 : i32
                  %298 = arith.cmpi slt, %296, %c0_i32_14 : i32
                  %299 = arith.andi %297, %298 : i1
                  %300 = arith.select %299, %c4_i32_11, %c0_i32_14 : i32
                  %301 = arith.addi %296, %300 overflow<nsw> : i32
                  %302 = arith.muli %301, %c64_i32 overflow<nsw> : i32
                  %303 = ascvec.add_scalar %262, %302, %261 : <ui32>, i32
                  %304 = arith.divsi %arg18, %c4_i32_11 : i32
                  %305 = arith.extui %299 : i1 to i32
                  %306 = arith.subi %304, %305 : i32
                  %307 = ascvec.shr %303, 3, %261 : <ui32> -> <ui32>
                  %308 = arith.muli %306, %c32_i32 : i32
                  %309 = ascvec.add_scalar %307, %308, %261 : <ui32>, i32
                  %310 = ascvec.gather %arg10[%c0_13], %309, %261 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                  %311 = ascvec.reinterpret %310 : <i32> -> <ui32>
                  %312 = ascvec.duplicate_scalar %c7_i32, %261 : i32 -> <ui32>
                  %313 = ascvec.and %303, %312, %261 : <ui32>, <ui32> -> <ui32>
                  %314 = ascvec.shl %311, 11, %261 : <ui32> -> <ui32>
                  %315 = ascvec.shr %311, 21, %261 : <ui32> -> <ui32>
                  %316 = ascvec.or %314, %315, %261 : <ui32>, <ui32> -> <ui32>
                  %317 = ascvec.shr %316, 3, %261 : <ui32> -> <ui32>
                  %318 = ascvec.shl %316, 29, %261 : <ui32> -> <ui32>
                  %319 = ascvec.or %317, %318, %261 : <ui32>, <ui32> -> <ui32>
                  %320 = ascvec.duplicate_scalar %c1_i32_10, %261 : i32 -> <ui32>
                  %321 = ascvec.and %313, %320, %261 : <ui32>, <ui32> -> <ui32>
                  %322 = ascvec.cmp_ne_scalar %321, %c0_i32_14, %261 : <ui32>, i32 -> !ascvec.mask
                  %323 = ascvec.select %319, %316, %322 : <ui32>
                  %324 = ascvec.shr %323, 8, %261 : <ui32> -> <ui32>
                  %325 = ascvec.shl %323, 24, %261 : <ui32> -> <ui32>
                  %326 = ascvec.or %324, %325, %261 : <ui32>, <ui32> -> <ui32>
                  %327 = ascvec.duplicate_scalar %c2_i32_9, %261 : i32 -> <ui32>
                  %328 = ascvec.and %313, %327, %261 : <ui32>, <ui32> -> <ui32>
                  %329 = ascvec.cmp_ne_scalar %328, %c0_i32_14, %261 : <ui32>, i32 -> !ascvec.mask
                  %330 = ascvec.select %326, %323, %329 : <ui32>
                  %331 = ascvec.shr %330, 16, %261 : <ui32> -> <ui32>
                  %332 = ascvec.shl %330, 16, %261 : <ui32> -> <ui32>
                  %333 = ascvec.or %331, %332, %261 : <ui32>, <ui32> -> <ui32>
                  %334 = ascvec.duplicate_scalar %c4_i32_11, %261 : i32 -> <ui32>
                  %335 = ascvec.and %313, %334, %261 : <ui32>, <ui32> -> <ui32>
                  %336 = ascvec.cmp_ne_scalar %335, %c0_i32_14, %261 : <ui32>, i32 -> !ascvec.mask
                  %337 = ascvec.select %333, %330, %336 : <ui32>
                  %338 = ascvec.duplicate_scalar %c65535_i32, %261 : i32 -> <ui32>
                  %339 = ascvec.and %337, %338, %261 : <ui32>, <ui32> -> <ui32>
                  %340 = ascvec.reinterpret %295 : <f32> -> <ui32>
                  %341 = ascvec.shr %340, 24, %261 : <ui32> -> <ui32>
                  %342 = ascvec.duplicate_scalar %c128_i32, %261 : i32 -> <ui32>
                  %343 = ascvec.and %341, %342, %261 : <ui32>, <ui32> -> <ui32>
                  %344 = ascvec.and %340, %275, %261 : <ui32>, <ui32> -> <ui32>
                  %345 = ascvec.shr %344, 5, %261 : <ui32> -> <ui32>
                  %346 = ascvec.duplicate_scalar %c29360128_i32, %261 : i32 -> <ui32>
                  %347 = ascvec.sub %345, %346, %261 : <ui32>
                  %348 = ascvec.reinterpret %344 : <ui32> -> <f32>
                  %349 = ascvec.mul_scalar %348, %cst, %261 : <f32>, f32
                  %350 = ascvec.cast %349, %261 {rounding = 4 : i32} : <f32> -> <i32>
                  %351 = ascvec.reinterpret %350 : <i32> -> <ui32>
                  %352 = ascvec.cmp_ge_scalar %344, %c947912704_i32, %261 : <ui32>, i32 -> !ascvec.mask
                  %353 = ascvec.select %347, %351, %352 : <ui32>
                  %354 = ascvec.and %339, %338, %261 : <ui32>, <ui32> -> <ui32>
                  %355 = ascvec.add %353, %354, %261 : <ui32>
                  %356 = ascvec.shr %355, 16, %261 : <ui32> -> <ui32>
                  %357 = ascvec.min_scalar %356, %c123_i32, %261 : <ui32>, i32
                  %358 = ascvec.or %357, %343, %261 : <ui32>, <ui32> -> <ui32>
                  %359 = ascvec.duplicate_scalar %c127_i32, %261 : i32 -> <ui32>
                  %360 = ascvec.cmp_gt_scalar %344, %c2139095040_i32, %261 : <ui32>, i32 -> !ascvec.mask
                  %361 = ascvec.select %359, %358, %360 : <ui32>
                  ascvec.storealign_pack %arg11[%269], %361, %mask {mode = "b32_to_b8"} : memref<32768xui8, 1>, index, <ui32>
                  %362 = arith.muli %arg18, %c2_i32_9 : i32
                  %363 = arith.index_cast %362 : i32 to index
                  ascvec.storealign_1st %arg12[%363], %286 : memref<1024xui32, 1>, index, <ui32>
                  %364 = ascvec.duplicate_scalar %c32_i32, %261 : i32 -> <ui32>
                  %365 = ascvec.gather_reg %286, %364 : <ui32>, <ui32> -> <ui32>
                  %366 = arith.addi %362, %c1_i32_10 : i32
                  %367 = arith.index_cast %366 : i32 to index
                  ascvec.storealign_1st %arg12[%367], %365 : memref<1024xui32, 1>, index, <ui32>
                }
                ascvec.mem_bar vst_vld
                %265 = scf.if %arg16 -> (i32) {
                  %266 = arith.remsi %arg14, %c32_i32 : i32
                  %267 = arith.cmpi ne, %266, %c0_i32_14 : i32
                  %268 = arith.cmpi slt, %266, %c0_i32_14 : i32
                  %269 = arith.andi %267, %268 : i1
                  %270 = arith.divsi %arg14, %c32_i32 : i32
                  %271 = arith.extui %269 : i1 to i32
                  %272 = arith.subi %270, %271 : i32
                  %273 = arith.muli %272, %c2_i32_9 : i32
                  scf.yield %273 : i32
                } else {
                  scf.yield %arg15 : i32
                }
                scf.for %arg18 = %c0_i32_14 to %265 step %c64_i32  : i32 {
                  %266 = arith.subi %265, %arg18 : i32
                  %mask, %remaining_out = ascvec.update_mask %266 : !ascvec.mask
                  %267 = ascvec.add_scalar %262, %arg18, %261 : <ui32>, i32
                  scf.if %arg16 {
                    %268 = ascvec.shr %267, 1, %261 : <ui32> -> <ui32>
                    %269 = ascvec.gather %arg12[%c0_13], %268, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %270 = ascvec.duplicate_scalar %c1_i32_10, %261 : i32 -> <ui32>
                    %271 = ascvec.and %262, %270, %261 : <ui32>, <ui32> -> <ui32>
                    %272 = ascvec.cmp_eq_scalar %271, %c0_i32_14, %261 : <ui32>, i32 -> !ascvec.mask
                    %273 = ascvec.duplicate_scalar %c0_i32_14, %261 : i32 -> <ui32>
                    %274 = ascvec.select %269, %273, %272 : <ui32>
                    %275 = arith.index_cast %arg18 : i32 to index
                    %c0_15 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg19 = %c0_15 to %c1 step %c1 {
                      ascvec.storealign_pack %arg17[%275], %274, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  } else {
                    %268 = ascvec.gather %arg12[%c0_13], %267, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %269 = arith.index_cast %arg18 : i32 to index
                    %c0_15 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg19 = %c0_15 to %c1 step %c1 {
                      ascvec.storealign_pack %arg17[%269], %268, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  }
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
            } else {
              %195 = arith.addi %62, %c63_i64 : i64
              %196 = arith.remsi %195, %c64_i64 : i64
              %197 = arith.cmpi ne, %196, %c0_i64 : i64
              %198 = arith.cmpi slt, %196, %c0_i64 : i64
              %199 = arith.andi %197, %198 : i1
              %200 = arith.divsi %195, %c64_i64 : i64
              %201 = arith.extui %199 : i1 to i64
              %202 = arith.subi %200, %201 : i64
              %203 = arith.addi %62, %c1023_i64 : i64
              %204 = arith.remsi %203, %c1024_i64 : i64
              %205 = arith.cmpi ne, %204, %c0_i64 : i64
              %206 = arith.cmpi slt, %204, %c0_i64 : i64
              %207 = arith.andi %205, %206 : i1
              %208 = arith.divsi %203, %c1024_i64 : i64
              %209 = arith.extui %207 : i1 to i64
              %210 = arith.subi %208, %209 : i64
              %211 = arith.trunci %arg4 : i64 to i32
              %212 = arith.extui %211 : i32 to i64
              %213 = arith.subi %arg4, %212 : i64
              %214 = arith.remsi %213, %c4294967296_i64 : i64
              %215 = arith.cmpi ne, %214, %c0_i64 : i64
              %216 = arith.cmpi slt, %214, %c0_i64 : i64
              %217 = arith.andi %215, %216 : i1
              %218 = arith.divsi %213, %c4294967296_i64 : i64
              %219 = arith.extui %217 : i1 to i64
              %220 = arith.subi %218, %219 : i64
              %221 = arith.trunci %220 : i64 to i32
              %222 = arith.remsi %104, %c16_i64 : i64
              %223 = arith.cmpi ne, %222, %c0_i64 : i64
              %224 = arith.cmpi slt, %222, %c0_i64 : i64
              %225 = arith.andi %223, %224 : i1
              %226 = arith.divsi %104, %c16_i64 : i64
              %227 = arith.extui %225 : i1 to i64
              %228 = arith.subi %226, %227 : i64
              %229 = arith.remsi %arg3, %c16_i64 : i64
              %230 = arith.cmpi ne, %229, %c0_i64 : i64
              %231 = arith.cmpi slt, %229, %c0_i64 : i64
              %232 = arith.andi %230, %231 : i1
              %233 = arith.divsi %arg3, %c16_i64 : i64
              %234 = arith.extui %232 : i1 to i64
              %235 = arith.subi %233, %234 : i64
              %236 = arith.remsi %228, %235 : i64
              %237 = arith.cmpi ne, %236, %c0_i64 : i64
              %238 = arith.cmpi slt, %236, %c0_i64 : i64
              %239 = arith.cmpi slt, %235, %c0_i64 : i64
              %240 = arith.xori %238, %239 : i1
              %241 = arith.andi %237, %240 : i1
              %242 = arith.divsi %228, %235 : i64
              %243 = arith.extui %241 : i1 to i64
              %244 = arith.subi %242, %243 : i64
              %245 = arith.select %241, %235, %c0_i64 : i64
              %246 = arith.addi %236, %245 : i64
              %247 = arith.trunci %244 : i64 to i32
              %248 = arith.extui %247 : i32 to i64
              %249 = arith.subi %244, %248 : i64
              %250 = arith.remsi %249, %c4294967296_i64 : i64
              %251 = arith.cmpi ne, %250, %c0_i64 : i64
              %252 = arith.cmpi slt, %250, %c0_i64 : i64
              %253 = arith.andi %251, %252 : i1
              %254 = arith.divsi %249, %c4294967296_i64 : i64
              %255 = arith.extui %253 : i1 to i64
              %256 = arith.subi %254, %255 : i64
              %257 = arith.trunci %256 : i64 to i32
              %258 = arith.trunci %246 : i64 to i32
              %259 = arith.extui %258 : i32 to i64
              %260 = arith.subi %246, %259 : i64
              %261 = arith.remsi %260, %c4294967296_i64 : i64
              %262 = arith.cmpi ne, %261, %c0_i64 : i64
              %263 = arith.cmpi slt, %261, %c0_i64 : i64
              %264 = arith.andi %262, %263 : i1
              %265 = arith.divsi %260, %c4294967296_i64 : i64
              %266 = arith.extui %264 : i1 to i64
              %267 = arith.subi %265, %266 : i64
              %268 = arith.trunci %267 : i64 to i32
              %269 = arith.remsi %c64_i64, %235 : i64
              %270 = arith.cmpi ne, %269, %c0_i64 : i64
              %271 = arith.cmpi slt, %269, %c0_i64 : i64
              %272 = arith.xori %271, %239 : i1
              %273 = arith.andi %270, %272 : i1
              %274 = arith.divsi %c64_i64, %235 : i64
              %275 = arith.extui %273 : i1 to i64
              %276 = arith.subi %274, %275 : i64
              %277 = arith.select %273, %235, %c0_i64 : i64
              %278 = arith.addi %269, %277 : i64
              %279 = arith.trunci %210 : i64 to i32
              ascvec.vf(%235, %arg5, %211, %221, %alloc_7, %278, %276, %279, %247, %257, %258, %268 : i64, i64, i32, i32, memref<8192xi32, 1>, i64, i64, i32, i32, i32, i32, i32) {
              ^bb0(%arg8: i64, %arg9: i64, %arg10: i32, %arg11: i32, %arg12: memref<8192xi32, 1>, %arg13: i64, %arg14: i64, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32):
                %c1_i32_9 = arith.constant 1 : i32
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
                %c1993301258_i32_10 = arith.constant 1993301258 : i32
                %c1013904242_i32_11 = arith.constant 1013904242 : i32
                %c-1150833019_i32_12 = arith.constant -1150833019 : i32
                %c-1640531527_i32_13 = arith.constant -1640531527 : i32
                %c-845247145_i32 = arith.constant -845247145 : i32
                %c-766435501_i32 = arith.constant -766435501 : i32
                %c16383_i32 = arith.constant 16383 : i32
                %c1073741824_i32_14 = arith.constant 1073741824 : i32
                %c4_i32_15 = arith.constant 4 : i32
                %c0_i64_16 = arith.constant 0 : i64
                %c0_i32_17 = arith.constant 0 : i32
                %c256_i32 = arith.constant 256 : i32
                %c4294967296_i64_18 = arith.constant 4294967296 : i64
                %285:4 = scf.for %arg20 = %c0_i32_17 to %arg15 step %c1_i32_9 iter_args(%arg21 = %arg16, %arg22 = %arg17, %arg23 = %arg18, %arg24 = %arg19) -> (i32, i32, i32, i32)  : i32 {
                  %286 = arith.extui %arg21 : i32 to i64
                  %287 = arith.extui %arg22 : i32 to i64
                  %288 = arith.muli %287, %c4294967296_i64_18 : i64
                  %289 = arith.addi %286, %288 : i64
                  %290 = arith.extui %arg23 : i32 to i64
                  %291 = arith.extui %arg24 : i32 to i64
                  %292 = arith.muli %291, %c4294967296_i64_18 : i64
                  %293 = arith.addi %290, %292 : i64
                  %294 = arith.muli %arg20, %c256_i32 : i32
                  %295 = ascvec.create_mask : !ascvec.mask
                  %296 = ascvec.arange %c0_i32_17 : i32 -> <ui32>
                  %297 = arith.muli %289, %arg8 : i64
                  %298 = arith.addi %297, %293 : i64
                  %299 = arith.trunci %298 : i64 to i32
                  %300 = arith.extui %299 : i32 to i64
                  %301 = arith.subi %298, %300 : i64
                  %302 = arith.remsi %301, %c4294967296_i64_18 : i64
                  %303 = arith.cmpi ne, %302, %c0_i64_16 : i64
                  %304 = arith.cmpi slt, %302, %c0_i64_16 : i64
                  %305 = arith.andi %303, %304 : i1
                  %306 = arith.divsi %301, %c4294967296_i64_18 : i64
                  %307 = arith.extui %305 : i1 to i64
                  %308 = arith.subi %306, %307 : i64
                  %309 = arith.trunci %308 : i64 to i32
                  %310 = ascvec.duplicate_scalar %299, %295 : i32 -> <ui32>
                  %311 = ascvec.duplicate_scalar %309, %295 : i32 -> <ui32>
                  %312 = ascvec.duplicate_scalar %c0_i32_17, %295 : i32 -> <ui32>
                  %carry, %dst = ascvec.addc %310, %296, %295 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %carry_19, %dst_20 = ascvec.addc %311, %312, %carry, %295 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %313 = arith.trunci %arg9 : i64 to i32
                  %314 = arith.extui %313 : i32 to i64
                  %315 = arith.subi %arg9, %314 : i64
                  %316 = arith.remsi %315, %c4294967296_i64_18 : i64
                  %317 = arith.cmpi ne, %316, %c0_i64_16 : i64
                  %318 = arith.cmpi slt, %316, %c0_i64_16 : i64
                  %319 = arith.andi %317, %318 : i1
                  %320 = arith.divsi %315, %c4294967296_i64_18 : i64
                  %321 = arith.extui %319 : i1 to i64
                  %322 = arith.subi %320, %321 : i64
                  %323 = arith.trunci %322 : i64 to i32
                  %324 = arith.divui %313, %c4_i32_15 : i32
                  %325 = arith.remui %323, %c4_i32_15 : i32
                  %326 = arith.muli %325, %c1073741824_i32_14 : i32
                  %327 = arith.addi %324, %326 : i32
                  %328 = arith.divui %323, %c4_i32_15 : i32
                  %329 = ascvec.shr %dst, 14, %295 : <ui32> -> <ui32>
                  %330 = ascvec.shl %dst_20, 18, %295 : <ui32> -> <ui32>
                  %331 = ascvec.or %329, %330, %295 : <ui32>, <ui32> -> <ui32>
                  %332 = ascvec.shr %dst_20, 14, %295 : <ui32> -> <ui32>
                  %333 = ascvec.duplicate_scalar %327, %295 : i32 -> <ui32>
                  %334 = ascvec.duplicate_scalar %328, %295 : i32 -> <ui32>
                  %carry_21, %dst_22 = ascvec.addc %333, %331, %295 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %carry_23, %dst_24 = ascvec.addc %334, %332, %carry_21, %295 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %335 = ascvec.duplicate_scalar %c16383_i32, %295 : i32 -> <ui32>
                  %336 = ascvec.and %dst, %335, %295 : <ui32>, <ui32> -> <ui32>
                  %337 = ascvec.duplicate_scalar %c-766435501_i32, %295 : i32 -> <ui32>
                  %338 = ascvec.duplicate_scalar %c-845247145_i32, %295 : i32 -> <ui32>
                  %dst0, %dst1 = ascvec.mull %312, %337, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_25, %dst1_26 = ascvec.mull %dst_22, %338, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %339 = ascvec.xor %dst1_26, %336, %295 : <ui32>, <ui32> -> <ui32>
                  %340 = ascvec.xor %dst1, %dst_24, %295 : <ui32>, <ui32> -> <ui32>
                  %341 = ascvec.duplicate_scalar %arg10, %295 : i32 -> <ui32>
                  %342 = ascvec.duplicate_scalar %arg11, %295 : i32 -> <ui32>
                  %343 = ascvec.xor %339, %341, %295 : <ui32>, <ui32> -> <ui32>
                  %344 = ascvec.xor %340, %342, %295 : <ui32>, <ui32> -> <ui32>
                  %dst0_27, %dst1_28 = ascvec.mull %343, %337, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_29, %dst1_30 = ascvec.mull %344, %338, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %345 = ascvec.xor %dst1_30, %dst0_25, %295 : <ui32>, <ui32> -> <ui32>
                  %346 = ascvec.xor %dst1_28, %dst0, %295 : <ui32>, <ui32> -> <ui32>
                  %347 = arith.addi %arg10, %c-1640531527_i32_13 : i32
                  %348 = ascvec.duplicate_scalar %347, %295 : i32 -> <ui32>
                  %349 = arith.addi %arg11, %c-1150833019_i32_12 : i32
                  %350 = ascvec.duplicate_scalar %349, %295 : i32 -> <ui32>
                  %351 = ascvec.xor %345, %348, %295 : <ui32>, <ui32> -> <ui32>
                  %352 = ascvec.xor %346, %350, %295 : <ui32>, <ui32> -> <ui32>
                  %dst0_31, %dst1_32 = ascvec.mull %351, %337, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_33, %dst1_34 = ascvec.mull %352, %338, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %353 = ascvec.xor %dst1_34, %dst0_29, %295 : <ui32>, <ui32> -> <ui32>
                  %354 = ascvec.xor %dst1_32, %dst0_27, %295 : <ui32>, <ui32> -> <ui32>
                  %355 = arith.addi %arg10, %c1013904242_i32_11 : i32
                  %356 = ascvec.duplicate_scalar %355, %295 : i32 -> <ui32>
                  %357 = arith.addi %arg11, %c1993301258_i32_10 : i32
                  %358 = ascvec.duplicate_scalar %357, %295 : i32 -> <ui32>
                  %359 = ascvec.xor %353, %356, %295 : <ui32>, <ui32> -> <ui32>
                  %360 = ascvec.xor %354, %358, %295 : <ui32>, <ui32> -> <ui32>
                  %dst0_35, %dst1_36 = ascvec.mull %359, %337, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_37, %dst1_38 = ascvec.mull %360, %338, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %361 = ascvec.xor %dst1_38, %dst0_33, %295 : <ui32>, <ui32> -> <ui32>
                  %362 = ascvec.xor %dst1_36, %dst0_31, %295 : <ui32>, <ui32> -> <ui32>
                  %363 = arith.addi %arg10, %c-626627285_i32 : i32
                  %364 = ascvec.duplicate_scalar %363, %295 : i32 -> <ui32>
                  %365 = arith.addi %arg11, %c842468239_i32 : i32
                  %366 = ascvec.duplicate_scalar %365, %295 : i32 -> <ui32>
                  %367 = ascvec.xor %361, %364, %295 : <ui32>, <ui32> -> <ui32>
                  %368 = ascvec.xor %362, %366, %295 : <ui32>, <ui32> -> <ui32>
                  %dst0_39, %dst1_40 = ascvec.mull %367, %337, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_41, %dst1_42 = ascvec.mull %368, %338, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %369 = ascvec.xor %dst1_42, %dst0_37, %295 : <ui32>, <ui32> -> <ui32>
                  %370 = ascvec.xor %dst1_40, %dst0_35, %295 : <ui32>, <ui32> -> <ui32>
                  %371 = arith.addi %arg10, %c2027808484_i32 : i32
                  %372 = ascvec.duplicate_scalar %371, %295 : i32 -> <ui32>
                  %373 = arith.addi %arg11, %c-308364780_i32 : i32
                  %374 = ascvec.duplicate_scalar %373, %295 : i32 -> <ui32>
                  %375 = ascvec.xor %369, %372, %295 : <ui32>, <ui32> -> <ui32>
                  %376 = ascvec.xor %370, %374, %295 : <ui32>, <ui32> -> <ui32>
                  %dst0_43, %dst1_44 = ascvec.mull %375, %337, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_45, %dst1_46 = ascvec.mull %376, %338, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %377 = ascvec.xor %dst1_46, %dst0_41, %295 : <ui32>, <ui32> -> <ui32>
                  %378 = ascvec.xor %dst1_44, %dst0_39, %295 : <ui32>, <ui32> -> <ui32>
                  %379 = arith.addi %arg10, %c387276957_i32 : i32
                  %380 = ascvec.duplicate_scalar %379, %295 : i32 -> <ui32>
                  %381 = arith.addi %arg11, %c-1459197799_i32 : i32
                  %382 = ascvec.duplicate_scalar %381, %295 : i32 -> <ui32>
                  %383 = ascvec.xor %377, %380, %295 : <ui32>, <ui32> -> <ui32>
                  %384 = ascvec.xor %378, %382, %295 : <ui32>, <ui32> -> <ui32>
                  %dst0_47, %dst1_48 = ascvec.mull %383, %337, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_49, %dst1_50 = ascvec.mull %384, %338, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %385 = ascvec.xor %dst1_50, %dst0_45, %295 : <ui32>, <ui32> -> <ui32>
                  %386 = ascvec.xor %dst1_48, %dst0_43, %295 : <ui32>, <ui32> -> <ui32>
                  %387 = arith.addi %arg10, %c-1253254570_i32 : i32
                  %388 = ascvec.duplicate_scalar %387, %295 : i32 -> <ui32>
                  %389 = arith.addi %arg11, %c1684936478_i32 : i32
                  %390 = ascvec.duplicate_scalar %389, %295 : i32 -> <ui32>
                  %391 = ascvec.xor %385, %388, %295 : <ui32>, <ui32> -> <ui32>
                  %392 = ascvec.xor %386, %390, %295 : <ui32>, <ui32> -> <ui32>
                  %dst0_51, %dst1_52 = ascvec.mull %391, %337, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_53, %dst1_54 = ascvec.mull %392, %338, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %393 = ascvec.xor %dst1_54, %dst0_49, %295 : <ui32>, <ui32> -> <ui32>
                  %394 = ascvec.xor %dst1_52, %dst0_47, %295 : <ui32>, <ui32> -> <ui32>
                  %395 = arith.addi %arg10, %c1401181199_i32 : i32
                  %396 = ascvec.duplicate_scalar %395, %295 : i32 -> <ui32>
                  %397 = arith.addi %arg11, %c534103459_i32 : i32
                  %398 = ascvec.duplicate_scalar %397, %295 : i32 -> <ui32>
                  %399 = ascvec.xor %393, %396, %295 : <ui32>, <ui32> -> <ui32>
                  %400 = ascvec.xor %394, %398, %295 : <ui32>, <ui32> -> <ui32>
                  %dst0_55, %dst1_56 = ascvec.mull %399, %337, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_57, %dst1_58 = ascvec.mull %400, %338, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %401 = ascvec.xor %dst1_58, %dst0_53, %295 : <ui32>, <ui32> -> <ui32>
                  %402 = ascvec.xor %dst1_56, %dst0_51, %295 : <ui32>, <ui32> -> <ui32>
                  %403 = arith.addi %arg10, %c-239350328_i32 : i32
                  %404 = ascvec.duplicate_scalar %403, %295 : i32 -> <ui32>
                  %405 = arith.addi %arg11, %c-616729560_i32 : i32
                  %406 = ascvec.duplicate_scalar %405, %295 : i32 -> <ui32>
                  %407 = ascvec.xor %401, %404, %295 : <ui32>, <ui32> -> <ui32>
                  %408 = ascvec.xor %402, %406, %295 : <ui32>, <ui32> -> <ui32>
                  %dst0_59, %dst1_60 = ascvec.mull %407, %337, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_61, %dst1_62 = ascvec.mull %408, %338, %295 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %409 = ascvec.xor %dst1_62, %dst0_57, %295 : <ui32>, <ui32> -> <ui32>
                  %410 = ascvec.xor %dst1_60, %dst0_55, %295 : <ui32>, <ui32> -> <ui32>
                  %411 = arith.addi %arg10, %c-1879881855_i32 : i32
                  %412 = ascvec.duplicate_scalar %411, %295 : i32 -> <ui32>
                  %413 = arith.addi %arg11, %c-1767562579_i32 : i32
                  %414 = ascvec.duplicate_scalar %413, %295 : i32 -> <ui32>
                  %415 = ascvec.xor %409, %412, %295 : <ui32>, <ui32> -> <ui32>
                  %416 = ascvec.xor %410, %414, %295 : <ui32>, <ui32> -> <ui32>
                  %417 = ascvec.reinterpret %415 : <ui32> -> <i32>
                  %418 = arith.index_cast %294 : i32 to index
                  ascvec.storealign %arg12[%418], %417, %295 : memref<8192xi32, 1>, index, <i32>
                  %419 = arith.addi %294, %c64_i32 : i32
                  %420 = ascvec.reinterpret %dst0_61 : <ui32> -> <i32>
                  %421 = arith.index_cast %419 : i32 to index
                  ascvec.storealign %arg12[%421], %420, %295 : memref<8192xi32, 1>, index, <i32>
                  %422 = arith.addi %294, %c128_i32 : i32
                  %423 = ascvec.reinterpret %416 : <ui32> -> <i32>
                  %424 = arith.index_cast %422 : i32 to index
                  ascvec.storealign %arg12[%424], %423, %295 : memref<8192xi32, 1>, index, <i32>
                  %425 = arith.addi %294, %c192_i32 : i32
                  %426 = ascvec.reinterpret %dst0_59 : <ui32> -> <i32>
                  %427 = arith.index_cast %425 : i32 to index
                  ascvec.storealign %arg12[%427], %426, %295 : memref<8192xi32, 1>, index, <i32>
                  %428 = arith.addi %293, %arg13 : i64
                  %429 = arith.cmpi sge, %428, %arg8 : i64
                  %430 = arith.extui %429 : i1 to i64
                  %431 = arith.addi %289, %arg14 : i64
                  %432 = arith.addi %431, %430 : i64
                  %433 = arith.muli %430, %arg8 : i64
                  %434 = arith.subi %428, %433 : i64
                  %435 = arith.trunci %432 : i64 to i32
                  %436 = arith.extui %435 : i32 to i64
                  %437 = arith.subi %432, %436 : i64
                  %438 = arith.remsi %437, %c4294967296_i64_18 : i64
                  %439 = arith.cmpi ne, %438, %c0_i64_16 : i64
                  %440 = arith.cmpi slt, %438, %c0_i64_16 : i64
                  %441 = arith.andi %439, %440 : i1
                  %442 = arith.divsi %437, %c4294967296_i64_18 : i64
                  %443 = arith.extui %441 : i1 to i64
                  %444 = arith.subi %442, %443 : i64
                  %445 = arith.trunci %444 : i64 to i32
                  %446 = arith.trunci %434 : i64 to i32
                  %447 = arith.extui %446 : i32 to i64
                  %448 = arith.subi %434, %447 : i64
                  %449 = arith.remsi %448, %c4294967296_i64_18 : i64
                  %450 = arith.cmpi ne, %449, %c0_i64_16 : i64
                  %451 = arith.cmpi slt, %449, %c0_i64_16 : i64
                  %452 = arith.andi %450, %451 : i1
                  %453 = arith.divsi %448, %c4294967296_i64_18 : i64
                  %454 = arith.extui %452 : i1 to i64
                  %455 = arith.subi %453, %454 : i64
                  %456 = arith.trunci %455 : i64 to i32
                  scf.yield %435, %445, %446, %456 : i32, i32, i32, i32
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              %280 = arith.trunci %202 : i64 to i32
              %281 = arith.trunci %62 : i64 to i32
              %282 = arith.trunci %44 : i64 to i32
              %283 = arith.muli %280, %c2_i32 : i32
              %284 = arith.cmpi eq, %282, %c4_i32 : i32
              ascvec.get_buf v, %68, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%62, %103, %alloc_7, %alloc_4, %alloc_6, %280, %281, %283, %284, %alloc_5 : i64, memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, memref<1024xui32, 1>, i32, i32, i32, i1, memref<2048xui8, 1>) {
              ^bb0(%arg8: i64, %arg9: memref<32768xbf16, 1>, %arg10: memref<8192xi32, 1>, %arg11: memref<32768xui8, 1>, %arg12: memref<1024xui32, 1>, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i1, %arg17: memref<2048xui8, 1>):
                %c2_i32_9 = arith.constant 2 : i32
                %c2139095040_i32 = arith.constant 2139095040 : i32
                %c127_i32 = arith.constant 127 : i32
                %c123_i32 = arith.constant 123 : i32
                %c947912704_i32 = arith.constant 947912704 : i32
                %cst = arith.constant 4.2949673E+9 : f32
                %c29360128_i32 = arith.constant 29360128 : i32
                %c128_i32 = arith.constant 128 : i32
                %c65535_i32 = arith.constant 65535 : i32
                %c4_i32_10 = arith.constant 4 : i32
                %c256_i32 = arith.constant 256 : i32
                %c1_i32_11 = arith.constant 1 : i32
                %c16_i32 = arith.constant 16 : i32
                %c3_i32 = arith.constant 3 : i32
                %c2139160576_i32 = arith.constant 2139160576 : i32
                %c254_i32 = arith.constant 254 : i32
                %c-15_i32 = arith.constant -15 : i32
                %c15_i32 = arith.constant 15 : i32
                %c255_i32 = arith.constant 255 : i32
                %c2147483647_i32 = arith.constant 2147483647 : i32
                %cst_12 = arith.constant 0.000000e+00 : f32
                %c0_13 = arith.constant 0 : index
                %c64_i32 = arith.constant 64 : i32
                %c32_i32 = arith.constant 32 : i32
                %c0_i32_14 = arith.constant 0 : i32
                %285 = ascvec.create_mask : !ascvec.mask
                %286 = ascvec.arange %c0_i32_14 : i32 -> <ui32>
                %287 = ascvec.cmp_lt_scalar %286, %c32_i32, %285 : <ui32>, i32 -> !ascvec.mask
                %288 = ascvec.cmp_ge_scalar %286, %c32_i32, %285 : <ui32>, i32 -> !ascvec.mask
                scf.for %arg18 = %c0_i32_14 to %arg13 step %c1_i32_11  : i32 {
                  %290 = arith.muli %arg18, %c64_i32 : i32
                  %291 = arith.trunci %arg8 : i64 to i32
                  %292 = arith.subi %291, %290 : i32
                  %mask, %remaining_out = ascvec.update_mask %292 : !ascvec.mask
                  %293 = arith.index_cast %290 : i32 to index
                  %294 = ascvec.loadalign_unpack %arg9[%293] {mode = "b16_to_b32"} : memref<32768xbf16, 1>, index -> <bf16>
                  %295 = ascvec.cast %294, %285 {rounding = 6 : i32} : <bf16> -> <f32>
                  %296 = ascvec.duplicate_scalar %cst_12, %285 : f32 -> <f32>
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
                  %313 = ascvec.cmp_ne_scalar %310, %c0_i32_14, %285 : <ui32>, i32 -> !ascvec.mask
                  %314 = ascvec.shl %312, 23, %313 : <ui32> -> <ui32>
                  %315 = ascvec.duplicate_scalar %c2139160576_i32, %307 merge %314 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %316 = ascvec.reinterpret %315 : <ui32> -> <f32>
                  %317 = ascvec.mul %297, %316, %285 : <f32>
                  %318 = ascvec.shr %286, 2, %285 : <ui32> -> <ui32>
                  %319 = ascvec.duplicate_scalar %c3_i32, %285 : i32 -> <ui32>
                  %320 = ascvec.and %318, %319, %285 : <ui32>, <ui32> -> <ui32>
                  %321 = ascvec.shl %320, 6, %285 : <ui32> -> <ui32>
                  %322 = ascvec.shr %286, 4, %285 : <ui32> -> <ui32>
                  %323 = ascvec.add %321, %322, %285 : <ui32>
                  %324 = arith.remsi %arg18, %c16_i32 : i32
                  %325 = arith.cmpi ne, %324, %c0_i32_14 : i32
                  %326 = arith.cmpi slt, %324, %c0_i32_14 : i32
                  %327 = arith.andi %325, %326 : i1
                  %328 = arith.divsi %arg18, %c16_i32 : i32
                  %329 = arith.extui %327 : i1 to i32
                  %330 = arith.subi %328, %329 : i32
                  %331 = arith.muli %330, %c256_i32 : i32
                  %332 = arith.select %327, %c16_i32, %c0_i32_14 : i32
                  %333 = arith.addi %324, %332 overflow<nsw> : i32
                  %334 = arith.muli %333, %c4_i32_10 overflow<nsw> : i32
                  %335 = arith.addi %331, %334 : i32
                  %336 = ascvec.add_scalar %323, %335, %285 : <ui32>, i32
                  %337 = ascvec.gather %arg10[%c0_13], %336, %285 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                  %338 = ascvec.reinterpret %337 : <i32> -> <ui32>
                  %339 = ascvec.shl %338, 8, %285 : <ui32> -> <ui32>
                  %340 = ascvec.shr %338, 24, %285 : <ui32> -> <ui32>
                  %341 = ascvec.or %339, %340, %285 : <ui32>, <ui32> -> <ui32>
                  %342 = ascvec.and %286, %319, %285 : <ui32>, <ui32> -> <ui32>
                  %343 = ascvec.duplicate_scalar %c1_i32_11, %285 : i32 -> <ui32>
                  %344 = ascvec.and %342, %343, %285 : <ui32>, <ui32> -> <ui32>
                  %345 = ascvec.cmp_eq_scalar %344, %c0_i32_14, %285 : <ui32>, i32 -> !ascvec.mask
                  %346 = ascvec.select %338, %341, %345 : <ui32>
                  %347 = ascvec.cmp_eq_scalar %342, %c0_i32_14, %285 : <ui32>, i32 -> !ascvec.mask
                  %348 = ascvec.cmp_eq_scalar %342, %c3_i32, %285 : <ui32>, i32 -> !ascvec.mask
                  %349 = ascvec.mask_or %347, %348, %285
                  %350 = ascvec.shr %346, 16, %285 : <ui32> -> <ui32>
                  %351 = ascvec.duplicate_scalar %c65535_i32, %285 : i32 -> <ui32>
                  %352 = ascvec.and %346, %351, %285 : <ui32>, <ui32> -> <ui32>
                  %353 = ascvec.select %350, %352, %349 : <ui32>
                  %354 = ascvec.reinterpret %317 : <f32> -> <ui32>
                  %355 = ascvec.shr %354, 24, %285 : <ui32> -> <ui32>
                  %356 = ascvec.duplicate_scalar %c128_i32, %285 : i32 -> <ui32>
                  %357 = ascvec.and %355, %356, %285 : <ui32>, <ui32> -> <ui32>
                  %358 = ascvec.and %354, %299, %285 : <ui32>, <ui32> -> <ui32>
                  %359 = ascvec.shr %358, 5, %285 : <ui32> -> <ui32>
                  %360 = ascvec.duplicate_scalar %c29360128_i32, %285 : i32 -> <ui32>
                  %361 = ascvec.sub %359, %360, %285 : <ui32>
                  %362 = ascvec.reinterpret %358 : <ui32> -> <f32>
                  %363 = ascvec.mul_scalar %362, %cst, %285 : <f32>, f32
                  %364 = ascvec.cast %363, %285 {rounding = 4 : i32} : <f32> -> <i32>
                  %365 = ascvec.reinterpret %364 : <i32> -> <ui32>
                  %366 = ascvec.cmp_ge_scalar %358, %c947912704_i32, %285 : <ui32>, i32 -> !ascvec.mask
                  %367 = ascvec.select %361, %365, %366 : <ui32>
                  %368 = ascvec.and %353, %351, %285 : <ui32>, <ui32> -> <ui32>
                  %369 = ascvec.add %367, %368, %285 : <ui32>
                  %370 = ascvec.shr %369, 16, %285 : <ui32> -> <ui32>
                  %371 = ascvec.min_scalar %370, %c123_i32, %285 : <ui32>, i32
                  %372 = ascvec.or %371, %357, %285 : <ui32>, <ui32> -> <ui32>
                  %373 = ascvec.duplicate_scalar %c127_i32, %285 : i32 -> <ui32>
                  %374 = ascvec.cmp_gt_scalar %358, %c2139095040_i32, %285 : <ui32>, i32 -> !ascvec.mask
                  %375 = ascvec.select %373, %372, %374 : <ui32>
                  ascvec.storealign_pack %arg11[%293], %375, %mask {mode = "b32_to_b8"} : memref<32768xui8, 1>, index, <ui32>
                  %376 = arith.muli %arg18, %c2_i32_9 : i32
                  %377 = arith.index_cast %376 : i32 to index
                  ascvec.storealign_1st %arg12[%377], %310 : memref<1024xui32, 1>, index, <ui32>
                  %378 = ascvec.duplicate_scalar %c32_i32, %285 : i32 -> <ui32>
                  %379 = ascvec.gather_reg %310, %378 : <ui32>, <ui32> -> <ui32>
                  %380 = arith.addi %376, %c1_i32_11 : i32
                  %381 = arith.index_cast %380 : i32 to index
                  ascvec.storealign_1st %arg12[%381], %379 : memref<1024xui32, 1>, index, <ui32>
                }
                ascvec.mem_bar vst_vld
                %289 = scf.if %arg16 -> (i32) {
                  %290 = arith.remsi %arg14, %c32_i32 : i32
                  %291 = arith.cmpi ne, %290, %c0_i32_14 : i32
                  %292 = arith.cmpi slt, %290, %c0_i32_14 : i32
                  %293 = arith.andi %291, %292 : i1
                  %294 = arith.divsi %arg14, %c32_i32 : i32
                  %295 = arith.extui %293 : i1 to i32
                  %296 = arith.subi %294, %295 : i32
                  %297 = arith.muli %296, %c2_i32_9 : i32
                  scf.yield %297 : i32
                } else {
                  scf.yield %arg15 : i32
                }
                scf.for %arg18 = %c0_i32_14 to %289 step %c64_i32  : i32 {
                  %290 = arith.subi %289, %arg18 : i32
                  %mask, %remaining_out = ascvec.update_mask %290 : !ascvec.mask
                  %291 = ascvec.add_scalar %286, %arg18, %285 : <ui32>, i32
                  scf.if %arg16 {
                    %292 = ascvec.shr %291, 1, %285 : <ui32> -> <ui32>
                    %293 = ascvec.gather %arg12[%c0_13], %292, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %294 = ascvec.duplicate_scalar %c1_i32_11, %285 : i32 -> <ui32>
                    %295 = ascvec.and %286, %294, %285 : <ui32>, <ui32> -> <ui32>
                    %296 = ascvec.cmp_eq_scalar %295, %c0_i32_14, %285 : <ui32>, i32 -> !ascvec.mask
                    %297 = ascvec.duplicate_scalar %c0_i32_14, %285 : i32 -> <ui32>
                    %298 = ascvec.select %293, %297, %296 : <ui32>
                    %299 = arith.index_cast %arg18 : i32 to index
                    %c0_15 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg19 = %c0_15 to %c1 step %c1 {
                      ascvec.storealign_pack %arg17[%299], %298, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  } else {
                    %292 = ascvec.gather %arg12[%c0_13], %291, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %293 = arith.index_cast %arg18 : i32 to index
                    %c0_15 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg19 = %c0_15 to %c1 step %c1 {
                      ascvec.storealign_pack %arg17[%293], %292, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  }
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %68, 0 {cannir.buffer.generated_sync}
            }
          }
          %109 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %110 = cannir.extract_shape %109 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
          %111 = cannir.extract_dim %110[0] : !cannir.shape<(-1, -1)>
          %112 = arith.minsi %111, %c0_i64 : i64
          %113 = arith.subi %111, %112 : i64
          %114 = arith.minsi %113, %c1_i64 : i64
          %115 = arith.cmpi sge, %111, %c0_i64 : i64
          %116 = arith.select %115, %114, %c0_i64 : i64
          %117 = arith.cmpi sgt, %116, %c1_i64 : i64
          %118 = arith.select %117, %c1_i64, %116 : i64
          %119 = cannir.extract_dim %110[1] : !cannir.shape<(-1, -1)>
          %120 = arith.divsi %119, %80 : i64
          %121 = arith.minsi %31, %120 : i64
          %122 = arith.muli %121, %30 : i64
          %123 = arith.subi %119, %122 : i64
          %124 = arith.minsi %30, %123 : i64
          %125 = arith.cmpi sle, %31, %120 : i64
          %126 = arith.select %125, %124, %c0_i64 : i64
          %127 = arith.divsi %126, %c32768_i64 : i64
          %128 = arith.minsi %arg6, %127 : i64
          %129 = arith.muli %128, %c32768_i64 : i64
          %130 = arith.subi %126, %129 : i64
          %131 = arith.minsi %130, %c32768_i64 : i64
          %132 = arith.cmpi sle, %arg6, %127 : i64
          %133 = arith.select %132, %131, %c0_i64 : i64
          %134 = arith.cmpi sgt, %133, %c32768_i64 : i64
          %135 = arith.select %134, %c32768_i64, %133 : i64
          %136 = arith.muli %104, %c8_i64 : i64
          %137 = ascscalar.memref_address %alloc_0 : memref<1xui8>
          %138 = arith.divui %136, %c8_i64 : i64
          %139 = arith.addi %137, %138 : i64
          %140 = ascscalar.bind_gm_ptr %139 : memref<1xui8>
          %141 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<32768xui8, 1> -> memref<?xui8, 1>
          %142 = cannir.extract_stride %109 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %143 = cannir.extract_dim %142[0] : !cannir.stride<(-1, 1)>
          %144 = arith.cmpi ugt, %118, %c0_i64 : i64
          %145 = arith.cmpi ugt, %135, %c0_i64 : i64
          %146 = arith.andi %144, %145 : i1
          scf.if %146 {
            ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
            %188 = arith.trunci %118 : i64 to i32
            %189 = arith.trunci %135 : i64 to i32
            ascvec.copy_ub2gm %140[%c0], %141[%c0], %188, %189, %143, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
          }
          %147 = arith.muli %44, %c512_i64 overflow<nsw> : i64
          %148 = cannir.extract_dim %34[0] : !cannir.shape<(-1, -1)>
          %149 = arith.minsi %148, %c0_i64 : i64
          %150 = arith.subi %148, %149 : i64
          %151 = arith.minsi %150, %c1_i64 : i64
          %152 = arith.cmpi sge, %148, %c0_i64 : i64
          %153 = arith.select %152, %151, %c0_i64 : i64
          %154 = arith.cmpi sgt, %153, %c1_i64 : i64
          %155 = arith.select %154, %c1_i64, %153 : i64
          %156 = arith.maxsi %46, %c1_i64 : i64
          %157 = arith.divsi %35, %156 : i64
          %158 = arith.minsi %31, %157 : i64
          %159 = arith.muli %158, %46 : i64
          %160 = arith.subi %35, %159 : i64
          %161 = arith.minsi %46, %160 : i64
          %162 = arith.cmpi sle, %31, %157 : i64
          %163 = arith.select %162, %161, %c0_i64 : i64
          %164 = arith.maxsi %147, %c1_i64 : i64
          %165 = arith.divsi %163, %164 : i64
          %166 = arith.minsi %arg6, %165 : i64
          %167 = arith.muli %166, %147 : i64
          %168 = arith.subi %163, %167 : i64
          %169 = arith.minsi %147, %168 : i64
          %170 = arith.cmpi sle, %arg6, %165 : i64
          %171 = arith.select %170, %169, %c0_i64 : i64
          %172 = arith.cmpi sgt, %171, %c2048_i64 : i64
          %173 = arith.select %172, %c2048_i64, %171 : i64
          %174 = arith.muli %arg6, %147 : i64
          %175 = arith.muli %31, %46 : i64
          %176 = arith.addi %174, %175 : i64
          %177 = arith.muli %176, %c8_i64 : i64
          %178 = ascscalar.memref_address %alloc_1 : memref<1xui8>
          %179 = arith.divui %177, %c8_i64 : i64
          %180 = arith.addi %178, %179 : i64
          %181 = ascscalar.bind_gm_ptr %180 : memref<1xui8>
          %182 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<2048xui8, 1> -> memref<?xui8, 1>
          %183 = cannir.extract_stride %33 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %184 = cannir.extract_dim %183[0] : !cannir.stride<(-1, 1)>
          %185 = arith.cmpi ugt, %155, %c0_i64 : i64
          %186 = arith.cmpi ugt, %173, %c0_i64 : i64
          %187 = arith.andi %185, %186 : i1
          scf.if %187 {
            ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
            %188 = arith.trunci %155 : i64 to i32
            %189 = arith.trunci %173 : i64 to i32
            ascvec.copy_ub2gm %181[%c0], %182[%c0], %188, %189, %184, %c2048_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %70 : i64
        }
      }
    } else {
      %5 = arith.cmpi sgt, %2, %c1_i64 : i64
      %6 = scf.if %5 -> (i1) {
        %7 = cannir.extract_dim %1[1] : !cannir.shape<(-1, -1)>
        %8 = arith.cmpi slt, %7, %c128_i64 : i64
        scf.yield %8 : i1
      } else {
        scf.yield %false : i1
      }
      scf.if %6 {
        %7 = cannir.extract_dim %1[1] : !cannir.shape<(-1, -1)>
        %8 = arith.cmpi sgt, %7, %c64_i64 : i64
        %9 = arith.select %8, %c7_i64, %c6_i64 : i64
        %10 = arith.select %8, %c256_i64, %c512_i64 : i64
        %11 = ascscalar.get_block_num
        %12 = arith.remsi %2, %11 : i64
        %13 = arith.cmpi ne, %12, %c0_i64 : i64
        %14 = arith.cmpi slt, %12, %c0_i64 : i64
        %15 = arith.cmpi slt, %11, %c0_i64 : i64
        %16 = arith.xori %14, %15 : i1
        %17 = arith.andi %13, %16 : i1
        %18 = arith.divsi %2, %11 : i64
        %19 = arith.extui %17 : i1 to i64
        %20 = arith.subi %18, %19 : i64
        %21 = arith.cmpi slt, %20, %c1_i64 : i64
        %22 = arith.select %21, %c1_i64, %20 : i64
        %23 = arith.cmpi sgt, %22, %10 : i64
        %24 = arith.select %23, %10, %22 : i64
        %25 = arith.addi %2, %24 : i64
        %26 = arith.subi %25, %c1_i64 : i64
        %27 = arith.remsi %26, %24 : i64
        %28 = arith.cmpi ne, %27, %c0_i64 : i64
        %29 = arith.cmpi slt, %27, %c0_i64 : i64
        %30 = arith.andi %28, %29 : i1
        %31 = arith.divsi %26, %24 : i64
        %32 = arith.extui %30 : i1 to i64
        %33 = arith.subi %31, %32 : i64
        %34 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
        %35 = cannir.extract_shape %34 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
        %36 = cannir.extract_dim %35[1] : !cannir.shape<(-1, -1)>
        %37 = arith.muli %2, %7 : i64
        %38 = arith.muli %2, %36 : i64
        %39 = ascscalar.get_block_idx
        %40 = scf.for %arg6 = %39 to %33 step %11 iter_args(%arg7 = %c0_i64) -> (i64)  : i64 {
          %41 = arith.muli %arg6, %24 : i64
          %42 = arith.subi %2, %41 : i64
          %43 = arith.cmpi sgt, %42, %24 : i64
          %44 = arith.select %43, %24, %42 : i64
          %45 = arith.cmpi eq, %arg7, %c0_i64 : i64
          %46 = ascscalar.memref_address %alloc_2 : memref<32768xbf16, 1>
          %47 = ascscalar.memref_address %alloc_3 : memref<32768xbf16, 1>
          %48 = arith.select %45, %46, %47 : i64
          %49 = arith.cmpi ne, %arg7, %c0_i64 : i64
          %50 = arith.extui %49 : i1 to i64
          %51 = arith.addi %arg7, %c1_i64 overflow<nsw> : i64
          %52 = arith.remui %51, %c2_i64 : i64
          %53 = arith.muli %24, %7 : i64
          %54 = arith.maxsi %53, %c1_i64 : i64
          %55 = arith.divsi %37, %54 : i64
          %56 = arith.minsi %arg6, %55 : i64
          %57 = arith.muli %56, %53 : i64
          %58 = arith.subi %37, %57 : i64
          %59 = arith.minsi %53, %58 : i64
          %60 = arith.cmpi sle, %arg6, %55 : i64
          %61 = arith.select %60, %59, %c0_i64 : i64
          %62 = arith.muli %arg6, %53 : i64
          %63 = arith.muli %62, %c2_i64 : i64
          %64 = arith.muli %61, %c2_i64 : i64
          %65 = arith.cmpi eq, %64, %c0_i64 : i64
          %66 = arith.cmpi ne, %64, %c0_i64 : i64
          %67 = arith.extui %66 : i1 to i64
          %68 = arith.select %65, %c2_i64, %64 : i64
          %69 = ascscalar.bind_local_ptr %48 : memref<1xbf16, 1>
          ascvec.get_buf mte2, %50, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%69, %alloc, %c0_i32, %c0_i32, %c0_i64, %63, %67, %68, %c0_i64, %68, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %50, 0 {cannir.buffer.generated_sync}
          %70 = ascscalar.bind_local_ptr %48 : memref<32768xbf16, 1>
          %71 = arith.muli %41, %7 : i64
          %72 = arith.cmpi eq, %9, %c7_i64 : i64
          %73 = arith.select %72, %c128_i64, %c64_i64 : i64
          %74 = arith.muli %44, %73 : i64
          %75 = arith.addi %74, %c255_i64 : i64
          %76 = arith.remsi %75, %c256_i64 : i64
          %77 = arith.cmpi ne, %76, %c0_i64 : i64
          %78 = arith.cmpi slt, %76, %c0_i64 : i64
          %79 = arith.andi %77, %78 : i1
          %80 = arith.divsi %75, %c256_i64 : i64
          %81 = arith.extui %79 : i1 to i64
          %82 = arith.subi %80, %81 : i64
          %83 = arith.divsi %73, %c32_i64 : i64
          %84 = arith.muli %44, %83 : i64
          %85 = arith.trunci %9 : i64 to i32
          %86 = arith.divsi %73, %c4_i64 : i64
          %87 = arith.subi %86, %c1_i64 overflow<nsw> : i64
          %88 = arith.trunci %87 : i64 to i32
          %89 = arith.remsi %7, %c4_i64 : i64
          %90 = arith.cmpi ne, %89, %c0_i64 : i64
          %91 = arith.cmpi slt, %89, %c0_i64 : i64
          %92 = arith.andi %90, %91 : i1
          %93 = arith.divsi %7, %c4_i64 : i64
          %94 = arith.extui %92 : i1 to i64
          %95 = arith.subi %93, %94 : i64
          %96 = arith.trunci %95 : i64 to i32
          %97 = arith.trunci %44 : i64 to i32
          %98 = arith.muli %44, %7 : i64
          %99 = arith.addi %98, %c1023_i64 : i64
          %100 = arith.remsi %99, %c1024_i64 : i64
          %101 = arith.cmpi ne, %100, %c0_i64 : i64
          %102 = arith.cmpi slt, %100, %c0_i64 : i64
          %103 = arith.andi %101, %102 : i1
          %104 = arith.divsi %99, %c1024_i64 : i64
          %105 = arith.extui %103 : i1 to i64
          %106 = arith.subi %104, %105 : i64
          %107 = arith.trunci %arg4 : i64 to i32
          %108 = arith.extui %107 : i32 to i64
          %109 = arith.subi %arg4, %108 : i64
          %110 = arith.remsi %109, %c4294967296_i64 : i64
          %111 = arith.cmpi ne, %110, %c0_i64 : i64
          %112 = arith.cmpi slt, %110, %c0_i64 : i64
          %113 = arith.andi %111, %112 : i1
          %114 = arith.divsi %109, %c4294967296_i64 : i64
          %115 = arith.extui %113 : i1 to i64
          %116 = arith.subi %114, %115 : i64
          %117 = arith.trunci %116 : i64 to i32
          %118 = arith.remsi %71, %c16_i64 : i64
          %119 = arith.cmpi ne, %118, %c0_i64 : i64
          %120 = arith.cmpi slt, %118, %c0_i64 : i64
          %121 = arith.andi %119, %120 : i1
          %122 = arith.divsi %71, %c16_i64 : i64
          %123 = arith.extui %121 : i1 to i64
          %124 = arith.subi %122, %123 : i64
          %125 = arith.remsi %7, %c16_i64 : i64
          %126 = arith.cmpi ne, %125, %c0_i64 : i64
          %127 = arith.cmpi slt, %125, %c0_i64 : i64
          %128 = arith.andi %126, %127 : i1
          %129 = arith.divsi %7, %c16_i64 : i64
          %130 = arith.extui %128 : i1 to i64
          %131 = arith.subi %129, %130 : i64
          %132 = arith.remsi %124, %131 : i64
          %133 = arith.cmpi ne, %132, %c0_i64 : i64
          %134 = arith.cmpi slt, %132, %c0_i64 : i64
          %135 = arith.cmpi slt, %131, %c0_i64 : i64
          %136 = arith.xori %134, %135 : i1
          %137 = arith.andi %133, %136 : i1
          %138 = arith.divsi %124, %131 : i64
          %139 = arith.extui %137 : i1 to i64
          %140 = arith.subi %138, %139 : i64
          %141 = arith.select %137, %131, %c0_i64 : i64
          %142 = arith.addi %132, %141 : i64
          %143 = arith.trunci %140 : i64 to i32
          %144 = arith.extui %143 : i32 to i64
          %145 = arith.subi %140, %144 : i64
          %146 = arith.remsi %145, %c4294967296_i64 : i64
          %147 = arith.cmpi ne, %146, %c0_i64 : i64
          %148 = arith.cmpi slt, %146, %c0_i64 : i64
          %149 = arith.andi %147, %148 : i1
          %150 = arith.divsi %145, %c4294967296_i64 : i64
          %151 = arith.extui %149 : i1 to i64
          %152 = arith.subi %150, %151 : i64
          %153 = arith.trunci %152 : i64 to i32
          %154 = arith.trunci %142 : i64 to i32
          %155 = arith.extui %154 : i32 to i64
          %156 = arith.subi %142, %155 : i64
          %157 = arith.remsi %156, %c4294967296_i64 : i64
          %158 = arith.cmpi ne, %157, %c0_i64 : i64
          %159 = arith.cmpi slt, %157, %c0_i64 : i64
          %160 = arith.andi %158, %159 : i1
          %161 = arith.divsi %156, %c4294967296_i64 : i64
          %162 = arith.extui %160 : i1 to i64
          %163 = arith.subi %161, %162 : i64
          %164 = arith.trunci %163 : i64 to i32
          %165 = arith.remsi %c64_i64, %131 : i64
          %166 = arith.cmpi ne, %165, %c0_i64 : i64
          %167 = arith.cmpi slt, %165, %c0_i64 : i64
          %168 = arith.xori %167, %135 : i1
          %169 = arith.andi %166, %168 : i1
          %170 = arith.divsi %c64_i64, %131 : i64
          %171 = arith.extui %169 : i1 to i64
          %172 = arith.subi %170, %171 : i64
          %173 = arith.select %169, %131, %c0_i64 : i64
          %174 = arith.addi %165, %173 : i64
          %175 = arith.trunci %106 : i64 to i32
          ascvec.vf(%131, %arg5, %107, %117, %alloc_7, %174, %172, %175, %143, %153, %154, %164 : i64, i64, i32, i32, memref<8192xi32, 1>, i64, i64, i32, i32, i32, i32, i32) {
          ^bb0(%arg8: i64, %arg9: i64, %arg10: i32, %arg11: i32, %arg12: memref<8192xi32, 1>, %arg13: i64, %arg14: i64, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32):
            %c1_i32_9 = arith.constant 1 : i32
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
            %c1993301258_i32_10 = arith.constant 1993301258 : i32
            %c1013904242_i32_11 = arith.constant 1013904242 : i32
            %c-1150833019_i32_12 = arith.constant -1150833019 : i32
            %c-1640531527_i32_13 = arith.constant -1640531527 : i32
            %c-845247145_i32 = arith.constant -845247145 : i32
            %c-766435501_i32 = arith.constant -766435501 : i32
            %c16383_i32 = arith.constant 16383 : i32
            %c1073741824_i32_14 = arith.constant 1073741824 : i32
            %c4_i32_15 = arith.constant 4 : i32
            %c0_i64_16 = arith.constant 0 : i64
            %c0_i32_17 = arith.constant 0 : i32
            %c256_i32 = arith.constant 256 : i32
            %c4294967296_i64_18 = arith.constant 4294967296 : i64
            %208:4 = scf.for %arg20 = %c0_i32_17 to %arg15 step %c1_i32_9 iter_args(%arg21 = %arg16, %arg22 = %arg17, %arg23 = %arg18, %arg24 = %arg19) -> (i32, i32, i32, i32)  : i32 {
              %209 = arith.extui %arg21 : i32 to i64
              %210 = arith.extui %arg22 : i32 to i64
              %211 = arith.muli %210, %c4294967296_i64_18 : i64
              %212 = arith.addi %209, %211 : i64
              %213 = arith.extui %arg23 : i32 to i64
              %214 = arith.extui %arg24 : i32 to i64
              %215 = arith.muli %214, %c4294967296_i64_18 : i64
              %216 = arith.addi %213, %215 : i64
              %217 = arith.muli %arg20, %c256_i32 : i32
              %218 = ascvec.create_mask : !ascvec.mask
              %219 = ascvec.arange %c0_i32_17 : i32 -> <ui32>
              %220 = arith.muli %212, %arg8 : i64
              %221 = arith.addi %220, %216 : i64
              %222 = arith.trunci %221 : i64 to i32
              %223 = arith.extui %222 : i32 to i64
              %224 = arith.subi %221, %223 : i64
              %225 = arith.remsi %224, %c4294967296_i64_18 : i64
              %226 = arith.cmpi ne, %225, %c0_i64_16 : i64
              %227 = arith.cmpi slt, %225, %c0_i64_16 : i64
              %228 = arith.andi %226, %227 : i1
              %229 = arith.divsi %224, %c4294967296_i64_18 : i64
              %230 = arith.extui %228 : i1 to i64
              %231 = arith.subi %229, %230 : i64
              %232 = arith.trunci %231 : i64 to i32
              %233 = ascvec.duplicate_scalar %222, %218 : i32 -> <ui32>
              %234 = ascvec.duplicate_scalar %232, %218 : i32 -> <ui32>
              %235 = ascvec.duplicate_scalar %c0_i32_17, %218 : i32 -> <ui32>
              %carry, %dst = ascvec.addc %233, %219, %218 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
              %carry_19, %dst_20 = ascvec.addc %234, %235, %carry, %218 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
              %236 = arith.trunci %arg9 : i64 to i32
              %237 = arith.extui %236 : i32 to i64
              %238 = arith.subi %arg9, %237 : i64
              %239 = arith.remsi %238, %c4294967296_i64_18 : i64
              %240 = arith.cmpi ne, %239, %c0_i64_16 : i64
              %241 = arith.cmpi slt, %239, %c0_i64_16 : i64
              %242 = arith.andi %240, %241 : i1
              %243 = arith.divsi %238, %c4294967296_i64_18 : i64
              %244 = arith.extui %242 : i1 to i64
              %245 = arith.subi %243, %244 : i64
              %246 = arith.trunci %245 : i64 to i32
              %247 = arith.divui %236, %c4_i32_15 : i32
              %248 = arith.remui %246, %c4_i32_15 : i32
              %249 = arith.muli %248, %c1073741824_i32_14 : i32
              %250 = arith.addi %247, %249 : i32
              %251 = arith.divui %246, %c4_i32_15 : i32
              %252 = ascvec.shr %dst, 14, %218 : <ui32> -> <ui32>
              %253 = ascvec.shl %dst_20, 18, %218 : <ui32> -> <ui32>
              %254 = ascvec.or %252, %253, %218 : <ui32>, <ui32> -> <ui32>
              %255 = ascvec.shr %dst_20, 14, %218 : <ui32> -> <ui32>
              %256 = ascvec.duplicate_scalar %250, %218 : i32 -> <ui32>
              %257 = ascvec.duplicate_scalar %251, %218 : i32 -> <ui32>
              %carry_21, %dst_22 = ascvec.addc %256, %254, %218 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
              %carry_23, %dst_24 = ascvec.addc %257, %255, %carry_21, %218 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
              %258 = ascvec.duplicate_scalar %c16383_i32, %218 : i32 -> <ui32>
              %259 = ascvec.and %dst, %258, %218 : <ui32>, <ui32> -> <ui32>
              %260 = ascvec.duplicate_scalar %c-766435501_i32, %218 : i32 -> <ui32>
              %261 = ascvec.duplicate_scalar %c-845247145_i32, %218 : i32 -> <ui32>
              %dst0, %dst1 = ascvec.mull %235, %260, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_25, %dst1_26 = ascvec.mull %dst_22, %261, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %262 = ascvec.xor %dst1_26, %259, %218 : <ui32>, <ui32> -> <ui32>
              %263 = ascvec.xor %dst1, %dst_24, %218 : <ui32>, <ui32> -> <ui32>
              %264 = ascvec.duplicate_scalar %arg10, %218 : i32 -> <ui32>
              %265 = ascvec.duplicate_scalar %arg11, %218 : i32 -> <ui32>
              %266 = ascvec.xor %262, %264, %218 : <ui32>, <ui32> -> <ui32>
              %267 = ascvec.xor %263, %265, %218 : <ui32>, <ui32> -> <ui32>
              %dst0_27, %dst1_28 = ascvec.mull %266, %260, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_29, %dst1_30 = ascvec.mull %267, %261, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %268 = ascvec.xor %dst1_30, %dst0_25, %218 : <ui32>, <ui32> -> <ui32>
              %269 = ascvec.xor %dst1_28, %dst0, %218 : <ui32>, <ui32> -> <ui32>
              %270 = arith.addi %arg10, %c-1640531527_i32_13 : i32
              %271 = ascvec.duplicate_scalar %270, %218 : i32 -> <ui32>
              %272 = arith.addi %arg11, %c-1150833019_i32_12 : i32
              %273 = ascvec.duplicate_scalar %272, %218 : i32 -> <ui32>
              %274 = ascvec.xor %268, %271, %218 : <ui32>, <ui32> -> <ui32>
              %275 = ascvec.xor %269, %273, %218 : <ui32>, <ui32> -> <ui32>
              %dst0_31, %dst1_32 = ascvec.mull %274, %260, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_33, %dst1_34 = ascvec.mull %275, %261, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %276 = ascvec.xor %dst1_34, %dst0_29, %218 : <ui32>, <ui32> -> <ui32>
              %277 = ascvec.xor %dst1_32, %dst0_27, %218 : <ui32>, <ui32> -> <ui32>
              %278 = arith.addi %arg10, %c1013904242_i32_11 : i32
              %279 = ascvec.duplicate_scalar %278, %218 : i32 -> <ui32>
              %280 = arith.addi %arg11, %c1993301258_i32_10 : i32
              %281 = ascvec.duplicate_scalar %280, %218 : i32 -> <ui32>
              %282 = ascvec.xor %276, %279, %218 : <ui32>, <ui32> -> <ui32>
              %283 = ascvec.xor %277, %281, %218 : <ui32>, <ui32> -> <ui32>
              %dst0_35, %dst1_36 = ascvec.mull %282, %260, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_37, %dst1_38 = ascvec.mull %283, %261, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %284 = ascvec.xor %dst1_38, %dst0_33, %218 : <ui32>, <ui32> -> <ui32>
              %285 = ascvec.xor %dst1_36, %dst0_31, %218 : <ui32>, <ui32> -> <ui32>
              %286 = arith.addi %arg10, %c-626627285_i32 : i32
              %287 = ascvec.duplicate_scalar %286, %218 : i32 -> <ui32>
              %288 = arith.addi %arg11, %c842468239_i32 : i32
              %289 = ascvec.duplicate_scalar %288, %218 : i32 -> <ui32>
              %290 = ascvec.xor %284, %287, %218 : <ui32>, <ui32> -> <ui32>
              %291 = ascvec.xor %285, %289, %218 : <ui32>, <ui32> -> <ui32>
              %dst0_39, %dst1_40 = ascvec.mull %290, %260, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_41, %dst1_42 = ascvec.mull %291, %261, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %292 = ascvec.xor %dst1_42, %dst0_37, %218 : <ui32>, <ui32> -> <ui32>
              %293 = ascvec.xor %dst1_40, %dst0_35, %218 : <ui32>, <ui32> -> <ui32>
              %294 = arith.addi %arg10, %c2027808484_i32 : i32
              %295 = ascvec.duplicate_scalar %294, %218 : i32 -> <ui32>
              %296 = arith.addi %arg11, %c-308364780_i32 : i32
              %297 = ascvec.duplicate_scalar %296, %218 : i32 -> <ui32>
              %298 = ascvec.xor %292, %295, %218 : <ui32>, <ui32> -> <ui32>
              %299 = ascvec.xor %293, %297, %218 : <ui32>, <ui32> -> <ui32>
              %dst0_43, %dst1_44 = ascvec.mull %298, %260, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_45, %dst1_46 = ascvec.mull %299, %261, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %300 = ascvec.xor %dst1_46, %dst0_41, %218 : <ui32>, <ui32> -> <ui32>
              %301 = ascvec.xor %dst1_44, %dst0_39, %218 : <ui32>, <ui32> -> <ui32>
              %302 = arith.addi %arg10, %c387276957_i32 : i32
              %303 = ascvec.duplicate_scalar %302, %218 : i32 -> <ui32>
              %304 = arith.addi %arg11, %c-1459197799_i32 : i32
              %305 = ascvec.duplicate_scalar %304, %218 : i32 -> <ui32>
              %306 = ascvec.xor %300, %303, %218 : <ui32>, <ui32> -> <ui32>
              %307 = ascvec.xor %301, %305, %218 : <ui32>, <ui32> -> <ui32>
              %dst0_47, %dst1_48 = ascvec.mull %306, %260, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_49, %dst1_50 = ascvec.mull %307, %261, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %308 = ascvec.xor %dst1_50, %dst0_45, %218 : <ui32>, <ui32> -> <ui32>
              %309 = ascvec.xor %dst1_48, %dst0_43, %218 : <ui32>, <ui32> -> <ui32>
              %310 = arith.addi %arg10, %c-1253254570_i32 : i32
              %311 = ascvec.duplicate_scalar %310, %218 : i32 -> <ui32>
              %312 = arith.addi %arg11, %c1684936478_i32 : i32
              %313 = ascvec.duplicate_scalar %312, %218 : i32 -> <ui32>
              %314 = ascvec.xor %308, %311, %218 : <ui32>, <ui32> -> <ui32>
              %315 = ascvec.xor %309, %313, %218 : <ui32>, <ui32> -> <ui32>
              %dst0_51, %dst1_52 = ascvec.mull %314, %260, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_53, %dst1_54 = ascvec.mull %315, %261, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %316 = ascvec.xor %dst1_54, %dst0_49, %218 : <ui32>, <ui32> -> <ui32>
              %317 = ascvec.xor %dst1_52, %dst0_47, %218 : <ui32>, <ui32> -> <ui32>
              %318 = arith.addi %arg10, %c1401181199_i32 : i32
              %319 = ascvec.duplicate_scalar %318, %218 : i32 -> <ui32>
              %320 = arith.addi %arg11, %c534103459_i32 : i32
              %321 = ascvec.duplicate_scalar %320, %218 : i32 -> <ui32>
              %322 = ascvec.xor %316, %319, %218 : <ui32>, <ui32> -> <ui32>
              %323 = ascvec.xor %317, %321, %218 : <ui32>, <ui32> -> <ui32>
              %dst0_55, %dst1_56 = ascvec.mull %322, %260, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_57, %dst1_58 = ascvec.mull %323, %261, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %324 = ascvec.xor %dst1_58, %dst0_53, %218 : <ui32>, <ui32> -> <ui32>
              %325 = ascvec.xor %dst1_56, %dst0_51, %218 : <ui32>, <ui32> -> <ui32>
              %326 = arith.addi %arg10, %c-239350328_i32 : i32
              %327 = ascvec.duplicate_scalar %326, %218 : i32 -> <ui32>
              %328 = arith.addi %arg11, %c-616729560_i32 : i32
              %329 = ascvec.duplicate_scalar %328, %218 : i32 -> <ui32>
              %330 = ascvec.xor %324, %327, %218 : <ui32>, <ui32> -> <ui32>
              %331 = ascvec.xor %325, %329, %218 : <ui32>, <ui32> -> <ui32>
              %dst0_59, %dst1_60 = ascvec.mull %330, %260, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_61, %dst1_62 = ascvec.mull %331, %261, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %332 = ascvec.xor %dst1_62, %dst0_57, %218 : <ui32>, <ui32> -> <ui32>
              %333 = ascvec.xor %dst1_60, %dst0_55, %218 : <ui32>, <ui32> -> <ui32>
              %334 = arith.addi %arg10, %c-1879881855_i32 : i32
              %335 = ascvec.duplicate_scalar %334, %218 : i32 -> <ui32>
              %336 = arith.addi %arg11, %c-1767562579_i32 : i32
              %337 = ascvec.duplicate_scalar %336, %218 : i32 -> <ui32>
              %338 = ascvec.xor %332, %335, %218 : <ui32>, <ui32> -> <ui32>
              %339 = ascvec.xor %333, %337, %218 : <ui32>, <ui32> -> <ui32>
              %low, %high = ascvec.intlv %338, %339 : <ui32>, <ui32> -> <ui32>, <ui32>
              %low_63, %high_64 = ascvec.intlv %dst0_61, %dst0_59 : <ui32>, <ui32> -> <ui32>, <ui32>
              %340 = arith.index_cast %217 : i32 to index
              ascvec.storealign_intlv %arg12[%340], %low, %low_63 : memref<8192xi32, 1>, <ui32>, <ui32>
              %341 = arith.addi %217, %c128_i32 : i32
              %342 = arith.index_cast %341 : i32 to index
              ascvec.storealign_intlv %arg12[%342], %high, %high_64 : memref<8192xi32, 1>, <ui32>, <ui32>
              %343 = arith.addi %216, %arg13 : i64
              %344 = arith.cmpi sge, %343, %arg8 : i64
              %345 = arith.extui %344 : i1 to i64
              %346 = arith.addi %212, %arg14 : i64
              %347 = arith.addi %346, %345 : i64
              %348 = arith.muli %345, %arg8 : i64
              %349 = arith.subi %343, %348 : i64
              %350 = arith.trunci %347 : i64 to i32
              %351 = arith.extui %350 : i32 to i64
              %352 = arith.subi %347, %351 : i64
              %353 = arith.remsi %352, %c4294967296_i64_18 : i64
              %354 = arith.cmpi ne, %353, %c0_i64_16 : i64
              %355 = arith.cmpi slt, %353, %c0_i64_16 : i64
              %356 = arith.andi %354, %355 : i1
              %357 = arith.divsi %352, %c4294967296_i64_18 : i64
              %358 = arith.extui %356 : i1 to i64
              %359 = arith.subi %357, %358 : i64
              %360 = arith.trunci %359 : i64 to i32
              %361 = arith.trunci %349 : i64 to i32
              %362 = arith.extui %361 : i32 to i64
              %363 = arith.subi %349, %362 : i64
              %364 = arith.remsi %363, %c4294967296_i64_18 : i64
              %365 = arith.cmpi ne, %364, %c0_i64_16 : i64
              %366 = arith.cmpi slt, %364, %c0_i64_16 : i64
              %367 = arith.andi %365, %366 : i1
              %368 = arith.divsi %363, %c4294967296_i64_18 : i64
              %369 = arith.extui %367 : i1 to i64
              %370 = arith.subi %368, %369 : i64
              %371 = arith.trunci %370 : i64 to i32
              scf.yield %350, %360, %361, %371 : i32, i32, i32, i32
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          %176 = arith.trunci %82 : i64 to i32
          %177 = arith.trunci %84 : i64 to i32
          ascvec.get_buf v, %50, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
          %178 = ascscalar.bitcast_memref %70 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui32, 1>
          %179 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<32768xui8, 1> -> memref<?xui32, 1>
          ascvec.vf(%85, %88, %96, %97, %178, %alloc_7, %179, %alloc_6, %176, %177, %alloc_5 : i32, i32, i32, i32, memref<?xui32, 1>, memref<8192xi32, 1>, memref<?xui32, 1>, memref<1024xui32, 1>, i32, i32, memref<2048xui8, 1>) {
          ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: memref<?xui32, 1>, %arg13: memref<8192xi32, 1>, %arg14: memref<?xui32, 1>, %arg15: memref<1024xui32, 1>, %arg16: i32, %arg17: i32, %arg18: memref<2048xui8, 1>):
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
            %c1_i32_9 = arith.constant 1 : i32
            %c0_10 = arith.constant 0 : index
            %c64_i32 = arith.constant 64 : i32
            %c7_i32 = arith.constant 7 : i32
            %c8_i32 = arith.constant 8 : i32
            %c0_i32_11 = arith.constant 0 : i32
            %208 = ascvec.create_mask : !ascvec.mask
            %209 = ascvec.arange %c0_i32_11 : i32 -> <ui32>
            %mask, %remaining_out = ascvec.update_mask %c8_i32 : !ascvec.mask
            %210 = ascvec.shr %209, 3, %208 : <ui32> -> <ui32>
            %211 = ascvec.duplicate_scalar %c0_i32_11, %208 : i32 -> <ui32>
            %212 = ascvec.duplicate_scalar %arg8, %208 : i32 -> <ui32>
            %213 = ascvec.cmp_eq_scalar %212, %c7_i32, %208 : <ui32>, i32 -> !ascvec.mask
            scf.for %arg19 = %c0_i32_11 to %arg16 step %c1_i32_9  : i32 {
              %214 = arith.muli %arg19, %c64_i32 : i32
              %215 = ascvec.add_scalar %209, %214, %208 : <ui32>, i32
              %216 = ascvec.shr %215, 5, %208 : <ui32> -> <ui32>
              %217 = ascvec.shr %215, 4, %208 : <ui32> -> <ui32>
              %218 = ascvec.select %216, %217, %213 : <ui32>
              %219 = ascvec.duplicate_scalar %arg9, %208 : i32 -> <ui32>
              %220 = ascvec.and %215, %219, %208 : <ui32>, <ui32> -> <ui32>
              %221 = ascvec.mul_scalar %218, %arg10, %208 : <ui32>, i32
              %222 = ascvec.add %221, %220, %208 : <ui32>
              %223 = ascvec.cmp_lt_scalar %218, %arg11, %208 : <ui32>, i32 -> !ascvec.mask
              %224 = ascvec.cmp_lt_scalar %220, %arg10, %208 : <ui32>, i32 -> !ascvec.mask
              %225 = ascvec.mask_and %223, %224, %208
              %226 = ascvec.shl %222, 1, %208 : <ui32> -> <ui32>
              %227 = ascvec.gather %arg12[%c0_10], %226, %225 : memref<?xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
              %228 = ascvec.add_scalar %226, %c1_i32_9, %208 : <ui32>, i32
              %229 = ascvec.gather %arg12[%c0_10], %228, %225 : memref<?xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
              %230 = ascvec.select %227, %211, %225 : <ui32>
              %231 = ascvec.select %229, %211, %225 : <ui32>
              %232 = ascvec.shl %230, 16, %208 : <ui32> -> <ui32>
              %233 = ascvec.reinterpret %232 : <ui32> -> <f32>
              %234 = ascvec.duplicate_scalar %c-65536_i32, %208 : i32 -> <ui32>
              %235 = ascvec.and %230, %234, %208 : <ui32>, <ui32> -> <ui32>
              %236 = ascvec.reinterpret %235 : <ui32> -> <f32>
              %237 = ascvec.shl %231, 16, %208 : <ui32> -> <ui32>
              %238 = ascvec.reinterpret %237 : <ui32> -> <f32>
              %239 = ascvec.and %231, %234, %208 : <ui32>, <ui32> -> <ui32>
              %240 = ascvec.reinterpret %239 : <ui32> -> <f32>
              %241 = ascvec.reinterpret %233 : <f32> -> <ui32>
              %242 = ascvec.duplicate_scalar %c2147483647_i32, %208 : i32 -> <ui32>
              %243 = ascvec.and %241, %242, %208 : <ui32>, <ui32> -> <ui32>
              %244 = ascvec.reinterpret %236 : <f32> -> <ui32>
              %245 = ascvec.and %244, %242, %208 : <ui32>, <ui32> -> <ui32>
              %246 = ascvec.reinterpret %238 : <f32> -> <ui32>
              %247 = ascvec.and %246, %242, %208 : <ui32>, <ui32> -> <ui32>
              %248 = ascvec.reinterpret %240 : <f32> -> <ui32>
              %249 = ascvec.and %248, %242, %208 : <ui32>, <ui32> -> <ui32>
              %250 = ascvec.max %243, %245, %208 : <ui32>
              %251 = ascvec.max %247, %249, %208 : <ui32>
              %252 = ascvec.max %250, %251, %208 : <ui32>
              %253 = ascvec.reduce_max_datablock %252, %208 : <ui32>
              %254 = ascvec.shr %253, 23, %208 : <ui32> -> <ui32>
              %255 = ascvec.cmp_ge_scalar %254, %c255_i32, %208 : <ui32>, i32 -> !ascvec.mask
              %256 = ascvec.max_scalar %254, %c15_i32, %208 : <ui32>, i32
              %257 = ascvec.add_scalar %256, %c-15_i32, %208 : <ui32>, i32
              %258 = ascvec.duplicate_scalar %c255_i32, %255 merge %257 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
              %259 = ascvec.duplicate_scalar %c254_i32, %208 : i32 -> <ui32>
              %260 = ascvec.sub %259, %258, %208 : <ui32>
              %261 = ascvec.cmp_ne_scalar %258, %c0_i32_11, %208 : <ui32>, i32 -> !ascvec.mask
              %262 = ascvec.shl %260, 23, %261 : <ui32> -> <ui32>
              %263 = ascvec.duplicate_scalar %c2139160576_i32, %255 merge %262 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
              %264 = ascvec.reinterpret %263 : <ui32> -> <f32>
              %265 = ascvec.gather_reg %264, %210 : <f32>, <ui32> -> <f32>
              %266 = ascvec.mul %233, %265, %208 : <f32>
              %267 = ascvec.mul %236, %265, %208 : <f32>
              %268 = ascvec.mul %238, %265, %208 : <f32>
              %269 = ascvec.mul %240, %265, %208 : <f32>
              %270 = ascvec.gather %arg13[%c0_10], %222, %225 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
              %271 = ascvec.reinterpret %270 : <i32> -> <ui32>
              %272 = ascvec.shl %271, 8, %225 : <ui32> -> <ui32>
              %273 = ascvec.shr %271, 24, %225 : <ui32> -> <ui32>
              %274 = ascvec.or %272, %273, %225 : <ui32>, <ui32> -> <ui32>
              %275 = ascvec.shr %271, 16, %208 : <ui32> -> <ui32>
              %276 = ascvec.duplicate_scalar %c65535_i32, %208 : i32 -> <ui32>
              %277 = ascvec.and %271, %276, %208 : <ui32>, <ui32> -> <ui32>
              %278 = ascvec.reinterpret %266 : <f32> -> <ui32>
              %279 = ascvec.shr %278, 24, %208 : <ui32> -> <ui32>
              %280 = ascvec.duplicate_scalar %c128_i32, %208 : i32 -> <ui32>
              %281 = ascvec.and %279, %280, %208 : <ui32>, <ui32> -> <ui32>
              %282 = ascvec.and %278, %242, %208 : <ui32>, <ui32> -> <ui32>
              %283 = ascvec.shr %282, 5, %208 : <ui32> -> <ui32>
              %284 = ascvec.duplicate_scalar %c29360128_i32, %208 : i32 -> <ui32>
              %285 = ascvec.sub %283, %284, %208 : <ui32>
              %286 = ascvec.reinterpret %282 : <ui32> -> <f32>
              %287 = ascvec.mul_scalar %286, %cst, %208 : <f32>, f32
              %288 = ascvec.cast %287, %208 {rounding = 4 : i32} : <f32> -> <i32>
              %289 = ascvec.reinterpret %288 : <i32> -> <ui32>
              %290 = ascvec.cmp_ge_scalar %282, %c947912704_i32, %208 : <ui32>, i32 -> !ascvec.mask
              %291 = ascvec.select %285, %289, %290 : <ui32>
              %292 = ascvec.and %275, %276, %208 : <ui32>, <ui32> -> <ui32>
              %293 = ascvec.add %291, %292, %208 : <ui32>
              %294 = ascvec.shr %293, 16, %208 : <ui32> -> <ui32>
              %295 = ascvec.min_scalar %294, %c123_i32, %208 : <ui32>, i32
              %296 = ascvec.or %295, %281, %208 : <ui32>, <ui32> -> <ui32>
              %297 = ascvec.duplicate_scalar %c127_i32, %208 : i32 -> <ui32>
              %298 = ascvec.cmp_gt_scalar %282, %c2139095040_i32, %208 : <ui32>, i32 -> !ascvec.mask
              %299 = ascvec.select %297, %296, %298 : <ui32>
              %300 = ascvec.and %274, %276, %208 : <ui32>, <ui32> -> <ui32>
              %301 = ascvec.reinterpret %267 : <f32> -> <ui32>
              %302 = ascvec.shr %301, 24, %208 : <ui32> -> <ui32>
              %303 = ascvec.and %302, %280, %208 : <ui32>, <ui32> -> <ui32>
              %304 = ascvec.and %301, %242, %208 : <ui32>, <ui32> -> <ui32>
              %305 = ascvec.shr %304, 5, %208 : <ui32> -> <ui32>
              %306 = ascvec.sub %305, %284, %208 : <ui32>
              %307 = ascvec.reinterpret %304 : <ui32> -> <f32>
              %308 = ascvec.mul_scalar %307, %cst, %208 : <f32>, f32
              %309 = ascvec.cast %308, %208 {rounding = 4 : i32} : <f32> -> <i32>
              %310 = ascvec.reinterpret %309 : <i32> -> <ui32>
              %311 = ascvec.cmp_ge_scalar %304, %c947912704_i32, %208 : <ui32>, i32 -> !ascvec.mask
              %312 = ascvec.select %306, %310, %311 : <ui32>
              %313 = ascvec.and %300, %276, %208 : <ui32>, <ui32> -> <ui32>
              %314 = ascvec.add %312, %313, %208 : <ui32>
              %315 = ascvec.shr %314, 16, %208 : <ui32> -> <ui32>
              %316 = ascvec.min_scalar %315, %c123_i32, %208 : <ui32>, i32
              %317 = ascvec.or %316, %303, %208 : <ui32>, <ui32> -> <ui32>
              %318 = ascvec.cmp_gt_scalar %304, %c2139095040_i32, %208 : <ui32>, i32 -> !ascvec.mask
              %319 = ascvec.select %297, %317, %318 : <ui32>
              %320 = ascvec.reinterpret %268 : <f32> -> <ui32>
              %321 = ascvec.shr %320, 24, %208 : <ui32> -> <ui32>
              %322 = ascvec.and %321, %280, %208 : <ui32>, <ui32> -> <ui32>
              %323 = ascvec.and %320, %242, %208 : <ui32>, <ui32> -> <ui32>
              %324 = ascvec.shr %323, 5, %208 : <ui32> -> <ui32>
              %325 = ascvec.sub %324, %284, %208 : <ui32>
              %326 = ascvec.reinterpret %323 : <ui32> -> <f32>
              %327 = ascvec.mul_scalar %326, %cst, %208 : <f32>, f32
              %328 = ascvec.cast %327, %208 {rounding = 4 : i32} : <f32> -> <i32>
              %329 = ascvec.reinterpret %328 : <i32> -> <ui32>
              %330 = ascvec.cmp_ge_scalar %323, %c947912704_i32, %208 : <ui32>, i32 -> !ascvec.mask
              %331 = ascvec.select %325, %329, %330 : <ui32>
              %332 = ascvec.and %277, %276, %208 : <ui32>, <ui32> -> <ui32>
              %333 = ascvec.add %331, %332, %208 : <ui32>
              %334 = ascvec.shr %333, 16, %208 : <ui32> -> <ui32>
              %335 = ascvec.min_scalar %334, %c123_i32, %208 : <ui32>, i32
              %336 = ascvec.or %335, %322, %208 : <ui32>, <ui32> -> <ui32>
              %337 = ascvec.cmp_gt_scalar %323, %c2139095040_i32, %208 : <ui32>, i32 -> !ascvec.mask
              %338 = ascvec.select %297, %336, %337 : <ui32>
              %339 = ascvec.shr %274, 16, %208 : <ui32> -> <ui32>
              %340 = ascvec.reinterpret %269 : <f32> -> <ui32>
              %341 = ascvec.shr %340, 24, %208 : <ui32> -> <ui32>
              %342 = ascvec.and %341, %280, %208 : <ui32>, <ui32> -> <ui32>
              %343 = ascvec.and %340, %242, %208 : <ui32>, <ui32> -> <ui32>
              %344 = ascvec.shr %343, 5, %208 : <ui32> -> <ui32>
              %345 = ascvec.sub %344, %284, %208 : <ui32>
              %346 = ascvec.reinterpret %343 : <ui32> -> <f32>
              %347 = ascvec.mul_scalar %346, %cst, %208 : <f32>, f32
              %348 = ascvec.cast %347, %208 {rounding = 4 : i32} : <f32> -> <i32>
              %349 = ascvec.reinterpret %348 : <i32> -> <ui32>
              %350 = ascvec.cmp_ge_scalar %343, %c947912704_i32, %208 : <ui32>, i32 -> !ascvec.mask
              %351 = ascvec.select %345, %349, %350 : <ui32>
              %352 = ascvec.and %339, %276, %208 : <ui32>, <ui32> -> <ui32>
              %353 = ascvec.add %351, %352, %208 : <ui32>
              %354 = ascvec.shr %353, 16, %208 : <ui32> -> <ui32>
              %355 = ascvec.min_scalar %354, %c123_i32, %208 : <ui32>, i32
              %356 = ascvec.or %355, %342, %208 : <ui32>, <ui32> -> <ui32>
              %357 = ascvec.cmp_gt_scalar %343, %c2139095040_i32, %208 : <ui32>, i32 -> !ascvec.mask
              %358 = ascvec.select %297, %356, %357 : <ui32>
              %359 = ascvec.shl %319, 8, %208 : <ui32> -> <ui32>
              %360 = ascvec.or %299, %359, %208 : <ui32>, <ui32> -> <ui32>
              %361 = ascvec.shl %338, 16, %208 : <ui32> -> <ui32>
              %362 = ascvec.shl %358, 24, %208 : <ui32> -> <ui32>
              %363 = ascvec.or %361, %362, %208 : <ui32>, <ui32> -> <ui32>
              %364 = ascvec.or %360, %363, %208 : <ui32>, <ui32> -> <ui32>
              ascvec.scatter %arg14[%c0_10], %364, %222, %225 : memref<?xui32, 1>, <ui32>, <ui32>, !ascvec.mask
              %365 = arith.muli %arg19, %c8_i32 : i32
              %366 = arith.index_cast %365 : i32 to index
              ascvec.storealign %arg15[%366], %258, %mask : memref<1024xui32, 1>, index, <ui32>
            }
            ascvec.mem_bar vst_vld
            scf.for %arg19 = %c0_i32_11 to %arg17 step %c64_i32  : i32 {
              %214 = arith.index_cast %arg19 : i32 to index
              %215 = ascvec.loadalign %arg15[%214] : memref<1024xui32, 1>, index -> <ui32>
              %216 = arith.subi %arg17, %arg19 : i32
              %mask_12, %remaining_out_13 = ascvec.update_mask %216 : !ascvec.mask
              ascvec.storealign_pack %arg18[%214], %215, %mask_12 {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %50, 0 {cannir.buffer.generated_sync}
          %180 = arith.cmpi sgt, %61, %c32768_i64 : i64
          %181 = arith.select %180, %c32768_i64, %61 : i64
          %182 = arith.muli %62, %c8_i64 : i64
          %183 = ascscalar.memref_address %alloc_0 : memref<1xui8>
          %184 = arith.divui %182, %c8_i64 : i64
          %185 = arith.addi %183, %184 : i64
          %186 = ascscalar.bind_gm_ptr %185 : memref<1xui8>
          %187 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<32768xui8, 1> -> memref<?xui8, 1>
          %188 = arith.cmpi ugt, %181, %c0_i64 : i64
          scf.if %188 {
            ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
            %208 = arith.trunci %181 : i64 to i32
            ascvec.copy_ub2gm %186[%c0], %187[%c0], %c1_i32, %208, %37, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
          }
          %189 = arith.muli %24, %36 : i64
          %190 = arith.maxsi %189, %c1_i64 : i64
          %191 = arith.divsi %38, %190 : i64
          %192 = arith.minsi %arg6, %191 : i64
          %193 = arith.muli %192, %189 : i64
          %194 = arith.subi %38, %193 : i64
          %195 = arith.minsi %189, %194 : i64
          %196 = arith.cmpi sle, %arg6, %191 : i64
          %197 = arith.select %196, %195, %c0_i64 : i64
          %198 = arith.cmpi sgt, %197, %c2048_i64 : i64
          %199 = arith.select %198, %c2048_i64, %197 : i64
          %200 = arith.muli %arg6, %189 : i64
          %201 = arith.muli %200, %c8_i64 : i64
          %202 = ascscalar.memref_address %alloc_1 : memref<1xui8>
          %203 = arith.divui %201, %c8_i64 : i64
          %204 = arith.addi %202, %203 : i64
          %205 = ascscalar.bind_gm_ptr %204 : memref<1xui8>
          %206 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<2048xui8, 1> -> memref<?xui8, 1>
          %207 = arith.cmpi ugt, %199, %c0_i64 : i64
          scf.if %207 {
            ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
            %208 = arith.trunci %199 : i64 to i32
            ascvec.copy_ub2gm %205[%c0], %206[%c0], %c1_i32, %208, %38, %c2048_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %52 : i64
        }
      } else {
        %7 = cannir.extract_dim %1[1] : !cannir.shape<(-1, -1)>
        %8 = arith.addi %7, %c63_i64 : i64
        %9 = arith.remsi %8, %c64_i64 : i64
        %10 = arith.cmpi ne, %9, %c0_i64 : i64
        %11 = arith.cmpi slt, %9, %c0_i64 : i64
        %12 = arith.andi %10, %11 : i1
        %13 = arith.divsi %8, %c64_i64 : i64
        %14 = arith.extui %12 : i1 to i64
        %15 = arith.subi %13, %14 : i64
        %16 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
        %17 = cannir.extract_shape %16 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
        %18 = cannir.extract_dim %17[1] : !cannir.shape<(-1, -1)>
        %19 = arith.remsi %7, %c16_i64 : i64
        %20 = arith.cmpi ne, %19, %c0_i64 : i64
        %21 = arith.cmpi slt, %19, %c0_i64 : i64
        %22 = arith.andi %20, %21 : i1
        %23 = arith.divsi %7, %c16_i64 : i64
        %24 = arith.extui %22 : i1 to i64
        %25 = arith.subi %23, %24 : i64
        %26 = arith.cmpi eq, %18, %25 : i64
        %27 = arith.select %26, %c4_i64, %c2_i64 : i64
        %28 = arith.muli %2, %15 : i64
        %29 = ascscalar.get_block_num
        %30 = arith.addi %28, %29 : i64
        %31 = arith.subi %30, %c1_i64 : i64
        %32 = arith.remsi %31, %29 : i64
        %33 = arith.cmpi ne, %32, %c0_i64 : i64
        %34 = arith.cmpi slt, %32, %c0_i64 : i64
        %35 = arith.cmpi slt, %29, %c0_i64 : i64
        %36 = arith.xori %34, %35 : i1
        %37 = arith.andi %33, %36 : i1
        %38 = arith.divsi %31, %29 : i64
        %39 = arith.extui %37 : i1 to i64
        %40 = arith.subi %38, %39 : i64
        %41 = arith.addi %40, %c511_i64 : i64
        %42 = arith.remsi %41, %c512_i64 : i64
        %43 = arith.cmpi ne, %42, %c0_i64 : i64
        %44 = arith.cmpi slt, %42, %c0_i64 : i64
        %45 = arith.andi %43, %44 : i1
        %46 = arith.divsi %41, %c512_i64 : i64
        %47 = arith.extui %45 : i1 to i64
        %48 = arith.subi %46, %47 : i64
        %49 = arith.addi %40, %48 : i64
        %50 = arith.subi %49, %c1_i64 : i64
        %51 = arith.remsi %50, %48 : i64
        %52 = arith.cmpi ne, %51, %c0_i64 : i64
        %53 = arith.cmpi slt, %51, %c0_i64 : i64
        %54 = arith.cmpi slt, %48, %c0_i64 : i64
        %55 = arith.xori %53, %54 : i1
        %56 = arith.andi %52, %55 : i1
        %57 = arith.divsi %50, %48 : i64
        %58 = arith.extui %56 : i1 to i64
        %59 = arith.subi %57, %58 : i64
        %60 = arith.cmpi sge, %59, %c4_i64 : i64
        %61 = scf.if %60 -> (i64) {
          %80 = arith.addi %59, %c3_i64 : i64
          %81 = arith.remsi %80, %c4_i64 : i64
          %82 = arith.cmpi ne, %81, %c0_i64 : i64
          %83 = arith.cmpi slt, %81, %c0_i64 : i64
          %84 = arith.andi %82, %83 : i1
          %85 = arith.divsi %80, %c4_i64 : i64
          %86 = arith.extui %84 : i1 to i64
          %87 = arith.subi %85, %86 : i64
          %88 = arith.muli %87, %c4_i64 : i64
          scf.yield %88 : i64
        } else {
          scf.yield %59 : i64
        }
        %62 = arith.muli %29, %c64_i64 : i64
        %63 = arith.cmpi sge, %28, %62 : i64
        %64 = scf.if %63 -> (i64) {
          %80 = arith.addi %61, %c15_i64 : i64
          %81 = arith.remsi %80, %c16_i64 : i64
          %82 = arith.cmpi ne, %81, %c0_i64 : i64
          %83 = arith.cmpi slt, %81, %c0_i64 : i64
          %84 = arith.andi %82, %83 : i1
          %85 = arith.divsi %80, %c16_i64 : i64
          %86 = arith.extui %84 : i1 to i64
          %87 = arith.subi %85, %86 : i64
          %88 = arith.muli %87, %c16_i64 : i64
          scf.yield %88 : i64
        } else {
          scf.yield %61 : i64
        }
        %65 = arith.muli %64, %c64_i64 : i64
        %66 = arith.addi %15, %64 : i64
        %67 = arith.subi %66, %c1_i64 : i64
        %68 = arith.remsi %67, %64 : i64
        %69 = arith.cmpi ne, %68, %c0_i64 : i64
        %70 = arith.cmpi slt, %68, %c0_i64 : i64
        %71 = arith.cmpi slt, %64, %c0_i64 : i64
        %72 = arith.xori %70, %71 : i1
        %73 = arith.andi %69, %72 : i1
        %74 = arith.divsi %67, %64 : i64
        %75 = arith.extui %73 : i1 to i64
        %76 = arith.subi %74, %75 : i64
        %77 = arith.muli %2, %76 : i64
        %78 = ascscalar.get_block_idx
        %79 = scf.for %arg6 = %78 to %77 step %29 iter_args(%arg7 = %c0_i64) -> (i64)  : i64 {
          %80 = arith.remsi %arg6, %76 : i64
          %81 = arith.cmpi ne, %80, %c0_i64 : i64
          %82 = arith.cmpi slt, %80, %c0_i64 : i64
          %83 = arith.cmpi slt, %76, %c0_i64 : i64
          %84 = arith.xori %82, %83 : i1
          %85 = arith.andi %81, %84 : i1
          %86 = arith.divsi %arg6, %76 : i64
          %87 = arith.extui %85 : i1 to i64
          %88 = arith.subi %86, %87 : i64
          %89 = arith.select %85, %76, %c0_i64 : i64
          %90 = arith.addi %80, %89 : i64
          %91 = arith.muli %90, %65 : i64
          %92 = arith.subi %7, %91 : i64
          %93 = arith.cmpi sgt, %92, %65 : i64
          %94 = arith.select %93, %65, %92 : i64
          %95 = arith.cmpi eq, %arg7, %c0_i64 : i64
          %96 = ascscalar.memref_address %alloc_2 : memref<32768xbf16, 1>
          %97 = ascscalar.memref_address %alloc_3 : memref<32768xbf16, 1>
          %98 = arith.select %95, %96, %97 : i64
          %99 = arith.cmpi ne, %arg7, %c0_i64 : i64
          %100 = arith.extui %99 : i1 to i64
          %101 = arith.addi %arg7, %c1_i64 overflow<nsw> : i64
          %102 = arith.remui %101, %c2_i64 : i64
          %103 = arith.minsi %88, %2 : i64
          %104 = arith.subi %2, %103 : i64
          %105 = arith.minsi %104, %c1_i64 : i64
          %106 = arith.cmpi sle, %88, %2 : i64
          %107 = arith.select %106, %105, %c0_i64 : i64
          %108 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %109 = cannir.extract_stride %108 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %110 = cannir.extract_dim %109[0] : !cannir.stride<(-1, 1)>
          %111 = arith.muli %110, %c2_i64 : i64
          %112 = arith.maxsi %65, %c1_i64 : i64
          %113 = arith.divsi %7, %112 : i64
          %114 = arith.minsi %90, %113 : i64
          %115 = arith.muli %114, %65 : i64
          %116 = arith.subi %7, %115 : i64
          %117 = arith.minsi %65, %116 : i64
          %118 = arith.cmpi sle, %90, %113 : i64
          %119 = arith.select %118, %117, %c0_i64 : i64
          %120 = arith.muli %88, %110 : i64
          %121 = arith.muli %120, %c2_i64 : i64
          %122 = arith.muli %91, %c2_i64 : i64
          %123 = arith.addi %121, %122 : i64
          %124 = arith.muli %119, %c2_i64 : i64
          %125 = arith.cmpi eq, %124, %c0_i64 : i64
          %126 = arith.select %125, %c0_i64, %107 : i64
          %127 = arith.select %125, %c2_i64, %124 : i64
          %128 = ascscalar.bind_local_ptr %98 : memref<1xbf16, 1>
          ascvec.get_buf mte2, %100, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%128, %alloc, %c0_i32, %c0_i32, %c0_i64, %123, %126, %127, %111, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %100, 0 {cannir.buffer.generated_sync}
          %129 = ascscalar.bind_local_ptr %98 : memref<32768xbf16, 1>
          %130 = arith.cmpi sge, %94, %c256_i64 : i64
          %131 = arith.remsi %94, %c256_i64 : i64
          %132 = arith.cmpi eq, %131, %c0_i64 : i64
          %133 = arith.select %130, %132, %false : i1
          scf.if %133 {
            %206 = arith.muli %88, %7 : i64
            %207 = arith.addi %206, %91 : i64
            %208 = arith.remsi %arg3, %c256_i64 : i64
            %209 = arith.cmpi ne, %208, %c0_i64 : i64
            %210 = arith.cmpi slt, %208, %c0_i64 : i64
            %211 = arith.andi %209, %210 : i1
            %212 = arith.select %211, %c256_i64, %c0_i64 : i64
            %213 = arith.addi %208, %212 overflow<nsw> : i64
            %214 = arith.cmpi eq, %213, %c0_i64 : i64
            scf.if %214 {
              %215 = arith.cmpi sge, %94, %c1024_i64 : i64
              %216 = arith.remsi %94, %c1024_i64 : i64
              %217 = arith.cmpi eq, %216, %c0_i64 : i64
              %218 = arith.select %215, %217, %false : i1
              scf.if %218 {
                %219 = arith.divsi %94, %c256_i64 : i64
                %220 = arith.addi %94, %c1023_i64 : i64
                %221 = arith.remsi %220, %c1024_i64 : i64
                %222 = arith.cmpi ne, %221, %c0_i64 : i64
                %223 = arith.cmpi slt, %221, %c0_i64 : i64
                %224 = arith.andi %222, %223 : i1
                %225 = arith.divsi %220, %c1024_i64 : i64
                %226 = arith.extui %224 : i1 to i64
                %227 = arith.subi %225, %226 : i64
                %228 = arith.trunci %arg4 : i64 to i32
                %229 = arith.extui %228 : i32 to i64
                %230 = arith.subi %arg4, %229 : i64
                %231 = arith.remsi %230, %c4294967296_i64 : i64
                %232 = arith.cmpi ne, %231, %c0_i64 : i64
                %233 = arith.cmpi slt, %231, %c0_i64 : i64
                %234 = arith.andi %232, %233 : i1
                %235 = arith.divsi %230, %c4294967296_i64 : i64
                %236 = arith.extui %234 : i1 to i64
                %237 = arith.subi %235, %236 : i64
                %238 = arith.trunci %237 : i64 to i32
                %239 = arith.divui %207, %c32_i64 : i64
                %240 = arith.trunci %239 : i64 to i32
                %241 = arith.extui %240 : i32 to i64
                %242 = arith.subi %239, %241 : i64
                %243 = arith.remsi %242, %c4294967296_i64 : i64
                %244 = arith.cmpi ne, %243, %c0_i64 : i64
                %245 = arith.cmpi slt, %243, %c0_i64 : i64
                %246 = arith.andi %244, %245 : i1
                %247 = arith.divsi %242, %c4294967296_i64 : i64
                %248 = arith.extui %246 : i1 to i64
                %249 = arith.subi %247, %248 : i64
                %250 = arith.trunci %249 : i64 to i32
                %251 = arith.trunci %arg5 : i64 to i32
                %252 = arith.extui %251 : i32 to i64
                %253 = arith.subi %arg5, %252 : i64
                %254 = arith.remsi %253, %c4294967296_i64 : i64
                %255 = arith.cmpi ne, %254, %c0_i64 : i64
                %256 = arith.cmpi slt, %254, %c0_i64 : i64
                %257 = arith.andi %255, %256 : i1
                %258 = arith.divsi %253, %c4294967296_i64 : i64
                %259 = arith.extui %257 : i1 to i64
                %260 = arith.subi %258, %259 : i64
                %261 = arith.trunci %260 : i64 to i32
                %262 = arith.divui %251, %c4_i32 : i32
                %263 = arith.remui %261, %c4_i32 : i32
                %264 = arith.muli %263, %c1073741824_i32 : i32
                %265 = arith.addi %262, %264 : i32
                %266 = arith.divui %261, %c4_i32 : i32
                %267 = arith.muli %227, %c32_i64 : i64
                %268 = arith.divui %239, %c16384_i64 : i64
                %269 = arith.addi %239, %267 : i64
                %270 = arith.subi %269, %c1_i64 : i64
                %271 = arith.divui %270, %c16384_i64 : i64
                %272 = arith.cmpi eq, %268, %271 : i64
                scf.if %272 {
                  %282 = arith.trunci %268 : i64 to i32
                  %283 = arith.extui %282 : i32 to i64
                  %284 = arith.subi %268, %283 : i64
                  %285 = arith.remsi %284, %c4294967296_i64 : i64
                  %286 = arith.cmpi ne, %285, %c0_i64 : i64
                  %287 = arith.cmpi slt, %285, %c0_i64 : i64
                  %288 = arith.andi %286, %287 : i1
                  %289 = arith.divsi %284, %c4294967296_i64 : i64
                  %290 = arith.extui %288 : i1 to i64
                  %291 = arith.subi %289, %290 : i64
                  %292 = arith.trunci %291 : i64 to i32
                  %293 = arith.addi %265, %282 : i32
                  %294 = arith.addi %266, %292 : i32
                  %295 = arith.cmpi ult, %293, %265 : i32
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
                  %311 = arith.xori %297, %238 : i32
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
                  %325 = arith.addi %228, %c-1640531527_i32 : i32
                  %326 = arith.addi %238, %c-1150833019_i32 : i32
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
                  %342 = arith.xori %310, %228 : i32
                  %343 = arith.addi %228, %c1013904242_i32 : i32
                  %344 = arith.xori %314, %343 : i32
                  %345 = arith.addi %238, %c1993301258_i32 : i32
                  %346 = arith.xori %341, %345 : i32
                  ascsimt.vf(%240, %342, %326, %344, %346, %228, %238, %331, %alloc_7, %267 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: memref<8192xi32, 1>, %arg17: i64):
                    %c256_i64_9 = arith.constant 256 : i64
                    %c3_i64_10 = arith.constant 3 : i64
                    %c2_i64_11 = arith.constant 2 : i64
                    %c1_i64_12 = arith.constant 1 : i64
                    %c4_i64_13 = arith.constant 4 : i64
                    %c1_i32_14 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c3_i32 = arith.constant 3 : i32
                    %c-1150833019_i32_15 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_16 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %347 = ascsimt.thread_idx {ascsimt.scope = 31 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %348 = arith.extsi %347 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg18 = %348 to %arg17 step %c256_i64_9  : i64 {
                      %349 = arith.trunci %arg18 : i64 to i32
                      %350 = arith.addi %arg8, %349 : i32
                      %351 = arith.remui %350, %c16384_i32 : i32
                      %352 = arith.xori %351, %arg9 : i32
                      %353 = arith.muli %352, %c-766435501_i32 : i32
                      %354 = "ascsimt.bitcast"(%352) : (i32) -> ui32
                      %355 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %356 = ascsimt.math_binary %354, %355, "mulhi" : ui32, ui32 -> ui32
                      %357 = "ascsimt.bitcast"(%356) : (ui32) -> i32
                      %358 = arith.xori %357, %arg10 : i32
                      %359 = arith.muli %358, %c-845247145_i32 : i32
                      %360 = "ascsimt.bitcast"(%358) : (i32) -> ui32
                      %361 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %362 = ascsimt.math_binary %360, %361, "mulhi" : ui32, ui32 -> ui32
                      %363 = "ascsimt.bitcast"(%362) : (ui32) -> i32
                      %364 = arith.xori %363, %arg11 : i32
                      %365 = arith.xori %353, %arg12 : i32
                      %366:4 = scf.for %arg19 = %c3_i32 to %c10_i32 step %c1_i32_14 iter_args(%arg20 = %364, %arg21 = %359, %arg22 = %365, %arg23 = %arg15) -> (i32, i32, i32, i32)  : i32 {
                        %375 = arith.muli %arg20, %c-766435501_i32 : i32
                        %376 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                        %377 = ascsimt.math_binary %376, %355, "mulhi" : ui32, ui32 -> ui32
                        %378 = "ascsimt.bitcast"(%377) : (ui32) -> i32
                        %379 = arith.muli %arg22, %c-845247145_i32 : i32
                        %380 = "ascsimt.bitcast"(%arg22) : (i32) -> ui32
                        %381 = ascsimt.math_binary %380, %361, "mulhi" : ui32, ui32 -> ui32
                        %382 = "ascsimt.bitcast"(%381) : (ui32) -> i32
                        %383 = arith.muli %arg19, %c-1640531527_i32_16 : i32
                        %384 = arith.addi %arg13, %383 : i32
                        %385 = arith.muli %arg19, %c-1150833019_i32_15 : i32
                        %386 = arith.addi %arg14, %385 : i32
                        %387 = arith.xori %382, %arg21 : i32
                        %388 = arith.xori %387, %384 : i32
                        %389 = arith.xori %378, %arg23 : i32
                        %390 = arith.xori %389, %386 : i32
                        scf.yield %388, %379, %390, %375 : i32, i32, i32, i32
                      }
                      %367 = arith.muli %arg18, %c4_i64_13 : i64
                      %368 = arith.index_cast %367 : i64 to index
                      ascsimt.store %arg16[%368], %366#0 : memref<8192xi32, 1>, i32
                      %369 = arith.addi %367, %c1_i64_12 : i64
                      %370 = arith.index_cast %369 : i64 to index
                      ascsimt.store %arg16[%370], %366#1 : memref<8192xi32, 1>, i32
                      %371 = arith.addi %367, %c2_i64_11 : i64
                      %372 = arith.index_cast %371 : i64 to index
                      ascsimt.store %arg16[%372], %366#2 : memref<8192xi32, 1>, i32
                      %373 = arith.addi %367, %c3_i64_10 : i64
                      %374 = arith.index_cast %373 : i64 to index
                      ascsimt.store %arg16[%374], %366#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%240, %250, %265, %266, %228, %238, %alloc_7, %267 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: memref<8192xi32, 1>, %arg15: i64):
                    %c256_i64_9 = arith.constant 256 : i64
                    %c3_i64_10 = arith.constant 3 : i64
                    %c2_i64_11 = arith.constant 2 : i64
                    %c1_i64_12 = arith.constant 1 : i64
                    %c4_i64_13 = arith.constant 4 : i64
                    %c1_i32_14 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c0_i32_15 = arith.constant 0 : i32
                    %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c262144_i32 = arith.constant 262144 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %282 = ascsimt.thread_idx {ascsimt.scope = 32 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %283 = arith.extsi %282 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg16 = %283 to %arg15 step %c256_i64_9  : i64 {
                      %284 = arith.trunci %arg16 : i64 to i32
                      %285 = arith.addi %arg8, %284 : i32
                      %286 = arith.cmpi ult, %285, %arg8 : i32
                      %287 = arith.extui %286 : i1 to i32
                      %288 = arith.addi %arg9, %287 : i32
                      %289 = arith.divui %285, %c16384_i32 : i32
                      %290 = arith.remui %288, %c16384_i32 : i32
                      %291 = arith.muli %290, %c262144_i32 : i32
                      %292 = arith.addi %289, %291 : i32
                      %293 = arith.divui %288, %c16384_i32 : i32
                      %294 = arith.addi %arg10, %292 : i32
                      %295 = arith.addi %arg11, %293 : i32
                      %296 = arith.cmpi ult, %294, %arg10 : i32
                      %297 = arith.extui %296 : i1 to i32
                      %298 = arith.addi %295, %297 : i32
                      %299 = arith.remui %285, %c16384_i32 : i32
                      %300:4 = scf.for %arg17 = %c0_i32_15 to %c10_i32 step %c1_i32_14 iter_args(%arg18 = %c0_i32_15, %arg19 = %299, %arg20 = %294, %arg21 = %298) -> (i32, i32, i32, i32)  : i32 {
                        %309 = arith.muli %arg18, %c-766435501_i32 : i32
                        %310 = "ascsimt.bitcast"(%arg18) : (i32) -> ui32
                        %311 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %312 = ascsimt.math_binary %310, %311, "mulhi" : ui32, ui32 -> ui32
                        %313 = "ascsimt.bitcast"(%312) : (ui32) -> i32
                        %314 = arith.muli %arg20, %c-845247145_i32 : i32
                        %315 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                        %316 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %317 = ascsimt.math_binary %315, %316, "mulhi" : ui32, ui32 -> ui32
                        %318 = "ascsimt.bitcast"(%317) : (ui32) -> i32
                        %319 = arith.muli %arg17, %c-1640531527_i32_17 : i32
                        %320 = arith.addi %arg12, %319 : i32
                        %321 = arith.muli %arg17, %c-1150833019_i32_16 : i32
                        %322 = arith.addi %arg13, %321 : i32
                        %323 = arith.xori %318, %arg19 : i32
                        %324 = arith.xori %323, %320 : i32
                        %325 = arith.xori %313, %arg21 : i32
                        %326 = arith.xori %325, %322 : i32
                        scf.yield %324, %314, %326, %309 : i32, i32, i32, i32
                      }
                      %301 = arith.muli %arg16, %c4_i64_13 : i64
                      %302 = arith.index_cast %301 : i64 to index
                      ascsimt.store %arg14[%302], %300#0 : memref<8192xi32, 1>, i32
                      %303 = arith.addi %301, %c1_i64_12 : i64
                      %304 = arith.index_cast %303 : i64 to index
                      ascsimt.store %arg14[%304], %300#1 : memref<8192xi32, 1>, i32
                      %305 = arith.addi %301, %c2_i64_11 : i64
                      %306 = arith.index_cast %305 : i64 to index
                      ascsimt.store %arg14[%306], %300#2 : memref<8192xi32, 1>, i32
                      %307 = arith.addi %301, %c3_i64_10 : i64
                      %308 = arith.index_cast %307 : i64 to index
                      ascsimt.store %arg14[%308], %300#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %273 = arith.trunci %27 : i64 to i32
                %274 = arith.divsi %273, %c2_i32 : i32
                %275 = arith.trunci %274 : i32 to i8
                %276 = arith.trunci %219 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %277 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%275, %alloc_5, %129, %277, %alloc_8, %276 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, memref<2048xui16, 1>, i32) {
                ^bb0(%arg8: i8, %arg9: memref<2048xui8, 1>, %arg10: memref<32768xbf16, 1>, %arg11: memref<?xui16, 1>, %arg12: memref<2048xui16, 1>, %arg13: i32):
                  %c16_9 = arith.constant 16 : index
                  %c4_i32_10 = arith.constant 4 : i32
                  %c0_i32_11 = arith.constant 0 : i32
                  %c3_i32 = arith.constant 3 : i32
                  %c2_i32_12 = arith.constant 2 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c1_i32_13 = arith.constant 1 : i32
                  %c0_14 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %282 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %283 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %284 = ascvec.duplicate_scalar %c32767_i16, %282 : i16 -> <ui16>
                  %285 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %286 = ascvec.shr %285, 4, %282 : <ui16> -> <ui16>
                  %287 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %288 = ascvec.duplicate_scalar %c1_i8, %283 : i8 -> <ui8>
                  %289 = ascvec.and %287, %288, %283 : <ui8>, <ui8> -> <ui8>
                  %290 = ascvec.cmp_lt_scalar %287, %c16_i8, %283 : <ui8>, i8 -> !ascvec.mask
                  %291 = ascvec.cmp_lt_scalar %289, %arg8, %283 : <ui8>, i8 -> !ascvec.mask
                  %292 = ascvec.mask_and %290, %291, %283
                  ascvec.store_unalign_begin %arg9 : memref<2048xui8, 1>
                  %mask_15, %remaining_out_16 = ascvec.update_mask %c1_i32_13 elem_bits 16 : !ascvec.mask
                  %293 = ascvec.duplicate_scalar %c-1_i16, %mask_15 : i16 -> <ui16>
                  %294 = scf.for %arg14 = %c0_i32_11 to %arg13 step %c4_i32_10 iter_args(%arg15 = %293) -> (!ascvec.vreg<ui16>)  : i32 {
                    %295 = ascvec.duplicate_scalar %c-1_i16, %282 : i16 -> <ui16>
                    %296 = ascvec.duplicate_scalar %c0_i16, %282 : i16 -> <ui16>
                    %297 = arith.muli %arg14, %c256_i32 : i32
                    %298 = arith.index_cast %297 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg10[%298] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %299 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %300 = ascvec.and %299, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %301 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %302 = ascvec.and %301, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %303 = ascvec.max %300, %302, %282 : <ui16>
                    %304 = ascvec.reduce_max_datablock %303, %282 : <ui16>
                    %305 = ascvec.shr %304, 7, %282 : <ui16> -> <ui16>
                    %306 = ascvec.cmp_ge_scalar %305, %c255_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %307 = ascvec.max_scalar %305, %c15_i16, %282 : <ui16>, i16
                    %308 = ascvec.add_scalar %307, %c-15_i16, %282 : <ui16>, i16
                    %309 = ascvec.duplicate_scalar %c255_i16, %306 merge %308 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %310 = ascvec.duplicate_scalar %c254_i16, %282 : i16 -> <ui16>
                    %311 = ascvec.sub %310, %309, %282 : <ui16>
                    %312 = ascvec.cmp_ne_scalar %309, %c0_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %313 = ascvec.shl %311, 7, %312 : <ui16> -> <ui16>
                    %314 = ascvec.max %296, %309, %mask : <ui16>
                    %315 = ascvec.gather_reg %313, %286 : <ui16>, <ui16> -> <ui16>
                    %316 = ascvec.reinterpret %315 : <ui16> -> <bf16>
                    %317 = ascvec.mul %result0, %316, %282 : <bf16>
                    %318 = ascvec.mul %result1, %316, %282 : <bf16>
                    ascvec.storealign_intlv %arg10[%298], %317, %318 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %319 = ascvec.reinterpret %317 : <bf16> -> <ui16>
                    %320 = ascvec.and %319, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %321 = ascvec.reinterpret %318 : <bf16> -> <ui16>
                    %322 = ascvec.and %321, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %323 = ascvec.min %320, %322, %282 : <ui16>
                    %324 = ascvec.min %295, %323, %282 : <ui16>
                    %325 = arith.muli %arg14, %c16_i32 : i32
                    %326 = arith.index_cast %325 : i32 to index
                    ascvec.storealign %arg11[%326], %309, %mask : memref<?xui16, 1>, index, <ui16>
                    %327 = ascvec.reinterpret %309 : <ui16> -> <ui8>
                    %328 = ascvec.squeeze_store_reg %327, %292 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_14], %328 : memref<2048xui8, 1>, <ui8>
                    %329 = arith.addi %arg14, %c1_i32_13 : i32
                    %330 = arith.muli %329, %c256_i32 : i32
                    %331 = arith.index_cast %330 : i32 to index
                    %result0_18, %result1_19 = ascvec.loadalign_deintlv %arg10[%331] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %332 = ascvec.reinterpret %result0_18 : <bf16> -> <ui16>
                    %333 = ascvec.and %332, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %334 = ascvec.reinterpret %result1_19 : <bf16> -> <ui16>
                    %335 = ascvec.and %334, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %336 = ascvec.max %333, %335, %282 : <ui16>
                    %337 = ascvec.reduce_max_datablock %336, %282 : <ui16>
                    %338 = ascvec.shr %337, 7, %282 : <ui16> -> <ui16>
                    %339 = ascvec.cmp_ge_scalar %338, %c255_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %340 = ascvec.max_scalar %338, %c15_i16, %282 : <ui16>, i16
                    %341 = ascvec.add_scalar %340, %c-15_i16, %282 : <ui16>, i16
                    %342 = ascvec.duplicate_scalar %c255_i16, %339 merge %341 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %343 = ascvec.sub %310, %342, %282 : <ui16>
                    %344 = ascvec.cmp_ne_scalar %342, %c0_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %345 = ascvec.shl %343, 7, %344 : <ui16> -> <ui16>
                    %346 = ascvec.max %314, %342, %mask : <ui16>
                    %347 = ascvec.gather_reg %345, %286 : <ui16>, <ui16> -> <ui16>
                    %348 = ascvec.reinterpret %347 : <ui16> -> <bf16>
                    %349 = ascvec.mul %result0_18, %348, %282 : <bf16>
                    %350 = ascvec.mul %result1_19, %348, %282 : <bf16>
                    ascvec.storealign_intlv %arg10[%331], %349, %350 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %351 = ascvec.reinterpret %349 : <bf16> -> <ui16>
                    %352 = ascvec.and %351, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %353 = ascvec.reinterpret %350 : <bf16> -> <ui16>
                    %354 = ascvec.and %353, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %355 = ascvec.min %352, %354, %282 : <ui16>
                    %356 = ascvec.min %324, %355, %282 : <ui16>
                    %357 = arith.muli %329, %c16_i32 : i32
                    %358 = arith.index_cast %357 : i32 to index
                    ascvec.storealign %arg11[%358], %342, %mask : memref<?xui16, 1>, index, <ui16>
                    %359 = ascvec.reinterpret %342 : <ui16> -> <ui8>
                    %360 = ascvec.squeeze_store_reg %359, %292 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_14], %360 : memref<2048xui8, 1>, <ui8>
                    %361 = arith.addi %arg14, %c2_i32_12 : i32
                    %362 = arith.muli %361, %c256_i32 : i32
                    %363 = arith.index_cast %362 : i32 to index
                    %result0_20, %result1_21 = ascvec.loadalign_deintlv %arg10[%363] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %364 = ascvec.reinterpret %result0_20 : <bf16> -> <ui16>
                    %365 = ascvec.and %364, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %366 = ascvec.reinterpret %result1_21 : <bf16> -> <ui16>
                    %367 = ascvec.and %366, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %368 = ascvec.max %365, %367, %282 : <ui16>
                    %369 = ascvec.reduce_max_datablock %368, %282 : <ui16>
                    %370 = ascvec.shr %369, 7, %282 : <ui16> -> <ui16>
                    %371 = ascvec.cmp_ge_scalar %370, %c255_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %372 = ascvec.max_scalar %370, %c15_i16, %282 : <ui16>, i16
                    %373 = ascvec.add_scalar %372, %c-15_i16, %282 : <ui16>, i16
                    %374 = ascvec.duplicate_scalar %c255_i16, %371 merge %373 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %375 = ascvec.sub %310, %374, %282 : <ui16>
                    %376 = ascvec.cmp_ne_scalar %374, %c0_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %377 = ascvec.shl %375, 7, %376 : <ui16> -> <ui16>
                    %378 = ascvec.max %346, %374, %mask : <ui16>
                    %379 = ascvec.gather_reg %377, %286 : <ui16>, <ui16> -> <ui16>
                    %380 = ascvec.reinterpret %379 : <ui16> -> <bf16>
                    %381 = ascvec.mul %result0_20, %380, %282 : <bf16>
                    %382 = ascvec.mul %result1_21, %380, %282 : <bf16>
                    ascvec.storealign_intlv %arg10[%363], %381, %382 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %383 = ascvec.reinterpret %381 : <bf16> -> <ui16>
                    %384 = ascvec.and %383, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %385 = ascvec.reinterpret %382 : <bf16> -> <ui16>
                    %386 = ascvec.and %385, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %387 = ascvec.min %384, %386, %282 : <ui16>
                    %388 = ascvec.min %356, %387, %282 : <ui16>
                    %389 = arith.muli %361, %c16_i32 : i32
                    %390 = arith.index_cast %389 : i32 to index
                    ascvec.storealign %arg11[%390], %374, %mask : memref<?xui16, 1>, index, <ui16>
                    %391 = ascvec.reinterpret %374 : <ui16> -> <ui8>
                    %392 = ascvec.squeeze_store_reg %391, %292 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_14], %392 : memref<2048xui8, 1>, <ui8>
                    %393 = arith.addi %arg14, %c3_i32 : i32
                    %394 = arith.muli %393, %c256_i32 : i32
                    %395 = arith.index_cast %394 : i32 to index
                    %result0_22, %result1_23 = ascvec.loadalign_deintlv %arg10[%395] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %396 = ascvec.reinterpret %result0_22 : <bf16> -> <ui16>
                    %397 = ascvec.and %396, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %398 = ascvec.reinterpret %result1_23 : <bf16> -> <ui16>
                    %399 = ascvec.and %398, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %400 = ascvec.max %397, %399, %282 : <ui16>
                    %401 = ascvec.reduce_max_datablock %400, %282 : <ui16>
                    %402 = ascvec.shr %401, 7, %282 : <ui16> -> <ui16>
                    %403 = ascvec.cmp_ge_scalar %402, %c255_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %404 = ascvec.max_scalar %402, %c15_i16, %282 : <ui16>, i16
                    %405 = ascvec.add_scalar %404, %c-15_i16, %282 : <ui16>, i16
                    %406 = ascvec.duplicate_scalar %c255_i16, %403 merge %405 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %407 = ascvec.sub %310, %406, %282 : <ui16>
                    %408 = ascvec.cmp_ne_scalar %406, %c0_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %409 = ascvec.shl %407, 7, %408 : <ui16> -> <ui16>
                    %410 = ascvec.max %378, %406, %mask : <ui16>
                    %411 = ascvec.gather_reg %409, %286 : <ui16>, <ui16> -> <ui16>
                    %412 = ascvec.reinterpret %411 : <ui16> -> <bf16>
                    %413 = ascvec.mul %result0_22, %412, %282 : <bf16>
                    %414 = ascvec.mul %result1_23, %412, %282 : <bf16>
                    ascvec.storealign_intlv %arg10[%395], %413, %414 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %415 = ascvec.reinterpret %413 : <bf16> -> <ui16>
                    %416 = ascvec.and %415, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %417 = ascvec.reinterpret %414 : <bf16> -> <ui16>
                    %418 = ascvec.and %417, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %419 = ascvec.min %416, %418, %282 : <ui16>
                    %420 = ascvec.min %388, %419, %282 : <ui16>
                    %421 = arith.muli %393, %c16_i32 : i32
                    %422 = arith.index_cast %421 : i32 to index
                    ascvec.storealign %arg11[%422], %406, %mask : memref<?xui16, 1>, index, <ui16>
                    %423 = ascvec.reinterpret %406 : <ui16> -> <ui8>
                    %424 = ascvec.squeeze_store_reg %423, %292 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_14], %424 : memref<2048xui8, 1>, <ui8>
                    %425 = ascvec.reduce_min %420, %282 : <ui16>
                    %426 = ascvec.reduce_max %410, %mask : <ui16>
                    %427 = ascvec.cmp_eq_scalar %426, %c255_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %428 = ascvec.duplicate_scalar %c0_i16, %427 merge %425 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.storealign_1st %arg12[%326], %428 : memref<2048xui16, 1>, index, <ui16>
                    %429 = ascvec.min %arg15, %428, %mask_15 : <ui16>
                    scf.yield %429 : !ascvec.vreg<ui16>
                  }
                  %c0_17 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg14 = %c0_17 to %c1 step %c1 {
                    ascvec.store_unalign_post %arg9[%c0_14] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg12[%c16_9], %294 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %278 = memref.load %alloc_8[%c16] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %279 = cannir.scalar_storage_cast %278 : ui16 -> i16
                %280 = arith.extui %279 : i16 to i64
                %281 = arith.cmpi sge, %280, %c14464_i64 : i64
                scf.if %281 {
                  ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %282 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                  %283 = ascscalar.bitcast_memref %129 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%129, %282, %283, %alloc_4, %276 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_9 = arith.constant 1 : i32
                    %c0_i32_10 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c128_i32 = arith.constant 128 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c31_i16 = arith.constant 31 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %284 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %285 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %286 = ascvec.duplicate_scalar %c255_i16, %284 : i16 -> <ui16>
                    %287 = ascvec.duplicate_scalar %c-128_i8, %285 : i8 -> <ui8>
                    %288 = ascvec.duplicate_scalar %c31_i16, %284 : i16 -> <ui16>
                    scf.for %arg13 = %c0_i32_10 to %arg12 step %c1_i32_9  : i32 {
                      %289 = arith.muli %arg13, %c256_i32 : i32
                      %290 = arith.index_cast %289 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg8[%290] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %291 = arith.muli %arg13, %c128_i32 : i32
                      %292 = arith.index_cast %291 : i32 to index
                      %293 = ascvec.loadalign %arg9[%292] : memref<?xui8, 1>, index -> <ui8>
                      %294 = ascvec.unpack %293 {part = "lower"} : <ui8> -> <ui16>
                      %295 = ascvec.and %294, %288, %284 : <ui16>, <ui16> -> <ui16>
                      %296 = ascvec.shr %294, 3, %284 : <ui16> -> <ui16>
                      %297 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %298 = ascvec.add %297, %295, %284 : <ui16>
                      %299 = ascvec.shr %298, 5, %284 : <ui16> -> <ui16>
                      %300 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %301 = ascvec.add %300, %296, %284 : <ui16>
                      %302 = ascvec.shr %301, 5, %284 : <ui16> -> <ui16>
                      %303 = ascvec.and %299, %286, %284 : <ui16>, <ui16> -> <ui16>
                      %304 = ascvec.shl %302, 8, %284 : <ui16> -> <ui16>
                      %305 = ascvec.or %303, %304, %284 : <ui16>, <ui16> -> <ui16>
                      %306 = ascvec.reinterpret %305 : <ui16> -> <ui8>
                      %307 = ascvec.add_scalar %306, %c64_i8, %285 : <ui8>, i8
                      %308 = ascvec.min_scalar %307, %c123_i8, %285 : <ui8>, i8
                      %309 = arith.muli %arg13, %c512_i32 : i32
                      %310 = arith.index_cast %309 : i32 to index
                      %result0_11, %result1_12 = ascvec.loadalign_deintlv %arg10[%310] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %311 = ascvec.and %result1_12, %287, %285 : <ui8>, <ui8> -> <ui8>
                      %312 = ascvec.or %308, %311, %285 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg11[%290], %312, %285 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
                } else {
                  scf.for %arg8 = %c0_i64 to %219 step %c16_i64  : i64 {
                    %282 = arith.addi %arg8, %c16_i64 : i64
                    %283 = arith.cmpi sgt, %282, %219 : i64
                    %284 = arith.select %283, %219, %282 : i64
                    %285 = arith.muli %arg8, %c16_i64 : i64
                    ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %286 = arith.index_cast %285 : i64 to index
                    %287 = memref.load %alloc_8[%286] : memref<2048xui16, 1>
                    ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %288 = cannir.scalar_storage_cast %287 : ui16 -> i16
                    %289 = arith.extui %288 : i16 to i64
                    %290 = arith.addi %arg8, %c4_i64 : i64
                    %291 = scf.for %arg9 = %290 to %284 step %c4_i64 iter_args(%arg10 = %289) -> (i64)  : i64 {
                      %293 = arith.muli %arg9, %c16_i64 : i64
                      ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %294 = arith.index_cast %293 : i64 to index
                      %295 = memref.load %alloc_8[%294] : memref<2048xui16, 1>
                      ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %296 = cannir.scalar_storage_cast %295 : ui16 -> i16
                      %297 = arith.extui %296 : i16 to i64
                      %298 = arith.cmpi slt, %297, %arg10 : i64
                      %299 = arith.select %298, %297, %arg10 : i64
                      scf.yield %299 : i64
                    }
                    %292 = arith.cmpi sge, %291, %c14464_i64 : i64
                    scf.if %292 {
                      %293 = arith.trunci %arg8 : i64 to i32
                      %294 = arith.trunci %284 : i64 to i32
                      ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %295 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                      %296 = ascscalar.bitcast_memref %129 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                      ascvec.vf(%129, %295, %296, %alloc_4, %293, %294 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                      ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32, %arg14: i32):
                        %c512_i32 = arith.constant 512 : i32
                        %c1_i32_9 = arith.constant 1 : i32
                        %c123_i8 = arith.constant 123 : i8
                        %c64_i8 = arith.constant 64 : i8
                        %c128_i32 = arith.constant 128 : i32
                        %c256_i32 = arith.constant 256 : i32
                        %c31_i16 = arith.constant 31 : i16
                        %c-128_i8 = arith.constant -128 : i8
                        %c255_i16 = arith.constant 255 : i16
                        %297 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                        %298 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                        %299 = ascvec.duplicate_scalar %c255_i16, %297 : i16 -> <ui16>
                        %300 = ascvec.duplicate_scalar %c-128_i8, %298 : i8 -> <ui8>
                        %301 = ascvec.duplicate_scalar %c31_i16, %297 : i16 -> <ui16>
                        scf.for %arg15 = %arg13 to %arg14 step %c1_i32_9  : i32 {
                          %302 = arith.muli %arg15, %c256_i32 : i32
                          %303 = arith.index_cast %302 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg9[%303] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                          %304 = arith.muli %arg15, %c128_i32 : i32
                          %305 = arith.index_cast %304 : i32 to index
                          %306 = ascvec.loadalign %arg10[%305] : memref<?xui8, 1>, index -> <ui8>
                          %307 = ascvec.unpack %306 {part = "lower"} : <ui8> -> <ui16>
                          %308 = ascvec.and %307, %301, %297 : <ui16>, <ui16> -> <ui16>
                          %309 = ascvec.shr %307, 3, %297 : <ui16> -> <ui16>
                          %310 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                          %311 = ascvec.add %310, %308, %297 : <ui16>
                          %312 = ascvec.shr %311, 5, %297 : <ui16> -> <ui16>
                          %313 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                          %314 = ascvec.add %313, %309, %297 : <ui16>
                          %315 = ascvec.shr %314, 5, %297 : <ui16> -> <ui16>
                          %316 = ascvec.and %312, %299, %297 : <ui16>, <ui16> -> <ui16>
                          %317 = ascvec.shl %315, 8, %297 : <ui16> -> <ui16>
                          %318 = ascvec.or %316, %317, %297 : <ui16>, <ui16> -> <ui16>
                          %319 = ascvec.reinterpret %318 : <ui16> -> <ui8>
                          %320 = ascvec.add_scalar %319, %c64_i8, %298 : <ui8>, i8
                          %321 = ascvec.min_scalar %320, %c123_i8, %298 : <ui8>, i8
                          %322 = arith.muli %arg15, %c512_i32 : i32
                          %323 = arith.index_cast %322 : i32 to index
                          %result0_10, %result1_11 = ascvec.loadalign_deintlv %arg11[%323] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                          %324 = ascvec.and %result1_11, %300, %298 : <ui8>, <ui8> -> <ui8>
                          %325 = ascvec.or %321, %324, %298 : <ui8>, <ui8> -> <ui8>
                          ascvec.storealign %arg12[%303], %325, %298 : memref<32768xui8, 1>, index, <ui8>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
                    } else {
                      %293 = arith.cmpi sge, %291, %c13568_i64 : i64
                      scf.if %293 {
                        %294 = arith.trunci %arg8 : i64 to i32
                        %295 = arith.trunci %284 : i64 to i32
                        ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %296 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                        %297 = ascscalar.bitcast_memref %129 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%129, %296, %297, %alloc_4, %294, %295 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32, %arg14: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_9 = arith.constant 1 : i32
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
                          %298 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %299 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %300 = ascvec.duplicate_scalar %c255_i16, %298 : i16 -> <ui16>
                          %301 = ascvec.duplicate_scalar %c-128_i8, %299 : i8 -> <ui8>
                          %302 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %303 = ascvec.shr %302, 1, %299 : <ui8> -> <ui8>
                          %304 = ascvec.shr %303, 2, %299 : <ui8> -> <ui8>
                          %305 = ascvec.shl %304, 2, %299 : <ui8> -> <ui8>
                          %306 = ascvec.duplicate_scalar %c1_i8, %299 : i8 -> <ui8>
                          %307 = ascvec.and %302, %306, %299 : <ui8>, <ui8> -> <ui8>
                          %308 = ascvec.add_scalar %303, %c3_i8, %299 : <ui8>, i8
                          %309 = ascvec.add %308, %307, %299 : <ui8>
                          %310 = ascvec.duplicate_scalar %c3_i8, %299 : i8 -> <ui8>
                          %311 = ascvec.and %309, %310, %299 : <ui8>, <ui8> -> <ui8>
                          %312 = ascvec.or %305, %311, %299 : <ui8>, <ui8> -> <ui8>
                          %313 = ascvec.duplicate_scalar %c4095_i16, %298 : i16 -> <ui16>
                          %314 = ascvec.duplicate_scalar %c32767_i16, %298 : i16 -> <ui16>
                          %315 = ascvec.duplicate_scalar %cst, %298 : bf16 -> <bf16>
                          scf.for %arg15 = %arg13 to %arg14 step %c1_i32_9  : i32 {
                            %316 = arith.muli %arg15, %c256_i32 : i32
                            %317 = arith.index_cast %316 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg9[%317] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %318 = arith.muli %arg15, %c128_i32 : i32
                            %319 = arith.index_cast %318 : i32 to index
                            %320 = ascvec.loadalign %arg10[%319] : memref<?xui8, 1>, index -> <ui8>
                            %321 = ascvec.gather_reg %320, %312 : <ui8>, <ui8> -> <ui8>
                            %322 = ascvec.reinterpret %321 : <ui8> -> <ui16>
                            %323 = ascvec.shr %322, 1, %298 : <ui16> -> <ui16>
                            %324 = ascvec.and %323, %313, %298 : <ui16>, <ui16> -> <ui16>
                            %325 = ascvec.shr %322, 4, %298 : <ui16> -> <ui16>
                            %326 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %327 = ascvec.and %326, %314, %298 : <ui16>, <ui16> -> <ui16>
                            %328 = ascvec.cmp_lt_scalar %327, %c14464_i16, %298 : <ui16>, i16 -> !ascvec.mask
                            %329 = ascvec.shr %324, 7, %298 : <ui16> -> <ui16>
                            %330 = ascvec.add %327, %329, %298 : <ui16>
                            %331 = ascvec.shr %330, 5, %298 : <ui16> -> <ui16>
                            %332 = ascvec.reinterpret %327 : <ui16> -> <bf16>
                            %333 = ascvec.mul %332, %315, %328 : <bf16>
                            %334 = ascvec.cast %333, %298 {rounding = 4 : i32} : <bf16> -> <f16>
                            %335 = ascvec.cast %334, %298 {rounding = 4 : i32} : <f16> -> <i16>
                            %336 = ascvec.reinterpret %335 : <i16> -> <ui16>
                            %337 = ascvec.add %336, %324, %298 : <ui16>
                            %338 = ascvec.shr %337, 12, %298 : <ui16> -> <ui16>
                            %339 = ascvec.add_scalar %338, %c448_i16, %298 : <ui16>, i16
                            %340 = ascvec.select %339, %331, %328 : <ui16>
                            %341 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %342 = ascvec.and %341, %314, %298 : <ui16>, <ui16> -> <ui16>
                            %343 = ascvec.cmp_lt_scalar %342, %c14464_i16, %298 : <ui16>, i16 -> !ascvec.mask
                            %344 = ascvec.shr %325, 7, %298 : <ui16> -> <ui16>
                            %345 = ascvec.add %342, %344, %298 : <ui16>
                            %346 = ascvec.shr %345, 5, %298 : <ui16> -> <ui16>
                            %347 = ascvec.reinterpret %342 : <ui16> -> <bf16>
                            %348 = ascvec.mul %347, %315, %343 : <bf16>
                            %349 = ascvec.cast %348, %298 {rounding = 4 : i32} : <bf16> -> <f16>
                            %350 = ascvec.cast %349, %298 {rounding = 4 : i32} : <f16> -> <i16>
                            %351 = ascvec.reinterpret %350 : <i16> -> <ui16>
                            %352 = ascvec.add %351, %325, %298 : <ui16>
                            %353 = ascvec.shr %352, 12, %298 : <ui16> -> <ui16>
                            %354 = ascvec.add_scalar %353, %c448_i16, %298 : <ui16>, i16
                            %355 = ascvec.select %354, %346, %343 : <ui16>
                            %356 = ascvec.and %340, %300, %298 : <ui16>, <ui16> -> <ui16>
                            %357 = ascvec.shl %355, 8, %298 : <ui16> -> <ui16>
                            %358 = ascvec.or %356, %357, %298 : <ui16>, <ui16> -> <ui16>
                            %359 = ascvec.reinterpret %358 : <ui16> -> <ui8>
                            %360 = ascvec.add_scalar %359, %c64_i8, %299 : <ui8>, i8
                            %361 = ascvec.min_scalar %360, %c123_i8, %299 : <ui8>, i8
                            %362 = arith.muli %arg15, %c512_i32 : i32
                            %363 = arith.index_cast %362 : i32 to index
                            %result0_10, %result1_11 = ascvec.loadalign_deintlv %arg11[%363] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %364 = ascvec.and %result1_11, %301, %299 : <ui8>, <ui8> -> <ui8>
                            %365 = ascvec.or %361, %364, %299 : <ui8>, <ui8> -> <ui8>
                            ascvec.storealign %arg12[%317], %365, %299 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
                      } else {
                        %294 = arith.trunci %arg8 : i64 to i32
                        %295 = arith.trunci %284 : i64 to i32
                        ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %296 = ascscalar.bitcast_memref %129 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%129, %alloc_7, %296, %277, %alloc_4, %294, %295 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<8192xi32, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<?xui16, 1>, %arg13: memref<32768xui8, 1>, %arg14: i32, %arg15: i32):
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
                          %c4_i32_9 = arith.constant 4 : i32
                          %c2_i32_10 = arith.constant 2 : i32
                          %c1_i32_11 = arith.constant 1 : i32
                          %c7_i32 = arith.constant 7 : i32
                          %c32_i32 = arith.constant 32 : i32
                          %c0_i32_12 = arith.constant 0 : i32
                          %c32767_i16 = arith.constant 32767 : i16
                          %c0_13 = arith.constant 0 : index
                          %c256_i32 = arith.constant 256 : i32
                          %c3_i8 = arith.constant 3 : i8
                          %c-128_i8 = arith.constant -128 : i8
                          %c0_i8 = arith.constant 0 : i8
                          %c8_i32 = arith.constant 8 : i32
                          %297 = ascvec.create_mask : !ascvec.mask
                          %298 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %299 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                          %300 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %301 = ascvec.duplicate_scalar %c-128_i8, %299 : i8 -> <ui8>
                          %302 = ascvec.shr %300, 5, %299 : <ui8> -> <ui8>
                          %303 = ascvec.duplicate_scalar %c3_i8, %299 : i8 -> <ui8>
                          %304 = ascvec.and %300, %303, %299 : <ui8>, <ui8> -> <ui8>
                          %305 = ascvec.shl %304, 6, %299 : <ui8> -> <ui8>
                          %306 = ascvec.shr %300, 2, %299 : <ui8> -> <ui8>
                          %307 = ascvec.or %305, %306, %299 : <ui8>, <ui8> -> <ui8>
                          scf.for %arg16 = %arg14 to %arg15 step %c1_i32_11  : i32 {
                            %308 = arith.muli %arg16, %c256_i32 : i32
                            %309 = arith.index_cast %308 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg9[%309] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %310 = ascvec.duplicate_scalar %c32767_i16, %298 : i16 -> <ui16>
                            %311 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %312 = ascvec.and %311, %310, %298 : <ui16>, <ui16> -> <ui16>
                            %313 = ascvec.reinterpret %312 : <ui16> -> <bf16>
                            %314 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %315 = ascvec.and %314, %310, %298 : <ui16>, <ui16> -> <ui16>
                            %316 = ascvec.reinterpret %315 : <ui16> -> <bf16>
                            %317 = ascvec.cast %313, %298 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %318 = ascvec.cast %313, %298 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %319 = ascvec.cast %316, %298 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %320 = ascvec.cast %316, %298 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %321 = ascvec.arange %c0_i32_12 : i32 -> <ui32>
                            %322 = ascvec.shl %321, 2, %297 : <ui32> -> <ui32>
                            %323 = ascvec.add_scalar %322, %c0_i32_12, %297 : <ui32>, i32
                            %324 = ascvec.shr %323, 3, %297 : <ui32> -> <ui32>
                            %325 = arith.muli %arg16, %c32_i32 : i32
                            %326 = ascvec.add_scalar %324, %325, %297 : <ui32>, i32
                            %327 = ascvec.gather %arg10[%c0_13], %326, %297 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %328 = ascvec.reinterpret %327 : <i32> -> <ui32>
                            %329 = ascvec.duplicate_scalar %c7_i32, %297 : i32 -> <ui32>
                            %330 = ascvec.and %323, %329, %297 : <ui32>, <ui32> -> <ui32>
                            %331 = ascvec.shl %328, 11, %297 : <ui32> -> <ui32>
                            %332 = ascvec.shr %328, 21, %297 : <ui32> -> <ui32>
                            %333 = ascvec.or %331, %332, %297 : <ui32>, <ui32> -> <ui32>
                            %334 = ascvec.shr %333, 3, %297 : <ui32> -> <ui32>
                            %335 = ascvec.shl %333, 29, %297 : <ui32> -> <ui32>
                            %336 = ascvec.or %334, %335, %297 : <ui32>, <ui32> -> <ui32>
                            %337 = ascvec.duplicate_scalar %c1_i32_11, %297 : i32 -> <ui32>
                            %338 = ascvec.and %330, %337, %297 : <ui32>, <ui32> -> <ui32>
                            %339 = ascvec.cmp_ne_scalar %338, %c0_i32_12, %297 : <ui32>, i32 -> !ascvec.mask
                            %340 = ascvec.select %336, %333, %339 : <ui32>
                            %341 = ascvec.shr %340, 8, %297 : <ui32> -> <ui32>
                            %342 = ascvec.shl %340, 24, %297 : <ui32> -> <ui32>
                            %343 = ascvec.or %341, %342, %297 : <ui32>, <ui32> -> <ui32>
                            %344 = ascvec.duplicate_scalar %c2_i32_10, %297 : i32 -> <ui32>
                            %345 = ascvec.and %330, %344, %297 : <ui32>, <ui32> -> <ui32>
                            %346 = ascvec.cmp_ne_scalar %345, %c0_i32_12, %297 : <ui32>, i32 -> !ascvec.mask
                            %347 = ascvec.select %343, %340, %346 : <ui32>
                            %348 = ascvec.shr %347, 16, %297 : <ui32> -> <ui32>
                            %349 = ascvec.shl %347, 16, %297 : <ui32> -> <ui32>
                            %350 = ascvec.or %348, %349, %297 : <ui32>, <ui32> -> <ui32>
                            %351 = ascvec.duplicate_scalar %c4_i32_9, %297 : i32 -> <ui32>
                            %352 = ascvec.and %330, %351, %297 : <ui32>, <ui32> -> <ui32>
                            %353 = ascvec.cmp_ne_scalar %352, %c0_i32_12, %297 : <ui32>, i32 -> !ascvec.mask
                            %354 = ascvec.select %350, %347, %353 : <ui32>
                            %355 = ascvec.duplicate_scalar %c65535_i32, %297 : i32 -> <ui32>
                            %356 = ascvec.and %354, %355, %297 : <ui32>, <ui32> -> <ui32>
                            %357 = ascvec.add_scalar %322, %c1_i32_11, %297 : <ui32>, i32
                            %358 = ascvec.shr %357, 3, %297 : <ui32> -> <ui32>
                            %359 = ascvec.add_scalar %358, %325, %297 : <ui32>, i32
                            %360 = ascvec.gather %arg10[%c0_13], %359, %297 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %361 = ascvec.reinterpret %360 : <i32> -> <ui32>
                            %362 = ascvec.and %357, %329, %297 : <ui32>, <ui32> -> <ui32>
                            %363 = ascvec.shl %361, 11, %297 : <ui32> -> <ui32>
                            %364 = ascvec.shr %361, 21, %297 : <ui32> -> <ui32>
                            %365 = ascvec.or %363, %364, %297 : <ui32>, <ui32> -> <ui32>
                            %366 = ascvec.shr %365, 3, %297 : <ui32> -> <ui32>
                            %367 = ascvec.shl %365, 29, %297 : <ui32> -> <ui32>
                            %368 = ascvec.or %366, %367, %297 : <ui32>, <ui32> -> <ui32>
                            %369 = ascvec.and %362, %337, %297 : <ui32>, <ui32> -> <ui32>
                            %370 = ascvec.cmp_ne_scalar %369, %c0_i32_12, %297 : <ui32>, i32 -> !ascvec.mask
                            %371 = ascvec.select %368, %365, %370 : <ui32>
                            %372 = ascvec.shr %371, 8, %297 : <ui32> -> <ui32>
                            %373 = ascvec.shl %371, 24, %297 : <ui32> -> <ui32>
                            %374 = ascvec.or %372, %373, %297 : <ui32>, <ui32> -> <ui32>
                            %375 = ascvec.and %362, %344, %297 : <ui32>, <ui32> -> <ui32>
                            %376 = ascvec.cmp_ne_scalar %375, %c0_i32_12, %297 : <ui32>, i32 -> !ascvec.mask
                            %377 = ascvec.select %374, %371, %376 : <ui32>
                            %378 = ascvec.shr %377, 16, %297 : <ui32> -> <ui32>
                            %379 = ascvec.shl %377, 16, %297 : <ui32> -> <ui32>
                            %380 = ascvec.or %378, %379, %297 : <ui32>, <ui32> -> <ui32>
                            %381 = ascvec.and %362, %351, %297 : <ui32>, <ui32> -> <ui32>
                            %382 = ascvec.cmp_ne_scalar %381, %c0_i32_12, %297 : <ui32>, i32 -> !ascvec.mask
                            %383 = ascvec.select %380, %377, %382 : <ui32>
                            %384 = ascvec.and %383, %355, %297 : <ui32>, <ui32> -> <ui32>
                            %385 = ascvec.add_scalar %322, %c2_i32_10, %297 : <ui32>, i32
                            %386 = ascvec.shr %385, 3, %297 : <ui32> -> <ui32>
                            %387 = ascvec.add_scalar %386, %325, %297 : <ui32>, i32
                            %388 = ascvec.gather %arg10[%c0_13], %387, %297 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %389 = ascvec.reinterpret %388 : <i32> -> <ui32>
                            %390 = ascvec.and %385, %329, %297 : <ui32>, <ui32> -> <ui32>
                            %391 = ascvec.shl %389, 11, %297 : <ui32> -> <ui32>
                            %392 = ascvec.shr %389, 21, %297 : <ui32> -> <ui32>
                            %393 = ascvec.or %391, %392, %297 : <ui32>, <ui32> -> <ui32>
                            %394 = ascvec.shr %393, 3, %297 : <ui32> -> <ui32>
                            %395 = ascvec.shl %393, 29, %297 : <ui32> -> <ui32>
                            %396 = ascvec.or %394, %395, %297 : <ui32>, <ui32> -> <ui32>
                            %397 = ascvec.and %390, %337, %297 : <ui32>, <ui32> -> <ui32>
                            %398 = ascvec.cmp_ne_scalar %397, %c0_i32_12, %297 : <ui32>, i32 -> !ascvec.mask
                            %399 = ascvec.select %396, %393, %398 : <ui32>
                            %400 = ascvec.shr %399, 8, %297 : <ui32> -> <ui32>
                            %401 = ascvec.shl %399, 24, %297 : <ui32> -> <ui32>
                            %402 = ascvec.or %400, %401, %297 : <ui32>, <ui32> -> <ui32>
                            %403 = ascvec.and %390, %344, %297 : <ui32>, <ui32> -> <ui32>
                            %404 = ascvec.cmp_ne_scalar %403, %c0_i32_12, %297 : <ui32>, i32 -> !ascvec.mask
                            %405 = ascvec.select %402, %399, %404 : <ui32>
                            %406 = ascvec.shr %405, 16, %297 : <ui32> -> <ui32>
                            %407 = ascvec.shl %405, 16, %297 : <ui32> -> <ui32>
                            %408 = ascvec.or %406, %407, %297 : <ui32>, <ui32> -> <ui32>
                            %409 = ascvec.and %390, %351, %297 : <ui32>, <ui32> -> <ui32>
                            %410 = ascvec.cmp_ne_scalar %409, %c0_i32_12, %297 : <ui32>, i32 -> !ascvec.mask
                            %411 = ascvec.select %408, %405, %410 : <ui32>
                            %412 = ascvec.and %411, %355, %297 : <ui32>, <ui32> -> <ui32>
                            %413 = ascvec.add_scalar %322, %c3_i32, %297 : <ui32>, i32
                            %414 = ascvec.shr %413, 3, %297 : <ui32> -> <ui32>
                            %415 = ascvec.add_scalar %414, %325, %297 : <ui32>, i32
                            %416 = ascvec.gather %arg10[%c0_13], %415, %297 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %417 = ascvec.reinterpret %416 : <i32> -> <ui32>
                            %418 = ascvec.and %413, %329, %297 : <ui32>, <ui32> -> <ui32>
                            %419 = ascvec.shl %417, 11, %297 : <ui32> -> <ui32>
                            %420 = ascvec.shr %417, 21, %297 : <ui32> -> <ui32>
                            %421 = ascvec.or %419, %420, %297 : <ui32>, <ui32> -> <ui32>
                            %422 = ascvec.shr %421, 3, %297 : <ui32> -> <ui32>
                            %423 = ascvec.shl %421, 29, %297 : <ui32> -> <ui32>
                            %424 = ascvec.or %422, %423, %297 : <ui32>, <ui32> -> <ui32>
                            %425 = ascvec.and %418, %337, %297 : <ui32>, <ui32> -> <ui32>
                            %426 = ascvec.cmp_ne_scalar %425, %c0_i32_12, %297 : <ui32>, i32 -> !ascvec.mask
                            %427 = ascvec.select %424, %421, %426 : <ui32>
                            %428 = ascvec.shr %427, 8, %297 : <ui32> -> <ui32>
                            %429 = ascvec.shl %427, 24, %297 : <ui32> -> <ui32>
                            %430 = ascvec.or %428, %429, %297 : <ui32>, <ui32> -> <ui32>
                            %431 = ascvec.and %418, %344, %297 : <ui32>, <ui32> -> <ui32>
                            %432 = ascvec.cmp_ne_scalar %431, %c0_i32_12, %297 : <ui32>, i32 -> !ascvec.mask
                            %433 = ascvec.select %430, %427, %432 : <ui32>
                            %434 = ascvec.shr %433, 16, %297 : <ui32> -> <ui32>
                            %435 = ascvec.shl %433, 16, %297 : <ui32> -> <ui32>
                            %436 = ascvec.or %434, %435, %297 : <ui32>, <ui32> -> <ui32>
                            %437 = ascvec.and %418, %351, %297 : <ui32>, <ui32> -> <ui32>
                            %438 = ascvec.cmp_ne_scalar %437, %c0_i32_12, %297 : <ui32>, i32 -> !ascvec.mask
                            %439 = ascvec.select %436, %433, %438 : <ui32>
                            %440 = ascvec.and %439, %355, %297 : <ui32>, <ui32> -> <ui32>
                            %441 = ascvec.reinterpret %317 : <f32> -> <ui32>
                            %442 = ascvec.reinterpret %441 : <ui32> -> <f32>
                            %443 = ascvec.max_scalar %442, %cst, %297 : <f32>, f32
                            %444 = ascvec.add %442, %443, %297 : <f32>
                            %445 = ascvec.reinterpret %444 : <f32> -> <ui32>
                            %446 = ascvec.shr %445, 5, %297 : <ui32> -> <ui32>
                            %447 = ascvec.add %446, %356, %297 : <ui32>
                            %448 = ascvec.reinterpret %319 : <f32> -> <ui32>
                            %449 = ascvec.reinterpret %448 : <ui32> -> <f32>
                            %450 = ascvec.max_scalar %449, %cst, %297 : <f32>, f32
                            %451 = ascvec.add %449, %450, %297 : <f32>
                            %452 = ascvec.reinterpret %451 : <f32> -> <ui32>
                            %453 = ascvec.shr %452, 5, %297 : <ui32> -> <ui32>
                            %454 = ascvec.add %453, %384, %297 : <ui32>
                            %455 = ascvec.reinterpret %318 : <f32> -> <ui32>
                            %456 = ascvec.reinterpret %455 : <ui32> -> <f32>
                            %457 = ascvec.max_scalar %456, %cst, %297 : <f32>, f32
                            %458 = ascvec.add %456, %457, %297 : <f32>
                            %459 = ascvec.reinterpret %458 : <f32> -> <ui32>
                            %460 = ascvec.shr %459, 5, %297 : <ui32> -> <ui32>
                            %461 = ascvec.add %460, %412, %297 : <ui32>
                            %462 = ascvec.reinterpret %320 : <f32> -> <ui32>
                            %463 = ascvec.reinterpret %462 : <ui32> -> <f32>
                            %464 = ascvec.max_scalar %463, %cst, %297 : <f32>, f32
                            %465 = ascvec.add %463, %464, %297 : <f32>
                            %466 = ascvec.reinterpret %465 : <f32> -> <ui32>
                            %467 = ascvec.shr %466, 5, %297 : <ui32> -> <ui32>
                            %468 = ascvec.add %467, %440, %297 : <ui32>
                            %469 = ascvec.reinterpret %447 : <ui32> -> <ui16>
                            %470 = ascvec.reinterpret %454 : <ui32> -> <ui16>
                            %result0_14, %result1_15 = ascvec.deintlv %469, %470 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %471 = ascvec.reinterpret %461 : <ui32> -> <ui16>
                            %472 = ascvec.reinterpret %468 : <ui32> -> <ui16>
                            %result0_16, %result1_17 = ascvec.deintlv %471, %472 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %473 = ascvec.reinterpret %result1_15 : <ui16> -> <ui8>
                            %474 = ascvec.reinterpret %result1_17 : <ui16> -> <ui8>
                            %result0_18, %result1_19 = ascvec.deintlv %473, %474 : <ui8>, <ui8> -> <ui8>, <ui8>
                            %475 = ascvec.gather_reg %result0_18, %307 : <ui8>, <ui8> -> <ui8>
                            %476 = ascvec.add_scalar %475, %c60_i8, %299 : <ui8>, i8
                            %477 = ascvec.cmp_ne_scalar %476, %c-1_i8, %299 : <ui8>, i8 -> !ascvec.mask
                            %478 = ascvec.min_scalar %476, %c123_i8, %477 : <ui8>, i8
                            %479 = arith.muli %arg16, %c512_i32 : i32
                            %480 = arith.index_cast %479 : i32 to index
                            %result0_20, %result1_21 = ascvec.loadalign_deintlv %arg11[%480] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %481 = ascvec.and %result1_21, %301, %299 : <ui8>, <ui8> -> <ui8>
                            %482 = ascvec.or %478, %481, %299 : <ui8>, <ui8> -> <ui8>
                            %483 = ascvec.arange %c0_i16 : i16 -> <ui16>
                            %484 = arith.muli %arg16, %c16_i32 : i32
                            %485 = arith.trunci %484 : i32 to i16
                            %486 = ascvec.add_scalar %483, %485, %298 : <ui16>, i16
                            %487 = ascvec.gather %arg12[%c0_13], %486, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                            %488 = ascvec.pack %487 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                            %489 = ascvec.gather_reg %488, %302 : <ui8>, <ui8> -> <ui8>
                            %490 = ascvec.cmp_eq_scalar %489, %c-1_i8, %299 : <ui8>, i8 -> !ascvec.mask
                            %491 = ascvec.duplicate_scalar %c127_i8, %490 merge %482 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                            ascvec.storealign %arg13[%309], %491, %299 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
                      }
                    }
                  }
                }
              } else {
                %219 = arith.divsi %94, %c256_i64 : i64
                %220 = arith.addi %94, %c1023_i64 : i64
                %221 = arith.remsi %220, %c1024_i64 : i64
                %222 = arith.cmpi ne, %221, %c0_i64 : i64
                %223 = arith.cmpi slt, %221, %c0_i64 : i64
                %224 = arith.andi %222, %223 : i1
                %225 = arith.divsi %220, %c1024_i64 : i64
                %226 = arith.extui %224 : i1 to i64
                %227 = arith.subi %225, %226 : i64
                %228 = arith.trunci %arg4 : i64 to i32
                %229 = arith.extui %228 : i32 to i64
                %230 = arith.subi %arg4, %229 : i64
                %231 = arith.remsi %230, %c4294967296_i64 : i64
                %232 = arith.cmpi ne, %231, %c0_i64 : i64
                %233 = arith.cmpi slt, %231, %c0_i64 : i64
                %234 = arith.andi %232, %233 : i1
                %235 = arith.divsi %230, %c4294967296_i64 : i64
                %236 = arith.extui %234 : i1 to i64
                %237 = arith.subi %235, %236 : i64
                %238 = arith.trunci %237 : i64 to i32
                %239 = arith.divui %207, %c32_i64 : i64
                %240 = arith.trunci %239 : i64 to i32
                %241 = arith.extui %240 : i32 to i64
                %242 = arith.subi %239, %241 : i64
                %243 = arith.remsi %242, %c4294967296_i64 : i64
                %244 = arith.cmpi ne, %243, %c0_i64 : i64
                %245 = arith.cmpi slt, %243, %c0_i64 : i64
                %246 = arith.andi %244, %245 : i1
                %247 = arith.divsi %242, %c4294967296_i64 : i64
                %248 = arith.extui %246 : i1 to i64
                %249 = arith.subi %247, %248 : i64
                %250 = arith.trunci %249 : i64 to i32
                %251 = arith.trunci %arg5 : i64 to i32
                %252 = arith.extui %251 : i32 to i64
                %253 = arith.subi %arg5, %252 : i64
                %254 = arith.remsi %253, %c4294967296_i64 : i64
                %255 = arith.cmpi ne, %254, %c0_i64 : i64
                %256 = arith.cmpi slt, %254, %c0_i64 : i64
                %257 = arith.andi %255, %256 : i1
                %258 = arith.divsi %253, %c4294967296_i64 : i64
                %259 = arith.extui %257 : i1 to i64
                %260 = arith.subi %258, %259 : i64
                %261 = arith.trunci %260 : i64 to i32
                %262 = arith.divui %251, %c4_i32 : i32
                %263 = arith.remui %261, %c4_i32 : i32
                %264 = arith.muli %263, %c1073741824_i32 : i32
                %265 = arith.addi %262, %264 : i32
                %266 = arith.divui %261, %c4_i32 : i32
                %267 = arith.muli %227, %c32_i64 : i64
                %268 = arith.divui %239, %c16384_i64 : i64
                %269 = arith.addi %239, %267 : i64
                %270 = arith.subi %269, %c1_i64 : i64
                %271 = arith.divui %270, %c16384_i64 : i64
                %272 = arith.cmpi eq, %268, %271 : i64
                scf.if %272 {
                  %282 = arith.trunci %268 : i64 to i32
                  %283 = arith.extui %282 : i32 to i64
                  %284 = arith.subi %268, %283 : i64
                  %285 = arith.remsi %284, %c4294967296_i64 : i64
                  %286 = arith.cmpi ne, %285, %c0_i64 : i64
                  %287 = arith.cmpi slt, %285, %c0_i64 : i64
                  %288 = arith.andi %286, %287 : i1
                  %289 = arith.divsi %284, %c4294967296_i64 : i64
                  %290 = arith.extui %288 : i1 to i64
                  %291 = arith.subi %289, %290 : i64
                  %292 = arith.trunci %291 : i64 to i32
                  %293 = arith.addi %265, %282 : i32
                  %294 = arith.addi %266, %292 : i32
                  %295 = arith.cmpi ult, %293, %265 : i32
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
                  %311 = arith.xori %297, %238 : i32
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
                  %325 = arith.addi %228, %c-1640531527_i32 : i32
                  %326 = arith.addi %238, %c-1150833019_i32 : i32
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
                  %342 = arith.xori %310, %228 : i32
                  %343 = arith.addi %228, %c1013904242_i32 : i32
                  %344 = arith.xori %314, %343 : i32
                  %345 = arith.addi %238, %c1993301258_i32 : i32
                  %346 = arith.xori %341, %345 : i32
                  ascsimt.vf(%240, %342, %326, %344, %346, %228, %238, %331, %alloc_7, %267 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: memref<8192xi32, 1>, %arg17: i64):
                    %c256_i64_9 = arith.constant 256 : i64
                    %c3_i64_10 = arith.constant 3 : i64
                    %c2_i64_11 = arith.constant 2 : i64
                    %c1_i64_12 = arith.constant 1 : i64
                    %c4_i64_13 = arith.constant 4 : i64
                    %c1_i32_14 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c3_i32 = arith.constant 3 : i32
                    %c-1150833019_i32_15 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_16 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %347 = ascsimt.thread_idx {ascsimt.scope = 38 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %348 = arith.extsi %347 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg18 = %348 to %arg17 step %c256_i64_9  : i64 {
                      %349 = arith.trunci %arg18 : i64 to i32
                      %350 = arith.addi %arg8, %349 : i32
                      %351 = arith.remui %350, %c16384_i32 : i32
                      %352 = arith.xori %351, %arg9 : i32
                      %353 = arith.muli %352, %c-766435501_i32 : i32
                      %354 = "ascsimt.bitcast"(%352) : (i32) -> ui32
                      %355 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %356 = ascsimt.math_binary %354, %355, "mulhi" : ui32, ui32 -> ui32
                      %357 = "ascsimt.bitcast"(%356) : (ui32) -> i32
                      %358 = arith.xori %357, %arg10 : i32
                      %359 = arith.muli %358, %c-845247145_i32 : i32
                      %360 = "ascsimt.bitcast"(%358) : (i32) -> ui32
                      %361 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %362 = ascsimt.math_binary %360, %361, "mulhi" : ui32, ui32 -> ui32
                      %363 = "ascsimt.bitcast"(%362) : (ui32) -> i32
                      %364 = arith.xori %363, %arg11 : i32
                      %365 = arith.xori %353, %arg12 : i32
                      %366:4 = scf.for %arg19 = %c3_i32 to %c10_i32 step %c1_i32_14 iter_args(%arg20 = %364, %arg21 = %359, %arg22 = %365, %arg23 = %arg15) -> (i32, i32, i32, i32)  : i32 {
                        %375 = arith.muli %arg20, %c-766435501_i32 : i32
                        %376 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                        %377 = ascsimt.math_binary %376, %355, "mulhi" : ui32, ui32 -> ui32
                        %378 = "ascsimt.bitcast"(%377) : (ui32) -> i32
                        %379 = arith.muli %arg22, %c-845247145_i32 : i32
                        %380 = "ascsimt.bitcast"(%arg22) : (i32) -> ui32
                        %381 = ascsimt.math_binary %380, %361, "mulhi" : ui32, ui32 -> ui32
                        %382 = "ascsimt.bitcast"(%381) : (ui32) -> i32
                        %383 = arith.muli %arg19, %c-1640531527_i32_16 : i32
                        %384 = arith.addi %arg13, %383 : i32
                        %385 = arith.muli %arg19, %c-1150833019_i32_15 : i32
                        %386 = arith.addi %arg14, %385 : i32
                        %387 = arith.xori %382, %arg21 : i32
                        %388 = arith.xori %387, %384 : i32
                        %389 = arith.xori %378, %arg23 : i32
                        %390 = arith.xori %389, %386 : i32
                        scf.yield %388, %379, %390, %375 : i32, i32, i32, i32
                      }
                      %367 = arith.muli %arg18, %c4_i64_13 : i64
                      %368 = arith.index_cast %367 : i64 to index
                      ascsimt.store %arg16[%368], %366#0 : memref<8192xi32, 1>, i32
                      %369 = arith.addi %367, %c1_i64_12 : i64
                      %370 = arith.index_cast %369 : i64 to index
                      ascsimt.store %arg16[%370], %366#1 : memref<8192xi32, 1>, i32
                      %371 = arith.addi %367, %c2_i64_11 : i64
                      %372 = arith.index_cast %371 : i64 to index
                      ascsimt.store %arg16[%372], %366#2 : memref<8192xi32, 1>, i32
                      %373 = arith.addi %367, %c3_i64_10 : i64
                      %374 = arith.index_cast %373 : i64 to index
                      ascsimt.store %arg16[%374], %366#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%240, %250, %265, %266, %228, %238, %alloc_7, %267 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: memref<8192xi32, 1>, %arg15: i64):
                    %c256_i64_9 = arith.constant 256 : i64
                    %c3_i64_10 = arith.constant 3 : i64
                    %c2_i64_11 = arith.constant 2 : i64
                    %c1_i64_12 = arith.constant 1 : i64
                    %c4_i64_13 = arith.constant 4 : i64
                    %c1_i32_14 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c0_i32_15 = arith.constant 0 : i32
                    %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c262144_i32 = arith.constant 262144 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %282 = ascsimt.thread_idx {ascsimt.scope = 39 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %283 = arith.extsi %282 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg16 = %283 to %arg15 step %c256_i64_9  : i64 {
                      %284 = arith.trunci %arg16 : i64 to i32
                      %285 = arith.addi %arg8, %284 : i32
                      %286 = arith.cmpi ult, %285, %arg8 : i32
                      %287 = arith.extui %286 : i1 to i32
                      %288 = arith.addi %arg9, %287 : i32
                      %289 = arith.divui %285, %c16384_i32 : i32
                      %290 = arith.remui %288, %c16384_i32 : i32
                      %291 = arith.muli %290, %c262144_i32 : i32
                      %292 = arith.addi %289, %291 : i32
                      %293 = arith.divui %288, %c16384_i32 : i32
                      %294 = arith.addi %arg10, %292 : i32
                      %295 = arith.addi %arg11, %293 : i32
                      %296 = arith.cmpi ult, %294, %arg10 : i32
                      %297 = arith.extui %296 : i1 to i32
                      %298 = arith.addi %295, %297 : i32
                      %299 = arith.remui %285, %c16384_i32 : i32
                      %300:4 = scf.for %arg17 = %c0_i32_15 to %c10_i32 step %c1_i32_14 iter_args(%arg18 = %c0_i32_15, %arg19 = %299, %arg20 = %294, %arg21 = %298) -> (i32, i32, i32, i32)  : i32 {
                        %309 = arith.muli %arg18, %c-766435501_i32 : i32
                        %310 = "ascsimt.bitcast"(%arg18) : (i32) -> ui32
                        %311 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %312 = ascsimt.math_binary %310, %311, "mulhi" : ui32, ui32 -> ui32
                        %313 = "ascsimt.bitcast"(%312) : (ui32) -> i32
                        %314 = arith.muli %arg20, %c-845247145_i32 : i32
                        %315 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                        %316 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %317 = ascsimt.math_binary %315, %316, "mulhi" : ui32, ui32 -> ui32
                        %318 = "ascsimt.bitcast"(%317) : (ui32) -> i32
                        %319 = arith.muli %arg17, %c-1640531527_i32_17 : i32
                        %320 = arith.addi %arg12, %319 : i32
                        %321 = arith.muli %arg17, %c-1150833019_i32_16 : i32
                        %322 = arith.addi %arg13, %321 : i32
                        %323 = arith.xori %318, %arg19 : i32
                        %324 = arith.xori %323, %320 : i32
                        %325 = arith.xori %313, %arg21 : i32
                        %326 = arith.xori %325, %322 : i32
                        scf.yield %324, %314, %326, %309 : i32, i32, i32, i32
                      }
                      %301 = arith.muli %arg16, %c4_i64_13 : i64
                      %302 = arith.index_cast %301 : i64 to index
                      ascsimt.store %arg14[%302], %300#0 : memref<8192xi32, 1>, i32
                      %303 = arith.addi %301, %c1_i64_12 : i64
                      %304 = arith.index_cast %303 : i64 to index
                      ascsimt.store %arg14[%304], %300#1 : memref<8192xi32, 1>, i32
                      %305 = arith.addi %301, %c2_i64_11 : i64
                      %306 = arith.index_cast %305 : i64 to index
                      ascsimt.store %arg14[%306], %300#2 : memref<8192xi32, 1>, i32
                      %307 = arith.addi %301, %c3_i64_10 : i64
                      %308 = arith.index_cast %307 : i64 to index
                      ascsimt.store %arg14[%308], %300#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %273 = arith.trunci %27 : i64 to i32
                %274 = arith.divsi %273, %c2_i32 : i32
                %275 = arith.trunci %274 : i32 to i8
                %276 = arith.trunci %219 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %277 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%275, %alloc_5, %129, %277, %276, %alloc_8 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, i32, memref<2048xui16, 1>) {
                ^bb0(%arg8: i8, %arg9: memref<2048xui8, 1>, %arg10: memref<32768xbf16, 1>, %arg11: memref<?xui16, 1>, %arg12: i32, %arg13: memref<2048xui16, 1>):
                  %c1_i32_9 = arith.constant 1 : i32
                  %c0_i32_10 = arith.constant 0 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c0_11 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %282 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %283 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %284 = ascvec.duplicate_scalar %c32767_i16, %282 : i16 -> <ui16>
                  %285 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %286 = ascvec.shr %285, 4, %282 : <ui16> -> <ui16>
                  %287 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %288 = ascvec.duplicate_scalar %c1_i8, %283 : i8 -> <ui8>
                  %289 = ascvec.and %287, %288, %283 : <ui8>, <ui8> -> <ui8>
                  %290 = ascvec.cmp_lt_scalar %287, %c16_i8, %283 : <ui8>, i8 -> !ascvec.mask
                  %291 = ascvec.cmp_lt_scalar %289, %arg8, %283 : <ui8>, i8 -> !ascvec.mask
                  %292 = ascvec.mask_and %290, %291, %283
                  ascvec.store_unalign_begin %arg9 : memref<2048xui8, 1>
                  %293 = ascvec.duplicate_scalar %c-1_i16, %282 : i16 -> <ui16>
                  %294 = ascvec.duplicate_scalar %c0_i16, %282 : i16 -> <ui16>
                  %295:2 = scf.for %arg14 = %c0_i32_10 to %arg12 step %c1_i32_9 iter_args(%arg15 = %294, %arg16 = %293) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i32 {
                    %296 = arith.muli %arg14, %c256_i32 : i32
                    %297 = arith.index_cast %296 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg10[%297] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %298 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %299 = ascvec.and %298, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %300 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %301 = ascvec.and %300, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %302 = ascvec.max %299, %301, %282 : <ui16>
                    %303 = ascvec.reduce_max_datablock %302, %282 : <ui16>
                    %304 = ascvec.shr %303, 7, %282 : <ui16> -> <ui16>
                    %305 = ascvec.cmp_ge_scalar %304, %c255_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %306 = ascvec.max_scalar %304, %c15_i16, %282 : <ui16>, i16
                    %307 = ascvec.add_scalar %306, %c-15_i16, %282 : <ui16>, i16
                    %308 = ascvec.duplicate_scalar %c255_i16, %305 merge %307 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %309 = ascvec.duplicate_scalar %c254_i16, %282 : i16 -> <ui16>
                    %310 = ascvec.sub %309, %308, %282 : <ui16>
                    %311 = ascvec.cmp_ne_scalar %308, %c0_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %312 = ascvec.shl %310, 7, %311 : <ui16> -> <ui16>
                    %313 = ascvec.max %arg15, %308, %mask : <ui16>
                    %314 = ascvec.gather_reg %312, %286 : <ui16>, <ui16> -> <ui16>
                    %315 = ascvec.reinterpret %314 : <ui16> -> <bf16>
                    %316 = ascvec.mul %result0, %315, %282 : <bf16>
                    %317 = ascvec.mul %result1, %315, %282 : <bf16>
                    ascvec.storealign_intlv %arg10[%297], %316, %317 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %318 = ascvec.reinterpret %316 : <bf16> -> <ui16>
                    %319 = ascvec.and %318, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %320 = ascvec.reinterpret %317 : <bf16> -> <ui16>
                    %321 = ascvec.and %320, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %322 = ascvec.min %319, %321, %282 : <ui16>
                    %323 = ascvec.min %arg16, %322, %282 : <ui16>
                    %324 = arith.muli %arg14, %c16_i32 : i32
                    %325 = arith.index_cast %324 : i32 to index
                    ascvec.storealign %arg11[%325], %308, %mask : memref<?xui16, 1>, index, <ui16>
                    %326 = ascvec.reinterpret %308 : <ui16> -> <ui8>
                    %327 = ascvec.squeeze_store_reg %326, %292 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_11], %327 : memref<2048xui8, 1>, <ui8>
                    scf.yield %313, %323 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                  }
                  %c0_12 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg14 = %c0_12 to %c1 step %c1 {
                    %296 = ascvec.reduce_min %295#1, %282 : <ui16>
                    %297 = ascvec.reduce_max %295#0, %mask : <ui16>
                    %298 = ascvec.cmp_eq_scalar %297, %c255_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %299 = ascvec.duplicate_scalar %c0_i16, %298 merge %296 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.store_unalign_post %arg9[%c0_11] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg13[%c0_11], %299 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %278 = memref.load %alloc_8[%c0] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %279 = cannir.scalar_storage_cast %278 : ui16 -> i16
                %280 = arith.extui %279 : i16 to i64
                %281 = arith.cmpi sge, %280, %c14464_i64 : i64
                scf.if %281 {
                  ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %282 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                  %283 = ascscalar.bitcast_memref %129 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%129, %282, %283, %alloc_4, %276 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_9 = arith.constant 1 : i32
                    %c0_i32_10 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c128_i32 = arith.constant 128 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c31_i16 = arith.constant 31 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %284 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %285 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %286 = ascvec.duplicate_scalar %c255_i16, %284 : i16 -> <ui16>
                    %287 = ascvec.duplicate_scalar %c-128_i8, %285 : i8 -> <ui8>
                    %288 = ascvec.duplicate_scalar %c31_i16, %284 : i16 -> <ui16>
                    scf.for %arg13 = %c0_i32_10 to %arg12 step %c1_i32_9  : i32 {
                      %289 = arith.muli %arg13, %c256_i32 : i32
                      %290 = arith.index_cast %289 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg8[%290] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %291 = arith.muli %arg13, %c128_i32 : i32
                      %292 = arith.index_cast %291 : i32 to index
                      %293 = ascvec.loadalign %arg9[%292] : memref<?xui8, 1>, index -> <ui8>
                      %294 = ascvec.unpack %293 {part = "lower"} : <ui8> -> <ui16>
                      %295 = ascvec.and %294, %288, %284 : <ui16>, <ui16> -> <ui16>
                      %296 = ascvec.shr %294, 3, %284 : <ui16> -> <ui16>
                      %297 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %298 = ascvec.add %297, %295, %284 : <ui16>
                      %299 = ascvec.shr %298, 5, %284 : <ui16> -> <ui16>
                      %300 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %301 = ascvec.add %300, %296, %284 : <ui16>
                      %302 = ascvec.shr %301, 5, %284 : <ui16> -> <ui16>
                      %303 = ascvec.and %299, %286, %284 : <ui16>, <ui16> -> <ui16>
                      %304 = ascvec.shl %302, 8, %284 : <ui16> -> <ui16>
                      %305 = ascvec.or %303, %304, %284 : <ui16>, <ui16> -> <ui16>
                      %306 = ascvec.reinterpret %305 : <ui16> -> <ui8>
                      %307 = ascvec.add_scalar %306, %c64_i8, %285 : <ui8>, i8
                      %308 = ascvec.min_scalar %307, %c123_i8, %285 : <ui8>, i8
                      %309 = arith.muli %arg13, %c512_i32 : i32
                      %310 = arith.index_cast %309 : i32 to index
                      %result0_11, %result1_12 = ascvec.loadalign_deintlv %arg10[%310] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %311 = ascvec.and %result1_12, %287, %285 : <ui8>, <ui8> -> <ui8>
                      %312 = ascvec.or %308, %311, %285 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg11[%290], %312, %285 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
                } else {
                  %282 = arith.cmpi sge, %280, %c13568_i64 : i64
                  scf.if %282 {
                    ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %283 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                    %284 = ascscalar.bitcast_memref %129 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                    ascvec.vf(%129, %283, %284, %alloc_4, %276 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                    ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<?xui8, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32):
                      %c512_i32 = arith.constant 512 : i32
                      %c1_i32_9 = arith.constant 1 : i32
                      %c0_i32_10 = arith.constant 0 : i32
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
                      %285 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                      %286 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                      %287 = ascvec.duplicate_scalar %c255_i16, %285 : i16 -> <ui16>
                      %288 = ascvec.duplicate_scalar %c-128_i8, %286 : i8 -> <ui8>
                      %289 = ascvec.arange %c0_i8 : i8 -> <ui8>
                      %290 = ascvec.shr %289, 1, %286 : <ui8> -> <ui8>
                      %291 = ascvec.shr %290, 2, %286 : <ui8> -> <ui8>
                      %292 = ascvec.shl %291, 2, %286 : <ui8> -> <ui8>
                      %293 = ascvec.duplicate_scalar %c1_i8, %286 : i8 -> <ui8>
                      %294 = ascvec.and %289, %293, %286 : <ui8>, <ui8> -> <ui8>
                      %295 = ascvec.add_scalar %290, %c3_i8, %286 : <ui8>, i8
                      %296 = ascvec.add %295, %294, %286 : <ui8>
                      %297 = ascvec.duplicate_scalar %c3_i8, %286 : i8 -> <ui8>
                      %298 = ascvec.and %296, %297, %286 : <ui8>, <ui8> -> <ui8>
                      %299 = ascvec.or %292, %298, %286 : <ui8>, <ui8> -> <ui8>
                      %300 = ascvec.duplicate_scalar %c4095_i16, %285 : i16 -> <ui16>
                      %301 = ascvec.duplicate_scalar %c32767_i16, %285 : i16 -> <ui16>
                      %302 = ascvec.duplicate_scalar %cst, %285 : bf16 -> <bf16>
                      scf.for %arg13 = %c0_i32_10 to %arg12 step %c1_i32_9  : i32 {
                        %303 = arith.muli %arg13, %c256_i32 : i32
                        %304 = arith.index_cast %303 : i32 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg8[%304] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                        %305 = arith.muli %arg13, %c128_i32 : i32
                        %306 = arith.index_cast %305 : i32 to index
                        %307 = ascvec.loadalign %arg9[%306] : memref<?xui8, 1>, index -> <ui8>
                        %308 = ascvec.gather_reg %307, %299 : <ui8>, <ui8> -> <ui8>
                        %309 = ascvec.reinterpret %308 : <ui8> -> <ui16>
                        %310 = ascvec.shr %309, 1, %285 : <ui16> -> <ui16>
                        %311 = ascvec.and %310, %300, %285 : <ui16>, <ui16> -> <ui16>
                        %312 = ascvec.shr %309, 4, %285 : <ui16> -> <ui16>
                        %313 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                        %314 = ascvec.and %313, %301, %285 : <ui16>, <ui16> -> <ui16>
                        %315 = ascvec.cmp_lt_scalar %314, %c14464_i16, %285 : <ui16>, i16 -> !ascvec.mask
                        %316 = ascvec.shr %311, 7, %285 : <ui16> -> <ui16>
                        %317 = ascvec.add %314, %316, %285 : <ui16>
                        %318 = ascvec.shr %317, 5, %285 : <ui16> -> <ui16>
                        %319 = ascvec.reinterpret %314 : <ui16> -> <bf16>
                        %320 = ascvec.mul %319, %302, %315 : <bf16>
                        %321 = ascvec.cast %320, %285 {rounding = 4 : i32} : <bf16> -> <f16>
                        %322 = ascvec.cast %321, %285 {rounding = 4 : i32} : <f16> -> <i16>
                        %323 = ascvec.reinterpret %322 : <i16> -> <ui16>
                        %324 = ascvec.add %323, %311, %285 : <ui16>
                        %325 = ascvec.shr %324, 12, %285 : <ui16> -> <ui16>
                        %326 = ascvec.add_scalar %325, %c448_i16, %285 : <ui16>, i16
                        %327 = ascvec.select %326, %318, %315 : <ui16>
                        %328 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                        %329 = ascvec.and %328, %301, %285 : <ui16>, <ui16> -> <ui16>
                        %330 = ascvec.cmp_lt_scalar %329, %c14464_i16, %285 : <ui16>, i16 -> !ascvec.mask
                        %331 = ascvec.shr %312, 7, %285 : <ui16> -> <ui16>
                        %332 = ascvec.add %329, %331, %285 : <ui16>
                        %333 = ascvec.shr %332, 5, %285 : <ui16> -> <ui16>
                        %334 = ascvec.reinterpret %329 : <ui16> -> <bf16>
                        %335 = ascvec.mul %334, %302, %330 : <bf16>
                        %336 = ascvec.cast %335, %285 {rounding = 4 : i32} : <bf16> -> <f16>
                        %337 = ascvec.cast %336, %285 {rounding = 4 : i32} : <f16> -> <i16>
                        %338 = ascvec.reinterpret %337 : <i16> -> <ui16>
                        %339 = ascvec.add %338, %312, %285 : <ui16>
                        %340 = ascvec.shr %339, 12, %285 : <ui16> -> <ui16>
                        %341 = ascvec.add_scalar %340, %c448_i16, %285 : <ui16>, i16
                        %342 = ascvec.select %341, %333, %330 : <ui16>
                        %343 = ascvec.and %327, %287, %285 : <ui16>, <ui16> -> <ui16>
                        %344 = ascvec.shl %342, 8, %285 : <ui16> -> <ui16>
                        %345 = ascvec.or %343, %344, %285 : <ui16>, <ui16> -> <ui16>
                        %346 = ascvec.reinterpret %345 : <ui16> -> <ui8>
                        %347 = ascvec.add_scalar %346, %c64_i8, %286 : <ui8>, i8
                        %348 = ascvec.min_scalar %347, %c123_i8, %286 : <ui8>, i8
                        %349 = arith.muli %arg13, %c512_i32 : i32
                        %350 = arith.index_cast %349 : i32 to index
                        %result0_11, %result1_12 = ascvec.loadalign_deintlv %arg10[%350] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                        %351 = ascvec.and %result1_12, %288, %286 : <ui8>, <ui8> -> <ui8>
                        %352 = ascvec.or %348, %351, %286 : <ui8>, <ui8> -> <ui8>
                        ascvec.storealign %arg11[%304], %352, %286 : memref<32768xui8, 1>, index, <ui8>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
                  } else {
                    ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %283 = ascscalar.bitcast_memref %129 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                    ascvec.vf(%129, %alloc_7, %283, %277, %alloc_4, %276 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32) {
                    ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<8192xi32, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<?xui16, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32):
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
                      %c4_i32_9 = arith.constant 4 : i32
                      %c2_i32_10 = arith.constant 2 : i32
                      %c1_i32_11 = arith.constant 1 : i32
                      %c7_i32 = arith.constant 7 : i32
                      %c32_i32 = arith.constant 32 : i32
                      %c0_i32_12 = arith.constant 0 : i32
                      %c32767_i16 = arith.constant 32767 : i16
                      %c0_13 = arith.constant 0 : index
                      %c256_i32 = arith.constant 256 : i32
                      %c3_i8 = arith.constant 3 : i8
                      %c-128_i8 = arith.constant -128 : i8
                      %c0_i8 = arith.constant 0 : i8
                      %c8_i32 = arith.constant 8 : i32
                      %284 = ascvec.create_mask : !ascvec.mask
                      %285 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                      %286 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                      %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                      %287 = ascvec.arange %c0_i8 : i8 -> <ui8>
                      %288 = ascvec.duplicate_scalar %c-128_i8, %286 : i8 -> <ui8>
                      %289 = ascvec.shr %287, 5, %286 : <ui8> -> <ui8>
                      %290 = ascvec.duplicate_scalar %c3_i8, %286 : i8 -> <ui8>
                      %291 = ascvec.and %287, %290, %286 : <ui8>, <ui8> -> <ui8>
                      %292 = ascvec.shl %291, 6, %286 : <ui8> -> <ui8>
                      %293 = ascvec.shr %287, 2, %286 : <ui8> -> <ui8>
                      %294 = ascvec.or %292, %293, %286 : <ui8>, <ui8> -> <ui8>
                      scf.for %arg14 = %c0_i32_12 to %arg13 step %c1_i32_11  : i32 {
                        %295 = arith.muli %arg14, %c256_i32 : i32
                        %296 = arith.index_cast %295 : i32 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg8[%296] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                        %297 = ascvec.duplicate_scalar %c32767_i16, %285 : i16 -> <ui16>
                        %298 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                        %299 = ascvec.and %298, %297, %285 : <ui16>, <ui16> -> <ui16>
                        %300 = ascvec.reinterpret %299 : <ui16> -> <bf16>
                        %301 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                        %302 = ascvec.and %301, %297, %285 : <ui16>, <ui16> -> <ui16>
                        %303 = ascvec.reinterpret %302 : <ui16> -> <bf16>
                        %304 = ascvec.cast %300, %285 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %305 = ascvec.cast %300, %285 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %306 = ascvec.cast %303, %285 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %307 = ascvec.cast %303, %285 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %308 = ascvec.arange %c0_i32_12 : i32 -> <ui32>
                        %309 = ascvec.shl %308, 2, %284 : <ui32> -> <ui32>
                        %310 = ascvec.add_scalar %309, %c0_i32_12, %284 : <ui32>, i32
                        %311 = ascvec.shr %310, 3, %284 : <ui32> -> <ui32>
                        %312 = arith.muli %arg14, %c32_i32 : i32
                        %313 = ascvec.add_scalar %311, %312, %284 : <ui32>, i32
                        %314 = ascvec.gather %arg9[%c0_13], %313, %284 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %315 = ascvec.reinterpret %314 : <i32> -> <ui32>
                        %316 = ascvec.duplicate_scalar %c7_i32, %284 : i32 -> <ui32>
                        %317 = ascvec.and %310, %316, %284 : <ui32>, <ui32> -> <ui32>
                        %318 = ascvec.shl %315, 11, %284 : <ui32> -> <ui32>
                        %319 = ascvec.shr %315, 21, %284 : <ui32> -> <ui32>
                        %320 = ascvec.or %318, %319, %284 : <ui32>, <ui32> -> <ui32>
                        %321 = ascvec.shr %320, 3, %284 : <ui32> -> <ui32>
                        %322 = ascvec.shl %320, 29, %284 : <ui32> -> <ui32>
                        %323 = ascvec.or %321, %322, %284 : <ui32>, <ui32> -> <ui32>
                        %324 = ascvec.duplicate_scalar %c1_i32_11, %284 : i32 -> <ui32>
                        %325 = ascvec.and %317, %324, %284 : <ui32>, <ui32> -> <ui32>
                        %326 = ascvec.cmp_ne_scalar %325, %c0_i32_12, %284 : <ui32>, i32 -> !ascvec.mask
                        %327 = ascvec.select %323, %320, %326 : <ui32>
                        %328 = ascvec.shr %327, 8, %284 : <ui32> -> <ui32>
                        %329 = ascvec.shl %327, 24, %284 : <ui32> -> <ui32>
                        %330 = ascvec.or %328, %329, %284 : <ui32>, <ui32> -> <ui32>
                        %331 = ascvec.duplicate_scalar %c2_i32_10, %284 : i32 -> <ui32>
                        %332 = ascvec.and %317, %331, %284 : <ui32>, <ui32> -> <ui32>
                        %333 = ascvec.cmp_ne_scalar %332, %c0_i32_12, %284 : <ui32>, i32 -> !ascvec.mask
                        %334 = ascvec.select %330, %327, %333 : <ui32>
                        %335 = ascvec.shr %334, 16, %284 : <ui32> -> <ui32>
                        %336 = ascvec.shl %334, 16, %284 : <ui32> -> <ui32>
                        %337 = ascvec.or %335, %336, %284 : <ui32>, <ui32> -> <ui32>
                        %338 = ascvec.duplicate_scalar %c4_i32_9, %284 : i32 -> <ui32>
                        %339 = ascvec.and %317, %338, %284 : <ui32>, <ui32> -> <ui32>
                        %340 = ascvec.cmp_ne_scalar %339, %c0_i32_12, %284 : <ui32>, i32 -> !ascvec.mask
                        %341 = ascvec.select %337, %334, %340 : <ui32>
                        %342 = ascvec.duplicate_scalar %c65535_i32, %284 : i32 -> <ui32>
                        %343 = ascvec.and %341, %342, %284 : <ui32>, <ui32> -> <ui32>
                        %344 = ascvec.add_scalar %309, %c1_i32_11, %284 : <ui32>, i32
                        %345 = ascvec.shr %344, 3, %284 : <ui32> -> <ui32>
                        %346 = ascvec.add_scalar %345, %312, %284 : <ui32>, i32
                        %347 = ascvec.gather %arg9[%c0_13], %346, %284 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %348 = ascvec.reinterpret %347 : <i32> -> <ui32>
                        %349 = ascvec.and %344, %316, %284 : <ui32>, <ui32> -> <ui32>
                        %350 = ascvec.shl %348, 11, %284 : <ui32> -> <ui32>
                        %351 = ascvec.shr %348, 21, %284 : <ui32> -> <ui32>
                        %352 = ascvec.or %350, %351, %284 : <ui32>, <ui32> -> <ui32>
                        %353 = ascvec.shr %352, 3, %284 : <ui32> -> <ui32>
                        %354 = ascvec.shl %352, 29, %284 : <ui32> -> <ui32>
                        %355 = ascvec.or %353, %354, %284 : <ui32>, <ui32> -> <ui32>
                        %356 = ascvec.and %349, %324, %284 : <ui32>, <ui32> -> <ui32>
                        %357 = ascvec.cmp_ne_scalar %356, %c0_i32_12, %284 : <ui32>, i32 -> !ascvec.mask
                        %358 = ascvec.select %355, %352, %357 : <ui32>
                        %359 = ascvec.shr %358, 8, %284 : <ui32> -> <ui32>
                        %360 = ascvec.shl %358, 24, %284 : <ui32> -> <ui32>
                        %361 = ascvec.or %359, %360, %284 : <ui32>, <ui32> -> <ui32>
                        %362 = ascvec.and %349, %331, %284 : <ui32>, <ui32> -> <ui32>
                        %363 = ascvec.cmp_ne_scalar %362, %c0_i32_12, %284 : <ui32>, i32 -> !ascvec.mask
                        %364 = ascvec.select %361, %358, %363 : <ui32>
                        %365 = ascvec.shr %364, 16, %284 : <ui32> -> <ui32>
                        %366 = ascvec.shl %364, 16, %284 : <ui32> -> <ui32>
                        %367 = ascvec.or %365, %366, %284 : <ui32>, <ui32> -> <ui32>
                        %368 = ascvec.and %349, %338, %284 : <ui32>, <ui32> -> <ui32>
                        %369 = ascvec.cmp_ne_scalar %368, %c0_i32_12, %284 : <ui32>, i32 -> !ascvec.mask
                        %370 = ascvec.select %367, %364, %369 : <ui32>
                        %371 = ascvec.and %370, %342, %284 : <ui32>, <ui32> -> <ui32>
                        %372 = ascvec.add_scalar %309, %c2_i32_10, %284 : <ui32>, i32
                        %373 = ascvec.shr %372, 3, %284 : <ui32> -> <ui32>
                        %374 = ascvec.add_scalar %373, %312, %284 : <ui32>, i32
                        %375 = ascvec.gather %arg9[%c0_13], %374, %284 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %376 = ascvec.reinterpret %375 : <i32> -> <ui32>
                        %377 = ascvec.and %372, %316, %284 : <ui32>, <ui32> -> <ui32>
                        %378 = ascvec.shl %376, 11, %284 : <ui32> -> <ui32>
                        %379 = ascvec.shr %376, 21, %284 : <ui32> -> <ui32>
                        %380 = ascvec.or %378, %379, %284 : <ui32>, <ui32> -> <ui32>
                        %381 = ascvec.shr %380, 3, %284 : <ui32> -> <ui32>
                        %382 = ascvec.shl %380, 29, %284 : <ui32> -> <ui32>
                        %383 = ascvec.or %381, %382, %284 : <ui32>, <ui32> -> <ui32>
                        %384 = ascvec.and %377, %324, %284 : <ui32>, <ui32> -> <ui32>
                        %385 = ascvec.cmp_ne_scalar %384, %c0_i32_12, %284 : <ui32>, i32 -> !ascvec.mask
                        %386 = ascvec.select %383, %380, %385 : <ui32>
                        %387 = ascvec.shr %386, 8, %284 : <ui32> -> <ui32>
                        %388 = ascvec.shl %386, 24, %284 : <ui32> -> <ui32>
                        %389 = ascvec.or %387, %388, %284 : <ui32>, <ui32> -> <ui32>
                        %390 = ascvec.and %377, %331, %284 : <ui32>, <ui32> -> <ui32>
                        %391 = ascvec.cmp_ne_scalar %390, %c0_i32_12, %284 : <ui32>, i32 -> !ascvec.mask
                        %392 = ascvec.select %389, %386, %391 : <ui32>
                        %393 = ascvec.shr %392, 16, %284 : <ui32> -> <ui32>
                        %394 = ascvec.shl %392, 16, %284 : <ui32> -> <ui32>
                        %395 = ascvec.or %393, %394, %284 : <ui32>, <ui32> -> <ui32>
                        %396 = ascvec.and %377, %338, %284 : <ui32>, <ui32> -> <ui32>
                        %397 = ascvec.cmp_ne_scalar %396, %c0_i32_12, %284 : <ui32>, i32 -> !ascvec.mask
                        %398 = ascvec.select %395, %392, %397 : <ui32>
                        %399 = ascvec.and %398, %342, %284 : <ui32>, <ui32> -> <ui32>
                        %400 = ascvec.add_scalar %309, %c3_i32, %284 : <ui32>, i32
                        %401 = ascvec.shr %400, 3, %284 : <ui32> -> <ui32>
                        %402 = ascvec.add_scalar %401, %312, %284 : <ui32>, i32
                        %403 = ascvec.gather %arg9[%c0_13], %402, %284 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %404 = ascvec.reinterpret %403 : <i32> -> <ui32>
                        %405 = ascvec.and %400, %316, %284 : <ui32>, <ui32> -> <ui32>
                        %406 = ascvec.shl %404, 11, %284 : <ui32> -> <ui32>
                        %407 = ascvec.shr %404, 21, %284 : <ui32> -> <ui32>
                        %408 = ascvec.or %406, %407, %284 : <ui32>, <ui32> -> <ui32>
                        %409 = ascvec.shr %408, 3, %284 : <ui32> -> <ui32>
                        %410 = ascvec.shl %408, 29, %284 : <ui32> -> <ui32>
                        %411 = ascvec.or %409, %410, %284 : <ui32>, <ui32> -> <ui32>
                        %412 = ascvec.and %405, %324, %284 : <ui32>, <ui32> -> <ui32>
                        %413 = ascvec.cmp_ne_scalar %412, %c0_i32_12, %284 : <ui32>, i32 -> !ascvec.mask
                        %414 = ascvec.select %411, %408, %413 : <ui32>
                        %415 = ascvec.shr %414, 8, %284 : <ui32> -> <ui32>
                        %416 = ascvec.shl %414, 24, %284 : <ui32> -> <ui32>
                        %417 = ascvec.or %415, %416, %284 : <ui32>, <ui32> -> <ui32>
                        %418 = ascvec.and %405, %331, %284 : <ui32>, <ui32> -> <ui32>
                        %419 = ascvec.cmp_ne_scalar %418, %c0_i32_12, %284 : <ui32>, i32 -> !ascvec.mask
                        %420 = ascvec.select %417, %414, %419 : <ui32>
                        %421 = ascvec.shr %420, 16, %284 : <ui32> -> <ui32>
                        %422 = ascvec.shl %420, 16, %284 : <ui32> -> <ui32>
                        %423 = ascvec.or %421, %422, %284 : <ui32>, <ui32> -> <ui32>
                        %424 = ascvec.and %405, %338, %284 : <ui32>, <ui32> -> <ui32>
                        %425 = ascvec.cmp_ne_scalar %424, %c0_i32_12, %284 : <ui32>, i32 -> !ascvec.mask
                        %426 = ascvec.select %423, %420, %425 : <ui32>
                        %427 = ascvec.and %426, %342, %284 : <ui32>, <ui32> -> <ui32>
                        %428 = ascvec.reinterpret %304 : <f32> -> <ui32>
                        %429 = ascvec.reinterpret %428 : <ui32> -> <f32>
                        %430 = ascvec.max_scalar %429, %cst, %284 : <f32>, f32
                        %431 = ascvec.add %429, %430, %284 : <f32>
                        %432 = ascvec.reinterpret %431 : <f32> -> <ui32>
                        %433 = ascvec.shr %432, 5, %284 : <ui32> -> <ui32>
                        %434 = ascvec.add %433, %343, %284 : <ui32>
                        %435 = ascvec.reinterpret %306 : <f32> -> <ui32>
                        %436 = ascvec.reinterpret %435 : <ui32> -> <f32>
                        %437 = ascvec.max_scalar %436, %cst, %284 : <f32>, f32
                        %438 = ascvec.add %436, %437, %284 : <f32>
                        %439 = ascvec.reinterpret %438 : <f32> -> <ui32>
                        %440 = ascvec.shr %439, 5, %284 : <ui32> -> <ui32>
                        %441 = ascvec.add %440, %371, %284 : <ui32>
                        %442 = ascvec.reinterpret %305 : <f32> -> <ui32>
                        %443 = ascvec.reinterpret %442 : <ui32> -> <f32>
                        %444 = ascvec.max_scalar %443, %cst, %284 : <f32>, f32
                        %445 = ascvec.add %443, %444, %284 : <f32>
                        %446 = ascvec.reinterpret %445 : <f32> -> <ui32>
                        %447 = ascvec.shr %446, 5, %284 : <ui32> -> <ui32>
                        %448 = ascvec.add %447, %399, %284 : <ui32>
                        %449 = ascvec.reinterpret %307 : <f32> -> <ui32>
                        %450 = ascvec.reinterpret %449 : <ui32> -> <f32>
                        %451 = ascvec.max_scalar %450, %cst, %284 : <f32>, f32
                        %452 = ascvec.add %450, %451, %284 : <f32>
                        %453 = ascvec.reinterpret %452 : <f32> -> <ui32>
                        %454 = ascvec.shr %453, 5, %284 : <ui32> -> <ui32>
                        %455 = ascvec.add %454, %427, %284 : <ui32>
                        %456 = ascvec.reinterpret %434 : <ui32> -> <ui16>
                        %457 = ascvec.reinterpret %441 : <ui32> -> <ui16>
                        %result0_14, %result1_15 = ascvec.deintlv %456, %457 : <ui16>, <ui16> -> <ui16>, <ui16>
                        %458 = ascvec.reinterpret %448 : <ui32> -> <ui16>
                        %459 = ascvec.reinterpret %455 : <ui32> -> <ui16>
                        %result0_16, %result1_17 = ascvec.deintlv %458, %459 : <ui16>, <ui16> -> <ui16>, <ui16>
                        %460 = ascvec.reinterpret %result1_15 : <ui16> -> <ui8>
                        %461 = ascvec.reinterpret %result1_17 : <ui16> -> <ui8>
                        %result0_18, %result1_19 = ascvec.deintlv %460, %461 : <ui8>, <ui8> -> <ui8>, <ui8>
                        %462 = ascvec.gather_reg %result0_18, %294 : <ui8>, <ui8> -> <ui8>
                        %463 = ascvec.add_scalar %462, %c60_i8, %286 : <ui8>, i8
                        %464 = ascvec.cmp_ne_scalar %463, %c-1_i8, %286 : <ui8>, i8 -> !ascvec.mask
                        %465 = ascvec.min_scalar %463, %c123_i8, %464 : <ui8>, i8
                        %466 = arith.muli %arg14, %c512_i32 : i32
                        %467 = arith.index_cast %466 : i32 to index
                        %result0_20, %result1_21 = ascvec.loadalign_deintlv %arg10[%467] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                        %468 = ascvec.and %result1_21, %288, %286 : <ui8>, <ui8> -> <ui8>
                        %469 = ascvec.or %465, %468, %286 : <ui8>, <ui8> -> <ui8>
                        %470 = ascvec.arange %c0_i16 : i16 -> <ui16>
                        %471 = arith.muli %arg14, %c16_i32 : i32
                        %472 = arith.trunci %471 : i32 to i16
                        %473 = ascvec.add_scalar %470, %472, %285 : <ui16>, i16
                        %474 = ascvec.gather %arg11[%c0_13], %473, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %475 = ascvec.pack %474 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                        %476 = ascvec.gather_reg %475, %289 : <ui8>, <ui8> -> <ui8>
                        %477 = ascvec.cmp_eq_scalar %476, %c-1_i8, %286 : <ui8>, i8 -> !ascvec.mask
                        %478 = ascvec.duplicate_scalar %c127_i8, %477 merge %469 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                        ascvec.storealign %arg12[%296], %478, %286 : memref<32768xui8, 1>, index, <ui8>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
                  }
                }
              }
            } else {
              %215 = arith.cmpi sge, %94, %c1024_i64 : i64
              %216 = arith.remsi %94, %c1024_i64 : i64
              %217 = arith.cmpi eq, %216, %c0_i64 : i64
              %218 = arith.select %215, %217, %false : i1
              scf.if %218 {
                %219 = arith.divsi %94, %c256_i64 : i64
                %220 = arith.addi %94, %c1023_i64 : i64
                %221 = arith.remsi %220, %c1024_i64 : i64
                %222 = arith.cmpi ne, %221, %c0_i64 : i64
                %223 = arith.cmpi slt, %221, %c0_i64 : i64
                %224 = arith.andi %222, %223 : i1
                %225 = arith.divsi %220, %c1024_i64 : i64
                %226 = arith.extui %224 : i1 to i64
                %227 = arith.subi %225, %226 : i64
                %228 = arith.trunci %arg4 : i64 to i32
                %229 = arith.extui %228 : i32 to i64
                %230 = arith.subi %arg4, %229 : i64
                %231 = arith.remsi %230, %c4294967296_i64 : i64
                %232 = arith.cmpi ne, %231, %c0_i64 : i64
                %233 = arith.cmpi slt, %231, %c0_i64 : i64
                %234 = arith.andi %232, %233 : i1
                %235 = arith.divsi %230, %c4294967296_i64 : i64
                %236 = arith.extui %234 : i1 to i64
                %237 = arith.subi %235, %236 : i64
                %238 = arith.trunci %237 : i64 to i32
                %239 = arith.divui %207, %c16_i64 : i64
                %240 = arith.trunci %239 : i64 to i32
                %241 = arith.extui %240 : i32 to i64
                %242 = arith.subi %239, %241 : i64
                %243 = arith.remsi %242, %c4294967296_i64 : i64
                %244 = arith.cmpi ne, %243, %c0_i64 : i64
                %245 = arith.cmpi slt, %243, %c0_i64 : i64
                %246 = arith.andi %244, %245 : i1
                %247 = arith.divsi %242, %c4294967296_i64 : i64
                %248 = arith.extui %246 : i1 to i64
                %249 = arith.subi %247, %248 : i64
                %250 = arith.trunci %249 : i64 to i32
                %251 = arith.trunci %arg5 : i64 to i32
                %252 = arith.extui %251 : i32 to i64
                %253 = arith.subi %arg5, %252 : i64
                %254 = arith.remsi %253, %c4294967296_i64 : i64
                %255 = arith.cmpi ne, %254, %c0_i64 : i64
                %256 = arith.cmpi slt, %254, %c0_i64 : i64
                %257 = arith.andi %255, %256 : i1
                %258 = arith.divsi %253, %c4294967296_i64 : i64
                %259 = arith.extui %257 : i1 to i64
                %260 = arith.subi %258, %259 : i64
                %261 = arith.trunci %260 : i64 to i32
                %262 = arith.divui %251, %c4_i32 : i32
                %263 = arith.remui %261, %c4_i32 : i32
                %264 = arith.muli %263, %c1073741824_i32 : i32
                %265 = arith.addi %262, %264 : i32
                %266 = arith.divui %261, %c4_i32 : i32
                %267 = arith.muli %227, %c64_i64 : i64
                %268 = arith.divui %239, %c16384_i64 : i64
                %269 = arith.addi %239, %267 : i64
                %270 = arith.subi %269, %c1_i64 : i64
                %271 = arith.divui %270, %c16384_i64 : i64
                %272 = arith.cmpi eq, %268, %271 : i64
                scf.if %272 {
                  %282 = arith.trunci %268 : i64 to i32
                  %283 = arith.extui %282 : i32 to i64
                  %284 = arith.subi %268, %283 : i64
                  %285 = arith.remsi %284, %c4294967296_i64 : i64
                  %286 = arith.cmpi ne, %285, %c0_i64 : i64
                  %287 = arith.cmpi slt, %285, %c0_i64 : i64
                  %288 = arith.andi %286, %287 : i1
                  %289 = arith.divsi %284, %c4294967296_i64 : i64
                  %290 = arith.extui %288 : i1 to i64
                  %291 = arith.subi %289, %290 : i64
                  %292 = arith.trunci %291 : i64 to i32
                  %293 = arith.addi %265, %282 : i32
                  %294 = arith.addi %266, %292 : i32
                  %295 = arith.cmpi ult, %293, %265 : i32
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
                  %311 = arith.xori %297, %238 : i32
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
                  %325 = arith.addi %228, %c-1640531527_i32 : i32
                  %326 = arith.addi %238, %c-1150833019_i32 : i32
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
                  %342 = arith.xori %310, %228 : i32
                  %343 = arith.addi %228, %c1013904242_i32 : i32
                  %344 = arith.xori %314, %343 : i32
                  %345 = arith.addi %238, %c1993301258_i32 : i32
                  %346 = arith.xori %341, %345 : i32
                  ascsimt.vf(%240, %342, %326, %344, %346, %228, %238, %331, %alloc_7, %267 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: memref<8192xi32, 1>, %arg17: i64):
                    %c256_i64_9 = arith.constant 256 : i64
                    %c3_i64_10 = arith.constant 3 : i64
                    %c2_i64_11 = arith.constant 2 : i64
                    %c1_i64_12 = arith.constant 1 : i64
                    %c4_i64_13 = arith.constant 4 : i64
                    %c1_i32_14 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c3_i32 = arith.constant 3 : i32
                    %c-1150833019_i32_15 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_16 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %347 = ascsimt.thread_idx {ascsimt.scope = 44 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %348 = arith.extsi %347 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg18 = %348 to %arg17 step %c256_i64_9  : i64 {
                      %349 = arith.trunci %arg18 : i64 to i32
                      %350 = arith.addi %arg8, %349 : i32
                      %351 = arith.remui %350, %c16384_i32 : i32
                      %352 = arith.xori %351, %arg9 : i32
                      %353 = arith.muli %352, %c-766435501_i32 : i32
                      %354 = "ascsimt.bitcast"(%352) : (i32) -> ui32
                      %355 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %356 = ascsimt.math_binary %354, %355, "mulhi" : ui32, ui32 -> ui32
                      %357 = "ascsimt.bitcast"(%356) : (ui32) -> i32
                      %358 = arith.xori %357, %arg10 : i32
                      %359 = arith.muli %358, %c-845247145_i32 : i32
                      %360 = "ascsimt.bitcast"(%358) : (i32) -> ui32
                      %361 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %362 = ascsimt.math_binary %360, %361, "mulhi" : ui32, ui32 -> ui32
                      %363 = "ascsimt.bitcast"(%362) : (ui32) -> i32
                      %364 = arith.xori %363, %arg11 : i32
                      %365 = arith.xori %353, %arg12 : i32
                      %366:4 = scf.for %arg19 = %c3_i32 to %c10_i32 step %c1_i32_14 iter_args(%arg20 = %364, %arg21 = %359, %arg22 = %365, %arg23 = %arg15) -> (i32, i32, i32, i32)  : i32 {
                        %375 = arith.muli %arg20, %c-766435501_i32 : i32
                        %376 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                        %377 = ascsimt.math_binary %376, %355, "mulhi" : ui32, ui32 -> ui32
                        %378 = "ascsimt.bitcast"(%377) : (ui32) -> i32
                        %379 = arith.muli %arg22, %c-845247145_i32 : i32
                        %380 = "ascsimt.bitcast"(%arg22) : (i32) -> ui32
                        %381 = ascsimt.math_binary %380, %361, "mulhi" : ui32, ui32 -> ui32
                        %382 = "ascsimt.bitcast"(%381) : (ui32) -> i32
                        %383 = arith.muli %arg19, %c-1640531527_i32_16 : i32
                        %384 = arith.addi %arg13, %383 : i32
                        %385 = arith.muli %arg19, %c-1150833019_i32_15 : i32
                        %386 = arith.addi %arg14, %385 : i32
                        %387 = arith.xori %382, %arg21 : i32
                        %388 = arith.xori %387, %384 : i32
                        %389 = arith.xori %378, %arg23 : i32
                        %390 = arith.xori %389, %386 : i32
                        scf.yield %388, %379, %390, %375 : i32, i32, i32, i32
                      }
                      %367 = arith.muli %arg18, %c4_i64_13 : i64
                      %368 = arith.index_cast %367 : i64 to index
                      ascsimt.store %arg16[%368], %366#0 : memref<8192xi32, 1>, i32
                      %369 = arith.addi %367, %c1_i64_12 : i64
                      %370 = arith.index_cast %369 : i64 to index
                      ascsimt.store %arg16[%370], %366#1 : memref<8192xi32, 1>, i32
                      %371 = arith.addi %367, %c2_i64_11 : i64
                      %372 = arith.index_cast %371 : i64 to index
                      ascsimt.store %arg16[%372], %366#2 : memref<8192xi32, 1>, i32
                      %373 = arith.addi %367, %c3_i64_10 : i64
                      %374 = arith.index_cast %373 : i64 to index
                      ascsimt.store %arg16[%374], %366#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%240, %250, %265, %266, %228, %238, %alloc_7, %267 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: memref<8192xi32, 1>, %arg15: i64):
                    %c256_i64_9 = arith.constant 256 : i64
                    %c3_i64_10 = arith.constant 3 : i64
                    %c2_i64_11 = arith.constant 2 : i64
                    %c1_i64_12 = arith.constant 1 : i64
                    %c4_i64_13 = arith.constant 4 : i64
                    %c1_i32_14 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c0_i32_15 = arith.constant 0 : i32
                    %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c262144_i32 = arith.constant 262144 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %282 = ascsimt.thread_idx {ascsimt.scope = 45 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %283 = arith.extsi %282 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg16 = %283 to %arg15 step %c256_i64_9  : i64 {
                      %284 = arith.trunci %arg16 : i64 to i32
                      %285 = arith.addi %arg8, %284 : i32
                      %286 = arith.cmpi ult, %285, %arg8 : i32
                      %287 = arith.extui %286 : i1 to i32
                      %288 = arith.addi %arg9, %287 : i32
                      %289 = arith.divui %285, %c16384_i32 : i32
                      %290 = arith.remui %288, %c16384_i32 : i32
                      %291 = arith.muli %290, %c262144_i32 : i32
                      %292 = arith.addi %289, %291 : i32
                      %293 = arith.divui %288, %c16384_i32 : i32
                      %294 = arith.addi %arg10, %292 : i32
                      %295 = arith.addi %arg11, %293 : i32
                      %296 = arith.cmpi ult, %294, %arg10 : i32
                      %297 = arith.extui %296 : i1 to i32
                      %298 = arith.addi %295, %297 : i32
                      %299 = arith.remui %285, %c16384_i32 : i32
                      %300:4 = scf.for %arg17 = %c0_i32_15 to %c10_i32 step %c1_i32_14 iter_args(%arg18 = %c0_i32_15, %arg19 = %299, %arg20 = %294, %arg21 = %298) -> (i32, i32, i32, i32)  : i32 {
                        %309 = arith.muli %arg18, %c-766435501_i32 : i32
                        %310 = "ascsimt.bitcast"(%arg18) : (i32) -> ui32
                        %311 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %312 = ascsimt.math_binary %310, %311, "mulhi" : ui32, ui32 -> ui32
                        %313 = "ascsimt.bitcast"(%312) : (ui32) -> i32
                        %314 = arith.muli %arg20, %c-845247145_i32 : i32
                        %315 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                        %316 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %317 = ascsimt.math_binary %315, %316, "mulhi" : ui32, ui32 -> ui32
                        %318 = "ascsimt.bitcast"(%317) : (ui32) -> i32
                        %319 = arith.muli %arg17, %c-1640531527_i32_17 : i32
                        %320 = arith.addi %arg12, %319 : i32
                        %321 = arith.muli %arg17, %c-1150833019_i32_16 : i32
                        %322 = arith.addi %arg13, %321 : i32
                        %323 = arith.xori %318, %arg19 : i32
                        %324 = arith.xori %323, %320 : i32
                        %325 = arith.xori %313, %arg21 : i32
                        %326 = arith.xori %325, %322 : i32
                        scf.yield %324, %314, %326, %309 : i32, i32, i32, i32
                      }
                      %301 = arith.muli %arg16, %c4_i64_13 : i64
                      %302 = arith.index_cast %301 : i64 to index
                      ascsimt.store %arg14[%302], %300#0 : memref<8192xi32, 1>, i32
                      %303 = arith.addi %301, %c1_i64_12 : i64
                      %304 = arith.index_cast %303 : i64 to index
                      ascsimt.store %arg14[%304], %300#1 : memref<8192xi32, 1>, i32
                      %305 = arith.addi %301, %c2_i64_11 : i64
                      %306 = arith.index_cast %305 : i64 to index
                      ascsimt.store %arg14[%306], %300#2 : memref<8192xi32, 1>, i32
                      %307 = arith.addi %301, %c3_i64_10 : i64
                      %308 = arith.index_cast %307 : i64 to index
                      ascsimt.store %arg14[%308], %300#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %273 = arith.trunci %27 : i64 to i32
                %274 = arith.divsi %273, %c2_i32 : i32
                %275 = arith.trunci %274 : i32 to i8
                %276 = arith.trunci %219 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %277 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%275, %alloc_5, %129, %277, %alloc_8, %276 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, memref<2048xui16, 1>, i32) {
                ^bb0(%arg8: i8, %arg9: memref<2048xui8, 1>, %arg10: memref<32768xbf16, 1>, %arg11: memref<?xui16, 1>, %arg12: memref<2048xui16, 1>, %arg13: i32):
                  %c16_9 = arith.constant 16 : index
                  %c4_i32_10 = arith.constant 4 : i32
                  %c0_i32_11 = arith.constant 0 : i32
                  %c3_i32 = arith.constant 3 : i32
                  %c2_i32_12 = arith.constant 2 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c1_i32_13 = arith.constant 1 : i32
                  %c0_14 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %282 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %283 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %284 = ascvec.duplicate_scalar %c32767_i16, %282 : i16 -> <ui16>
                  %285 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %286 = ascvec.shr %285, 4, %282 : <ui16> -> <ui16>
                  %287 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %288 = ascvec.duplicate_scalar %c1_i8, %283 : i8 -> <ui8>
                  %289 = ascvec.and %287, %288, %283 : <ui8>, <ui8> -> <ui8>
                  %290 = ascvec.cmp_lt_scalar %287, %c16_i8, %283 : <ui8>, i8 -> !ascvec.mask
                  %291 = ascvec.cmp_lt_scalar %289, %arg8, %283 : <ui8>, i8 -> !ascvec.mask
                  %292 = ascvec.mask_and %290, %291, %283
                  ascvec.store_unalign_begin %arg9 : memref<2048xui8, 1>
                  %mask_15, %remaining_out_16 = ascvec.update_mask %c1_i32_13 elem_bits 16 : !ascvec.mask
                  %293 = ascvec.duplicate_scalar %c-1_i16, %mask_15 : i16 -> <ui16>
                  %294 = scf.for %arg14 = %c0_i32_11 to %arg13 step %c4_i32_10 iter_args(%arg15 = %293) -> (!ascvec.vreg<ui16>)  : i32 {
                    %295 = ascvec.duplicate_scalar %c-1_i16, %282 : i16 -> <ui16>
                    %296 = ascvec.duplicate_scalar %c0_i16, %282 : i16 -> <ui16>
                    %297 = arith.muli %arg14, %c256_i32 : i32
                    %298 = arith.index_cast %297 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg10[%298] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %299 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %300 = ascvec.and %299, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %301 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %302 = ascvec.and %301, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %303 = ascvec.max %300, %302, %282 : <ui16>
                    %304 = ascvec.reduce_max_datablock %303, %282 : <ui16>
                    %305 = ascvec.shr %304, 7, %282 : <ui16> -> <ui16>
                    %306 = ascvec.cmp_ge_scalar %305, %c255_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %307 = ascvec.max_scalar %305, %c15_i16, %282 : <ui16>, i16
                    %308 = ascvec.add_scalar %307, %c-15_i16, %282 : <ui16>, i16
                    %309 = ascvec.duplicate_scalar %c255_i16, %306 merge %308 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %310 = ascvec.duplicate_scalar %c254_i16, %282 : i16 -> <ui16>
                    %311 = ascvec.sub %310, %309, %282 : <ui16>
                    %312 = ascvec.cmp_ne_scalar %309, %c0_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %313 = ascvec.shl %311, 7, %312 : <ui16> -> <ui16>
                    %314 = ascvec.max %296, %309, %mask : <ui16>
                    %315 = ascvec.gather_reg %313, %286 : <ui16>, <ui16> -> <ui16>
                    %316 = ascvec.reinterpret %315 : <ui16> -> <bf16>
                    %317 = ascvec.mul %result0, %316, %282 : <bf16>
                    %318 = ascvec.mul %result1, %316, %282 : <bf16>
                    ascvec.storealign_intlv %arg10[%298], %317, %318 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %319 = ascvec.reinterpret %317 : <bf16> -> <ui16>
                    %320 = ascvec.and %319, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %321 = ascvec.reinterpret %318 : <bf16> -> <ui16>
                    %322 = ascvec.and %321, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %323 = ascvec.min %320, %322, %282 : <ui16>
                    %324 = ascvec.min %295, %323, %282 : <ui16>
                    %325 = arith.muli %arg14, %c16_i32 : i32
                    %326 = arith.index_cast %325 : i32 to index
                    ascvec.storealign %arg11[%326], %309, %mask : memref<?xui16, 1>, index, <ui16>
                    %327 = ascvec.reinterpret %309 : <ui16> -> <ui8>
                    %328 = ascvec.squeeze_store_reg %327, %292 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_14], %328 : memref<2048xui8, 1>, <ui8>
                    %329 = arith.addi %arg14, %c1_i32_13 : i32
                    %330 = arith.muli %329, %c256_i32 : i32
                    %331 = arith.index_cast %330 : i32 to index
                    %result0_18, %result1_19 = ascvec.loadalign_deintlv %arg10[%331] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %332 = ascvec.reinterpret %result0_18 : <bf16> -> <ui16>
                    %333 = ascvec.and %332, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %334 = ascvec.reinterpret %result1_19 : <bf16> -> <ui16>
                    %335 = ascvec.and %334, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %336 = ascvec.max %333, %335, %282 : <ui16>
                    %337 = ascvec.reduce_max_datablock %336, %282 : <ui16>
                    %338 = ascvec.shr %337, 7, %282 : <ui16> -> <ui16>
                    %339 = ascvec.cmp_ge_scalar %338, %c255_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %340 = ascvec.max_scalar %338, %c15_i16, %282 : <ui16>, i16
                    %341 = ascvec.add_scalar %340, %c-15_i16, %282 : <ui16>, i16
                    %342 = ascvec.duplicate_scalar %c255_i16, %339 merge %341 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %343 = ascvec.sub %310, %342, %282 : <ui16>
                    %344 = ascvec.cmp_ne_scalar %342, %c0_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %345 = ascvec.shl %343, 7, %344 : <ui16> -> <ui16>
                    %346 = ascvec.max %314, %342, %mask : <ui16>
                    %347 = ascvec.gather_reg %345, %286 : <ui16>, <ui16> -> <ui16>
                    %348 = ascvec.reinterpret %347 : <ui16> -> <bf16>
                    %349 = ascvec.mul %result0_18, %348, %282 : <bf16>
                    %350 = ascvec.mul %result1_19, %348, %282 : <bf16>
                    ascvec.storealign_intlv %arg10[%331], %349, %350 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %351 = ascvec.reinterpret %349 : <bf16> -> <ui16>
                    %352 = ascvec.and %351, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %353 = ascvec.reinterpret %350 : <bf16> -> <ui16>
                    %354 = ascvec.and %353, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %355 = ascvec.min %352, %354, %282 : <ui16>
                    %356 = ascvec.min %324, %355, %282 : <ui16>
                    %357 = arith.muli %329, %c16_i32 : i32
                    %358 = arith.index_cast %357 : i32 to index
                    ascvec.storealign %arg11[%358], %342, %mask : memref<?xui16, 1>, index, <ui16>
                    %359 = ascvec.reinterpret %342 : <ui16> -> <ui8>
                    %360 = ascvec.squeeze_store_reg %359, %292 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_14], %360 : memref<2048xui8, 1>, <ui8>
                    %361 = arith.addi %arg14, %c2_i32_12 : i32
                    %362 = arith.muli %361, %c256_i32 : i32
                    %363 = arith.index_cast %362 : i32 to index
                    %result0_20, %result1_21 = ascvec.loadalign_deintlv %arg10[%363] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %364 = ascvec.reinterpret %result0_20 : <bf16> -> <ui16>
                    %365 = ascvec.and %364, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %366 = ascvec.reinterpret %result1_21 : <bf16> -> <ui16>
                    %367 = ascvec.and %366, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %368 = ascvec.max %365, %367, %282 : <ui16>
                    %369 = ascvec.reduce_max_datablock %368, %282 : <ui16>
                    %370 = ascvec.shr %369, 7, %282 : <ui16> -> <ui16>
                    %371 = ascvec.cmp_ge_scalar %370, %c255_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %372 = ascvec.max_scalar %370, %c15_i16, %282 : <ui16>, i16
                    %373 = ascvec.add_scalar %372, %c-15_i16, %282 : <ui16>, i16
                    %374 = ascvec.duplicate_scalar %c255_i16, %371 merge %373 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %375 = ascvec.sub %310, %374, %282 : <ui16>
                    %376 = ascvec.cmp_ne_scalar %374, %c0_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %377 = ascvec.shl %375, 7, %376 : <ui16> -> <ui16>
                    %378 = ascvec.max %346, %374, %mask : <ui16>
                    %379 = ascvec.gather_reg %377, %286 : <ui16>, <ui16> -> <ui16>
                    %380 = ascvec.reinterpret %379 : <ui16> -> <bf16>
                    %381 = ascvec.mul %result0_20, %380, %282 : <bf16>
                    %382 = ascvec.mul %result1_21, %380, %282 : <bf16>
                    ascvec.storealign_intlv %arg10[%363], %381, %382 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %383 = ascvec.reinterpret %381 : <bf16> -> <ui16>
                    %384 = ascvec.and %383, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %385 = ascvec.reinterpret %382 : <bf16> -> <ui16>
                    %386 = ascvec.and %385, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %387 = ascvec.min %384, %386, %282 : <ui16>
                    %388 = ascvec.min %356, %387, %282 : <ui16>
                    %389 = arith.muli %361, %c16_i32 : i32
                    %390 = arith.index_cast %389 : i32 to index
                    ascvec.storealign %arg11[%390], %374, %mask : memref<?xui16, 1>, index, <ui16>
                    %391 = ascvec.reinterpret %374 : <ui16> -> <ui8>
                    %392 = ascvec.squeeze_store_reg %391, %292 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_14], %392 : memref<2048xui8, 1>, <ui8>
                    %393 = arith.addi %arg14, %c3_i32 : i32
                    %394 = arith.muli %393, %c256_i32 : i32
                    %395 = arith.index_cast %394 : i32 to index
                    %result0_22, %result1_23 = ascvec.loadalign_deintlv %arg10[%395] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %396 = ascvec.reinterpret %result0_22 : <bf16> -> <ui16>
                    %397 = ascvec.and %396, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %398 = ascvec.reinterpret %result1_23 : <bf16> -> <ui16>
                    %399 = ascvec.and %398, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %400 = ascvec.max %397, %399, %282 : <ui16>
                    %401 = ascvec.reduce_max_datablock %400, %282 : <ui16>
                    %402 = ascvec.shr %401, 7, %282 : <ui16> -> <ui16>
                    %403 = ascvec.cmp_ge_scalar %402, %c255_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %404 = ascvec.max_scalar %402, %c15_i16, %282 : <ui16>, i16
                    %405 = ascvec.add_scalar %404, %c-15_i16, %282 : <ui16>, i16
                    %406 = ascvec.duplicate_scalar %c255_i16, %403 merge %405 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %407 = ascvec.sub %310, %406, %282 : <ui16>
                    %408 = ascvec.cmp_ne_scalar %406, %c0_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %409 = ascvec.shl %407, 7, %408 : <ui16> -> <ui16>
                    %410 = ascvec.max %378, %406, %mask : <ui16>
                    %411 = ascvec.gather_reg %409, %286 : <ui16>, <ui16> -> <ui16>
                    %412 = ascvec.reinterpret %411 : <ui16> -> <bf16>
                    %413 = ascvec.mul %result0_22, %412, %282 : <bf16>
                    %414 = ascvec.mul %result1_23, %412, %282 : <bf16>
                    ascvec.storealign_intlv %arg10[%395], %413, %414 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %415 = ascvec.reinterpret %413 : <bf16> -> <ui16>
                    %416 = ascvec.and %415, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %417 = ascvec.reinterpret %414 : <bf16> -> <ui16>
                    %418 = ascvec.and %417, %284, %282 : <ui16>, <ui16> -> <ui16>
                    %419 = ascvec.min %416, %418, %282 : <ui16>
                    %420 = ascvec.min %388, %419, %282 : <ui16>
                    %421 = arith.muli %393, %c16_i32 : i32
                    %422 = arith.index_cast %421 : i32 to index
                    ascvec.storealign %arg11[%422], %406, %mask : memref<?xui16, 1>, index, <ui16>
                    %423 = ascvec.reinterpret %406 : <ui16> -> <ui8>
                    %424 = ascvec.squeeze_store_reg %423, %292 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_14], %424 : memref<2048xui8, 1>, <ui8>
                    %425 = ascvec.reduce_min %420, %282 : <ui16>
                    %426 = ascvec.reduce_max %410, %mask : <ui16>
                    %427 = ascvec.cmp_eq_scalar %426, %c255_i16, %282 : <ui16>, i16 -> !ascvec.mask
                    %428 = ascvec.duplicate_scalar %c0_i16, %427 merge %425 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.storealign_1st %arg12[%326], %428 : memref<2048xui16, 1>, index, <ui16>
                    %429 = ascvec.min %arg15, %428, %mask_15 : <ui16>
                    scf.yield %429 : !ascvec.vreg<ui16>
                  }
                  %c0_17 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg14 = %c0_17 to %c1 step %c1 {
                    ascvec.store_unalign_post %arg9[%c0_14] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg12[%c16_9], %294 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %278 = memref.load %alloc_8[%c16] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %279 = cannir.scalar_storage_cast %278 : ui16 -> i16
                %280 = arith.extui %279 : i16 to i64
                %281 = arith.cmpi sge, %280, %c14464_i64 : i64
                scf.if %281 {
                  ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %282 = ascscalar.bitcast_memref %129 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%129, %alloc_7, %282, %alloc_4, %276 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg8: memref<32768xbf16, 1>, %arg9: memref<8192xi32, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<32768xui8, 1>, %arg12: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_9 = arith.constant 1 : i32
                    %c0_i32_10 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c64_i32 = arith.constant 64 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c1_i16 = arith.constant 1 : i16
                    %c0_i16 = arith.constant 0 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %283 = ascvec.create_mask : !ascvec.mask
                    %284 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %285 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %286 = ascvec.duplicate_scalar %c255_i16, %284 : i16 -> <ui16>
                    %287 = ascvec.duplicate_scalar %c-128_i8, %285 : i8 -> <ui8>
                    %288 = ascvec.arange %c0_i16 : i16 -> <ui16>
                    %289 = ascvec.duplicate_scalar %c1_i16, %284 : i16 -> <ui16>
                    %290 = ascvec.xor %288, %289, %284 : <ui16>, <ui16> -> <ui16>
                    scf.for %arg13 = %c0_i32_10 to %arg12 step %c1_i32_9  : i32 {
                      %291 = arith.muli %arg13, %c256_i32 : i32
                      %292 = arith.index_cast %291 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg8[%292] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %293 = arith.muli %arg13, %c64_i32 : i32
                      %294 = arith.index_cast %293 : i32 to index
                      %295 = ascvec.loadalign %arg9[%294] : memref<8192xi32, 1>, index -> <i32>
                      %296 = ascvec.reinterpret %295 : <i32> -> <ui16>
                      %297 = ascvec.reinterpret %296 : <ui16> -> <ui32>
                      %298 = ascvec.shl %297, 8, %283 : <ui32> -> <ui32>
                      %299 = ascvec.shr %297, 24, %283 : <ui32> -> <ui32>
                      %300 = ascvec.or %298, %299, %283 : <ui32>, <ui32> -> <ui32>
                      %301 = ascvec.reinterpret %300 : <ui32> -> <ui16>
                      %302 = ascvec.gather_reg %296, %290 : <ui16>, <ui16> -> <ui16>
                      %303 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %304 = ascvec.shr %302, 11, %284 : <ui16> -> <ui16>
                      %305 = ascvec.add %303, %304, %284 : <ui16>
                      %306 = ascvec.shr %305, 5, %284 : <ui16> -> <ui16>
                      %307 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %308 = ascvec.shr %301, 11, %284 : <ui16> -> <ui16>
                      %309 = ascvec.add %307, %308, %284 : <ui16>
                      %310 = ascvec.shr %309, 5, %284 : <ui16> -> <ui16>
                      %311 = ascvec.and %306, %286, %284 : <ui16>, <ui16> -> <ui16>
                      %312 = ascvec.shl %310, 8, %284 : <ui16> -> <ui16>
                      %313 = ascvec.or %311, %312, %284 : <ui16>, <ui16> -> <ui16>
                      %314 = ascvec.reinterpret %313 : <ui16> -> <ui8>
                      %315 = ascvec.add_scalar %314, %c64_i8, %285 : <ui8>, i8
                      %316 = ascvec.min_scalar %315, %c123_i8, %285 : <ui8>, i8
                      %317 = arith.muli %arg13, %c512_i32 : i32
                      %318 = arith.index_cast %317 : i32 to index
                      %result0_11, %result1_12 = ascvec.loadalign_deintlv %arg10[%318] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %319 = ascvec.and %result1_12, %287, %285 : <ui8>, <ui8> -> <ui8>
                      %320 = ascvec.or %316, %319, %285 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg11[%292], %320, %285 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
                } else {
                  scf.for %arg8 = %c0_i64 to %219 step %c16_i64  : i64 {
                    %282 = arith.addi %arg8, %c16_i64 : i64
                    %283 = arith.cmpi sgt, %282, %219 : i64
                    %284 = arith.select %283, %219, %282 : i64
                    %285 = arith.muli %arg8, %c16_i64 : i64
                    ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %286 = arith.index_cast %285 : i64 to index
                    %287 = memref.load %alloc_8[%286] : memref<2048xui16, 1>
                    ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %288 = cannir.scalar_storage_cast %287 : ui16 -> i16
                    %289 = arith.extui %288 : i16 to i64
                    %290 = arith.addi %arg8, %c4_i64 : i64
                    %291 = scf.for %arg9 = %290 to %284 step %c4_i64 iter_args(%arg10 = %289) -> (i64)  : i64 {
                      %293 = arith.muli %arg9, %c16_i64 : i64
                      ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %294 = arith.index_cast %293 : i64 to index
                      %295 = memref.load %alloc_8[%294] : memref<2048xui16, 1>
                      ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %296 = cannir.scalar_storage_cast %295 : ui16 -> i16
                      %297 = arith.extui %296 : i16 to i64
                      %298 = arith.cmpi slt, %297, %arg10 : i64
                      %299 = arith.select %298, %297, %arg10 : i64
                      scf.yield %299 : i64
                    }
                    %292 = arith.cmpi sge, %291, %c14464_i64 : i64
                    scf.if %292 {
                      %293 = arith.trunci %arg8 : i64 to i32
                      %294 = arith.trunci %284 : i64 to i32
                      ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %295 = ascscalar.bitcast_memref %129 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                      ascvec.vf(%129, %alloc_7, %295, %alloc_4, %293, %294 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                      ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<8192xi32, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32, %arg14: i32):
                        %c512_i32 = arith.constant 512 : i32
                        %c1_i32_9 = arith.constant 1 : i32
                        %c123_i8 = arith.constant 123 : i8
                        %c64_i8 = arith.constant 64 : i8
                        %c64_i32 = arith.constant 64 : i32
                        %c256_i32 = arith.constant 256 : i32
                        %c1_i16 = arith.constant 1 : i16
                        %c0_i16 = arith.constant 0 : i16
                        %c-128_i8 = arith.constant -128 : i8
                        %c255_i16 = arith.constant 255 : i16
                        %296 = ascvec.create_mask : !ascvec.mask
                        %297 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                        %298 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                        %299 = ascvec.duplicate_scalar %c255_i16, %297 : i16 -> <ui16>
                        %300 = ascvec.duplicate_scalar %c-128_i8, %298 : i8 -> <ui8>
                        %301 = ascvec.arange %c0_i16 : i16 -> <ui16>
                        %302 = ascvec.duplicate_scalar %c1_i16, %297 : i16 -> <ui16>
                        %303 = ascvec.xor %301, %302, %297 : <ui16>, <ui16> -> <ui16>
                        scf.for %arg15 = %arg13 to %arg14 step %c1_i32_9  : i32 {
                          %304 = arith.muli %arg15, %c256_i32 : i32
                          %305 = arith.index_cast %304 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg9[%305] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                          %306 = arith.muli %arg15, %c64_i32 : i32
                          %307 = arith.index_cast %306 : i32 to index
                          %308 = ascvec.loadalign %arg10[%307] : memref<8192xi32, 1>, index -> <i32>
                          %309 = ascvec.reinterpret %308 : <i32> -> <ui16>
                          %310 = ascvec.reinterpret %309 : <ui16> -> <ui32>
                          %311 = ascvec.shl %310, 8, %296 : <ui32> -> <ui32>
                          %312 = ascvec.shr %310, 24, %296 : <ui32> -> <ui32>
                          %313 = ascvec.or %311, %312, %296 : <ui32>, <ui32> -> <ui32>
                          %314 = ascvec.reinterpret %313 : <ui32> -> <ui16>
                          %315 = ascvec.gather_reg %309, %303 : <ui16>, <ui16> -> <ui16>
                          %316 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                          %317 = ascvec.shr %315, 11, %297 : <ui16> -> <ui16>
                          %318 = ascvec.add %316, %317, %297 : <ui16>
                          %319 = ascvec.shr %318, 5, %297 : <ui16> -> <ui16>
                          %320 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                          %321 = ascvec.shr %314, 11, %297 : <ui16> -> <ui16>
                          %322 = ascvec.add %320, %321, %297 : <ui16>
                          %323 = ascvec.shr %322, 5, %297 : <ui16> -> <ui16>
                          %324 = ascvec.and %319, %299, %297 : <ui16>, <ui16> -> <ui16>
                          %325 = ascvec.shl %323, 8, %297 : <ui16> -> <ui16>
                          %326 = ascvec.or %324, %325, %297 : <ui16>, <ui16> -> <ui16>
                          %327 = ascvec.reinterpret %326 : <ui16> -> <ui8>
                          %328 = ascvec.add_scalar %327, %c64_i8, %298 : <ui8>, i8
                          %329 = ascvec.min_scalar %328, %c123_i8, %298 : <ui8>, i8
                          %330 = arith.muli %arg15, %c512_i32 : i32
                          %331 = arith.index_cast %330 : i32 to index
                          %result0_10, %result1_11 = ascvec.loadalign_deintlv %arg11[%331] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                          %332 = ascvec.and %result1_11, %300, %298 : <ui8>, <ui8> -> <ui8>
                          %333 = ascvec.or %329, %332, %298 : <ui8>, <ui8> -> <ui8>
                          ascvec.storealign %arg12[%305], %333, %298 : memref<32768xui8, 1>, index, <ui8>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
                    } else {
                      %293 = arith.cmpi sge, %291, %c13568_i64 : i64
                      scf.if %293 {
                        %294 = arith.trunci %arg8 : i64 to i32
                        %295 = arith.trunci %284 : i64 to i32
                        ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %296 = ascscalar.bitcast_memref %129 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%129, %alloc_7, %296, %alloc_4, %294, %295 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<8192xi32, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32, %arg14: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_9 = arith.constant 1 : i32
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
                          %297 = ascvec.create_mask : !ascvec.mask
                          %298 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %299 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %300 = ascvec.duplicate_scalar %c255_i16, %298 : i16 -> <ui16>
                          %301 = ascvec.duplicate_scalar %c-128_i8, %299 : i8 -> <ui8>
                          %302 = ascvec.arange %c0_i16 : i16 -> <ui16>
                          %303 = ascvec.duplicate_scalar %c1_i16, %298 : i16 -> <ui16>
                          %304 = ascvec.xor %302, %303, %298 : <ui16>, <ui16> -> <ui16>
                          %305 = ascvec.duplicate_scalar %c32767_i16, %298 : i16 -> <ui16>
                          %306 = ascvec.duplicate_scalar %cst, %298 : bf16 -> <bf16>
                          scf.for %arg15 = %arg13 to %arg14 step %c1_i32_9  : i32 {
                            %307 = arith.muli %arg15, %c256_i32 : i32
                            %308 = arith.index_cast %307 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg9[%308] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %309 = arith.muli %arg15, %c64_i32 : i32
                            %310 = arith.index_cast %309 : i32 to index
                            %311 = ascvec.loadalign %arg10[%310] : memref<8192xi32, 1>, index -> <i32>
                            %312 = ascvec.reinterpret %311 : <i32> -> <ui16>
                            %313 = ascvec.reinterpret %312 : <ui16> -> <ui32>
                            %314 = ascvec.shl %313, 8, %297 : <ui32> -> <ui32>
                            %315 = ascvec.shr %313, 24, %297 : <ui32> -> <ui32>
                            %316 = ascvec.or %314, %315, %297 : <ui32>, <ui32> -> <ui32>
                            %317 = ascvec.reinterpret %316 : <ui32> -> <ui16>
                            %318 = ascvec.gather_reg %312, %304 : <ui16>, <ui16> -> <ui16>
                            %319 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %320 = ascvec.and %319, %305, %298 : <ui16>, <ui16> -> <ui16>
                            %321 = ascvec.cmp_lt_scalar %320, %c14464_i16, %298 : <ui16>, i16 -> !ascvec.mask
                            %322 = ascvec.shr %318, 11, %298 : <ui16> -> <ui16>
                            %323 = ascvec.add %320, %322, %298 : <ui16>
                            %324 = ascvec.shr %323, 5, %298 : <ui16> -> <ui16>
                            %325 = ascvec.reinterpret %320 : <ui16> -> <bf16>
                            %326 = ascvec.mul %325, %306, %321 : <bf16>
                            %327 = ascvec.cast %326, %298 {rounding = 4 : i32} : <bf16> -> <f16>
                            %328 = ascvec.cast %327, %298 {rounding = 4 : i32} : <f16> -> <i16>
                            %329 = ascvec.reinterpret %328 : <i16> -> <ui16>
                            %330 = ascvec.shr %318, 4, %298 : <ui16> -> <ui16>
                            %331 = ascvec.add %329, %330, %298 : <ui16>
                            %332 = ascvec.shr %331, 12, %298 : <ui16> -> <ui16>
                            %333 = ascvec.add_scalar %332, %c448_i16, %298 : <ui16>, i16
                            %334 = ascvec.select %333, %324, %321 : <ui16>
                            %335 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %336 = ascvec.and %335, %305, %298 : <ui16>, <ui16> -> <ui16>
                            %337 = ascvec.cmp_lt_scalar %336, %c14464_i16, %298 : <ui16>, i16 -> !ascvec.mask
                            %338 = ascvec.shr %317, 11, %298 : <ui16> -> <ui16>
                            %339 = ascvec.add %336, %338, %298 : <ui16>
                            %340 = ascvec.shr %339, 5, %298 : <ui16> -> <ui16>
                            %341 = ascvec.reinterpret %336 : <ui16> -> <bf16>
                            %342 = ascvec.mul %341, %306, %337 : <bf16>
                            %343 = ascvec.cast %342, %298 {rounding = 4 : i32} : <bf16> -> <f16>
                            %344 = ascvec.cast %343, %298 {rounding = 4 : i32} : <f16> -> <i16>
                            %345 = ascvec.reinterpret %344 : <i16> -> <ui16>
                            %346 = ascvec.shr %317, 4, %298 : <ui16> -> <ui16>
                            %347 = ascvec.add %345, %346, %298 : <ui16>
                            %348 = ascvec.shr %347, 12, %298 : <ui16> -> <ui16>
                            %349 = ascvec.add_scalar %348, %c448_i16, %298 : <ui16>, i16
                            %350 = ascvec.select %349, %340, %337 : <ui16>
                            %351 = ascvec.and %334, %300, %298 : <ui16>, <ui16> -> <ui16>
                            %352 = ascvec.shl %350, 8, %298 : <ui16> -> <ui16>
                            %353 = ascvec.or %351, %352, %298 : <ui16>, <ui16> -> <ui16>
                            %354 = ascvec.reinterpret %353 : <ui16> -> <ui8>
                            %355 = ascvec.add_scalar %354, %c64_i8, %299 : <ui8>, i8
                            %356 = ascvec.min_scalar %355, %c123_i8, %299 : <ui8>, i8
                            %357 = arith.muli %arg15, %c512_i32 : i32
                            %358 = arith.index_cast %357 : i32 to index
                            %result0_10, %result1_11 = ascvec.loadalign_deintlv %arg11[%358] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %359 = ascvec.and %result1_11, %301, %299 : <ui8>, <ui8> -> <ui8>
                            %360 = ascvec.or %356, %359, %299 : <ui8>, <ui8> -> <ui8>
                            ascvec.storealign %arg12[%308], %360, %299 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
                      } else {
                        %294 = arith.trunci %arg8 : i64 to i32
                        %295 = arith.trunci %284 : i64 to i32
                        ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %296 = ascscalar.bitcast_memref %129 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%129, %alloc_7, %296, %277, %alloc_4, %294, %295 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<8192xi32, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<?xui16, 1>, %arg13: memref<32768xui8, 1>, %arg14: i32, %arg15: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_9 = arith.constant 1 : i32
                          %c127_i8 = arith.constant 127 : i8
                          %c16_i32 = arith.constant 16 : i32
                          %c0_i16 = arith.constant 0 : i16
                          %c123_i8 = arith.constant 123 : i8
                          %c-1_i8 = arith.constant -1 : i8
                          %c60_i8 = arith.constant 60 : i8
                          %cst = arith.constant 6.10351526E-5 : f32
                          %c64_i32 = arith.constant 64 : i32
                          %c32767_i16 = arith.constant 32767 : i16
                          %c0_10 = arith.constant 0 : index
                          %c256_i32 = arith.constant 256 : i32
                          %c65535_i32 = arith.constant 65535 : i32
                          %c3_i8 = arith.constant 3 : i8
                          %c-128_i8 = arith.constant -128 : i8
                          %c0_i8 = arith.constant 0 : i8
                          %c8_i32 = arith.constant 8 : i32
                          %297 = ascvec.create_mask : !ascvec.mask
                          %298 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %299 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                          %300 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %301 = ascvec.duplicate_scalar %c-128_i8, %299 : i8 -> <ui8>
                          %302 = ascvec.shr %300, 5, %299 : <ui8> -> <ui8>
                          %303 = ascvec.duplicate_scalar %c3_i8, %299 : i8 -> <ui8>
                          %304 = ascvec.and %300, %303, %299 : <ui8>, <ui8> -> <ui8>
                          %305 = ascvec.shl %304, 6, %299 : <ui8> -> <ui8>
                          %306 = ascvec.shr %300, 2, %299 : <ui8> -> <ui8>
                          %307 = ascvec.or %305, %306, %299 : <ui8>, <ui8> -> <ui8>
                          %308 = ascvec.duplicate_scalar %c65535_i32, %297 : i32 -> <ui32>
                          scf.for %arg16 = %arg14 to %arg15 step %c1_i32_9  : i32 {
                            %309 = arith.muli %arg16, %c256_i32 : i32
                            %310 = arith.index_cast %309 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg9[%310] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %311 = ascvec.duplicate_scalar %c32767_i16, %298 : i16 -> <ui16>
                            %312 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %313 = ascvec.and %312, %311, %298 : <ui16>, <ui16> -> <ui16>
                            %314 = ascvec.reinterpret %313 : <ui16> -> <bf16>
                            %315 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %316 = ascvec.and %315, %311, %298 : <ui16>, <ui16> -> <ui16>
                            %317 = ascvec.reinterpret %316 : <ui16> -> <bf16>
                            %318 = ascvec.cast %314, %298 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %319 = ascvec.cast %314, %298 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %320 = ascvec.cast %317, %298 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %321 = ascvec.cast %317, %298 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %322 = arith.muli %arg16, %c64_i32 : i32
                            %323 = arith.index_cast %322 : i32 to index
                            %324 = ascvec.loadalign %arg10[%323] : memref<8192xi32, 1>, index -> <i32>
                            %325 = ascvec.reinterpret %324 : <i32> -> <ui32>
                            %326 = ascvec.shl %325, 8, %297 : <ui32> -> <ui32>
                            %327 = ascvec.shr %325, 24, %297 : <ui32> -> <ui32>
                            %328 = ascvec.or %326, %327, %297 : <ui32>, <ui32> -> <ui32>
                            %329 = ascvec.shr %325, 16, %297 : <ui32> -> <ui32>
                            %330 = ascvec.and %328, %308, %297 : <ui32>, <ui32> -> <ui32>
                            %331 = ascvec.and %325, %308, %297 : <ui32>, <ui32> -> <ui32>
                            %332 = ascvec.shr %328, 16, %297 : <ui32> -> <ui32>
                            %333 = ascvec.reinterpret %318 : <f32> -> <ui32>
                            %334 = ascvec.reinterpret %333 : <ui32> -> <f32>
                            %335 = ascvec.max_scalar %334, %cst, %297 : <f32>, f32
                            %336 = ascvec.add %334, %335, %297 : <f32>
                            %337 = ascvec.reinterpret %336 : <f32> -> <ui32>
                            %338 = ascvec.shr %337, 5, %297 : <ui32> -> <ui32>
                            %339 = ascvec.add %338, %329, %297 : <ui32>
                            %340 = ascvec.reinterpret %320 : <f32> -> <ui32>
                            %341 = ascvec.reinterpret %340 : <ui32> -> <f32>
                            %342 = ascvec.max_scalar %341, %cst, %297 : <f32>, f32
                            %343 = ascvec.add %341, %342, %297 : <f32>
                            %344 = ascvec.reinterpret %343 : <f32> -> <ui32>
                            %345 = ascvec.shr %344, 5, %297 : <ui32> -> <ui32>
                            %346 = ascvec.add %345, %330, %297 : <ui32>
                            %347 = ascvec.reinterpret %319 : <f32> -> <ui32>
                            %348 = ascvec.reinterpret %347 : <ui32> -> <f32>
                            %349 = ascvec.max_scalar %348, %cst, %297 : <f32>, f32
                            %350 = ascvec.add %348, %349, %297 : <f32>
                            %351 = ascvec.reinterpret %350 : <f32> -> <ui32>
                            %352 = ascvec.shr %351, 5, %297 : <ui32> -> <ui32>
                            %353 = ascvec.add %352, %331, %297 : <ui32>
                            %354 = ascvec.reinterpret %321 : <f32> -> <ui32>
                            %355 = ascvec.reinterpret %354 : <ui32> -> <f32>
                            %356 = ascvec.max_scalar %355, %cst, %297 : <f32>, f32
                            %357 = ascvec.add %355, %356, %297 : <f32>
                            %358 = ascvec.reinterpret %357 : <f32> -> <ui32>
                            %359 = ascvec.shr %358, 5, %297 : <ui32> -> <ui32>
                            %360 = ascvec.add %359, %332, %297 : <ui32>
                            %361 = ascvec.reinterpret %339 : <ui32> -> <ui16>
                            %362 = ascvec.reinterpret %346 : <ui32> -> <ui16>
                            %result0_11, %result1_12 = ascvec.deintlv %361, %362 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %363 = ascvec.reinterpret %353 : <ui32> -> <ui16>
                            %364 = ascvec.reinterpret %360 : <ui32> -> <ui16>
                            %result0_13, %result1_14 = ascvec.deintlv %363, %364 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %365 = ascvec.reinterpret %result1_12 : <ui16> -> <ui8>
                            %366 = ascvec.reinterpret %result1_14 : <ui16> -> <ui8>
                            %result0_15, %result1_16 = ascvec.deintlv %365, %366 : <ui8>, <ui8> -> <ui8>, <ui8>
                            %367 = ascvec.gather_reg %result0_15, %307 : <ui8>, <ui8> -> <ui8>
                            %368 = ascvec.add_scalar %367, %c60_i8, %299 : <ui8>, i8
                            %369 = ascvec.cmp_ne_scalar %368, %c-1_i8, %299 : <ui8>, i8 -> !ascvec.mask
                            %370 = ascvec.min_scalar %368, %c123_i8, %369 : <ui8>, i8
                            %371 = arith.muli %arg16, %c512_i32 : i32
                            %372 = arith.index_cast %371 : i32 to index
                            %result0_17, %result1_18 = ascvec.loadalign_deintlv %arg11[%372] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %373 = ascvec.and %result1_18, %301, %299 : <ui8>, <ui8> -> <ui8>
                            %374 = ascvec.or %370, %373, %299 : <ui8>, <ui8> -> <ui8>
                            %375 = ascvec.arange %c0_i16 : i16 -> <ui16>
                            %376 = arith.muli %arg16, %c16_i32 : i32
                            %377 = arith.trunci %376 : i32 to i16
                            %378 = ascvec.add_scalar %375, %377, %298 : <ui16>, i16
                            %379 = ascvec.gather %arg12[%c0_10], %378, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                            %380 = ascvec.pack %379 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                            %381 = ascvec.gather_reg %380, %302 : <ui8>, <ui8> -> <ui8>
                            %382 = ascvec.cmp_eq_scalar %381, %c-1_i8, %299 : <ui8>, i8 -> !ascvec.mask
                            %383 = ascvec.duplicate_scalar %c127_i8, %382 merge %374 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                            ascvec.storealign %arg13[%310], %383, %299 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
                      }
                    }
                  }
                }
              } else {
                %219 = arith.divsi %94, %c256_i64 : i64
                %220 = arith.addi %94, %c1023_i64 : i64
                %221 = arith.remsi %220, %c1024_i64 : i64
                %222 = arith.cmpi ne, %221, %c0_i64 : i64
                %223 = arith.cmpi slt, %221, %c0_i64 : i64
                %224 = arith.andi %222, %223 : i1
                %225 = arith.divsi %220, %c1024_i64 : i64
                %226 = arith.extui %224 : i1 to i64
                %227 = arith.subi %225, %226 : i64
                %228 = arith.trunci %arg4 : i64 to i32
                %229 = arith.extui %228 : i32 to i64
                %230 = arith.subi %arg4, %229 : i64
                %231 = arith.remsi %230, %c4294967296_i64 : i64
                %232 = arith.cmpi ne, %231, %c0_i64 : i64
                %233 = arith.cmpi slt, %231, %c0_i64 : i64
                %234 = arith.andi %232, %233 : i1
                %235 = arith.divsi %230, %c4294967296_i64 : i64
                %236 = arith.extui %234 : i1 to i64
                %237 = arith.subi %235, %236 : i64
                %238 = arith.trunci %237 : i64 to i32
                %239 = arith.divui %207, %c16_i64 : i64
                %240 = arith.trunci %239 : i64 to i32
                %241 = arith.extui %240 : i32 to i64
                %242 = arith.subi %239, %241 : i64
                %243 = arith.remsi %242, %c4294967296_i64 : i64
                %244 = arith.cmpi ne, %243, %c0_i64 : i64
                %245 = arith.cmpi slt, %243, %c0_i64 : i64
                %246 = arith.andi %244, %245 : i1
                %247 = arith.divsi %242, %c4294967296_i64 : i64
                %248 = arith.extui %246 : i1 to i64
                %249 = arith.subi %247, %248 : i64
                %250 = arith.trunci %249 : i64 to i32
                %251 = arith.trunci %arg5 : i64 to i32
                %252 = arith.extui %251 : i32 to i64
                %253 = arith.subi %arg5, %252 : i64
                %254 = arith.remsi %253, %c4294967296_i64 : i64
                %255 = arith.cmpi ne, %254, %c0_i64 : i64
                %256 = arith.cmpi slt, %254, %c0_i64 : i64
                %257 = arith.andi %255, %256 : i1
                %258 = arith.divsi %253, %c4294967296_i64 : i64
                %259 = arith.extui %257 : i1 to i64
                %260 = arith.subi %258, %259 : i64
                %261 = arith.trunci %260 : i64 to i32
                %262 = arith.divui %251, %c4_i32 : i32
                %263 = arith.remui %261, %c4_i32 : i32
                %264 = arith.muli %263, %c1073741824_i32 : i32
                %265 = arith.addi %262, %264 : i32
                %266 = arith.divui %261, %c4_i32 : i32
                %267 = arith.muli %227, %c64_i64 : i64
                %268 = arith.divui %239, %c16384_i64 : i64
                %269 = arith.addi %239, %267 : i64
                %270 = arith.subi %269, %c1_i64 : i64
                %271 = arith.divui %270, %c16384_i64 : i64
                %272 = arith.cmpi eq, %268, %271 : i64
                scf.if %272 {
                  %278 = arith.trunci %268 : i64 to i32
                  %279 = arith.extui %278 : i32 to i64
                  %280 = arith.subi %268, %279 : i64
                  %281 = arith.remsi %280, %c4294967296_i64 : i64
                  %282 = arith.cmpi ne, %281, %c0_i64 : i64
                  %283 = arith.cmpi slt, %281, %c0_i64 : i64
                  %284 = arith.andi %282, %283 : i1
                  %285 = arith.divsi %280, %c4294967296_i64 : i64
                  %286 = arith.extui %284 : i1 to i64
                  %287 = arith.subi %285, %286 : i64
                  %288 = arith.trunci %287 : i64 to i32
                  %289 = arith.addi %265, %278 : i32
                  %290 = arith.addi %266, %288 : i32
                  %291 = arith.cmpi ult, %289, %265 : i32
                  %292 = arith.extui %291 : i1 to i32
                  %293 = arith.addi %290, %292 : i32
                  %294 = arith.extui %289 : i32 to i64
                  %295 = arith.muli %294, %c3449720151_i64 : i64
                  %296 = arith.trunci %295 : i64 to i32
                  %297 = arith.extui %296 : i32 to i64
                  %298 = arith.subi %295, %297 : i64
                  %299 = arith.remsi %298, %c4294967296_i64 : i64
                  %300 = arith.cmpi ne, %299, %c0_i64 : i64
                  %301 = arith.cmpi slt, %299, %c0_i64 : i64
                  %302 = arith.andi %300, %301 : i1
                  %303 = arith.divsi %298, %c4294967296_i64 : i64
                  %304 = arith.extui %302 : i1 to i64
                  %305 = arith.subi %303, %304 : i64
                  %306 = arith.trunci %305 : i64 to i32
                  %307 = arith.xori %293, %238 : i32
                  %308 = arith.extui %307 : i32 to i64
                  %309 = arith.muli %308, %c3449720151_i64 : i64
                  %310 = arith.trunci %309 : i64 to i32
                  %311 = arith.extui %310 : i32 to i64
                  %312 = arith.subi %309, %311 : i64
                  %313 = arith.remsi %312, %c4294967296_i64 : i64
                  %314 = arith.cmpi ne, %313, %c0_i64 : i64
                  %315 = arith.cmpi slt, %313, %c0_i64 : i64
                  %316 = arith.andi %314, %315 : i1
                  %317 = arith.divsi %312, %c4294967296_i64 : i64
                  %318 = arith.extui %316 : i1 to i64
                  %319 = arith.subi %317, %318 : i64
                  %320 = arith.trunci %319 : i64 to i32
                  %321 = arith.addi %228, %c-1640531527_i32 : i32
                  %322 = arith.addi %238, %c-1150833019_i32 : i32
                  %323 = arith.xori %320, %296 : i32
                  %324 = arith.xori %323, %321 : i32
                  %325 = arith.extui %324 : i32 to i64
                  %326 = arith.muli %325, %c3528531795_i64 : i64
                  %327 = arith.trunci %326 : i64 to i32
                  %328 = arith.extui %327 : i32 to i64
                  %329 = arith.subi %326, %328 : i64
                  %330 = arith.remsi %329, %c4294967296_i64 : i64
                  %331 = arith.cmpi ne, %330, %c0_i64 : i64
                  %332 = arith.cmpi slt, %330, %c0_i64 : i64
                  %333 = arith.andi %331, %332 : i1
                  %334 = arith.divsi %329, %c4294967296_i64 : i64
                  %335 = arith.extui %333 : i1 to i64
                  %336 = arith.subi %334, %335 : i64
                  %337 = arith.trunci %336 : i64 to i32
                  %338 = arith.xori %306, %228 : i32
                  %339 = arith.addi %228, %c1013904242_i32 : i32
                  %340 = arith.xori %310, %339 : i32
                  %341 = arith.addi %238, %c1993301258_i32 : i32
                  %342 = arith.xori %337, %341 : i32
                  ascsimt.vf(%240, %338, %322, %340, %342, %228, %238, %327, %alloc_7, %267 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: memref<8192xi32, 1>, %arg17: i64):
                    %c256_i64_9 = arith.constant 256 : i64
                    %c3_i64_10 = arith.constant 3 : i64
                    %c2_i64_11 = arith.constant 2 : i64
                    %c1_i64_12 = arith.constant 1 : i64
                    %c4_i64_13 = arith.constant 4 : i64
                    %c1_i32_14 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c3_i32 = arith.constant 3 : i32
                    %c-1150833019_i32_15 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_16 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %343 = ascsimt.thread_idx {ascsimt.scope = 51 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %344 = arith.extsi %343 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg18 = %344 to %arg17 step %c256_i64_9  : i64 {
                      %345 = arith.trunci %arg18 : i64 to i32
                      %346 = arith.addi %arg8, %345 : i32
                      %347 = arith.remui %346, %c16384_i32 : i32
                      %348 = arith.xori %347, %arg9 : i32
                      %349 = arith.muli %348, %c-766435501_i32 : i32
                      %350 = "ascsimt.bitcast"(%348) : (i32) -> ui32
                      %351 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %352 = ascsimt.math_binary %350, %351, "mulhi" : ui32, ui32 -> ui32
                      %353 = "ascsimt.bitcast"(%352) : (ui32) -> i32
                      %354 = arith.xori %353, %arg10 : i32
                      %355 = arith.muli %354, %c-845247145_i32 : i32
                      %356 = "ascsimt.bitcast"(%354) : (i32) -> ui32
                      %357 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %358 = ascsimt.math_binary %356, %357, "mulhi" : ui32, ui32 -> ui32
                      %359 = "ascsimt.bitcast"(%358) : (ui32) -> i32
                      %360 = arith.xori %359, %arg11 : i32
                      %361 = arith.xori %349, %arg12 : i32
                      %362:4 = scf.for %arg19 = %c3_i32 to %c10_i32 step %c1_i32_14 iter_args(%arg20 = %360, %arg21 = %355, %arg22 = %361, %arg23 = %arg15) -> (i32, i32, i32, i32)  : i32 {
                        %371 = arith.muli %arg20, %c-766435501_i32 : i32
                        %372 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                        %373 = ascsimt.math_binary %372, %351, "mulhi" : ui32, ui32 -> ui32
                        %374 = "ascsimt.bitcast"(%373) : (ui32) -> i32
                        %375 = arith.muli %arg22, %c-845247145_i32 : i32
                        %376 = "ascsimt.bitcast"(%arg22) : (i32) -> ui32
                        %377 = ascsimt.math_binary %376, %357, "mulhi" : ui32, ui32 -> ui32
                        %378 = "ascsimt.bitcast"(%377) : (ui32) -> i32
                        %379 = arith.muli %arg19, %c-1640531527_i32_16 : i32
                        %380 = arith.addi %arg13, %379 : i32
                        %381 = arith.muli %arg19, %c-1150833019_i32_15 : i32
                        %382 = arith.addi %arg14, %381 : i32
                        %383 = arith.xori %378, %arg21 : i32
                        %384 = arith.xori %383, %380 : i32
                        %385 = arith.xori %374, %arg23 : i32
                        %386 = arith.xori %385, %382 : i32
                        scf.yield %384, %375, %386, %371 : i32, i32, i32, i32
                      }
                      %363 = arith.muli %arg18, %c4_i64_13 : i64
                      %364 = arith.index_cast %363 : i64 to index
                      ascsimt.store %arg16[%364], %362#0 : memref<8192xi32, 1>, i32
                      %365 = arith.addi %363, %c1_i64_12 : i64
                      %366 = arith.index_cast %365 : i64 to index
                      ascsimt.store %arg16[%366], %362#1 : memref<8192xi32, 1>, i32
                      %367 = arith.addi %363, %c2_i64_11 : i64
                      %368 = arith.index_cast %367 : i64 to index
                      ascsimt.store %arg16[%368], %362#2 : memref<8192xi32, 1>, i32
                      %369 = arith.addi %363, %c3_i64_10 : i64
                      %370 = arith.index_cast %369 : i64 to index
                      ascsimt.store %arg16[%370], %362#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%240, %250, %265, %266, %228, %238, %alloc_7, %267 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: memref<8192xi32, 1>, %arg15: i64):
                    %c256_i64_9 = arith.constant 256 : i64
                    %c3_i64_10 = arith.constant 3 : i64
                    %c2_i64_11 = arith.constant 2 : i64
                    %c1_i64_12 = arith.constant 1 : i64
                    %c4_i64_13 = arith.constant 4 : i64
                    %c1_i32_14 = arith.constant 1 : i32
                    %c10_i32 = arith.constant 10 : i32
                    %c0_i32_15 = arith.constant 0 : i32
                    %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                    %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                    %c-845247145_i32 = arith.constant -845247145 : i32
                    %c-766435501_i32 = arith.constant -766435501 : i32
                    %c262144_i32 = arith.constant 262144 : i32
                    %c16384_i32 = arith.constant 16384 : i32
                    %278 = ascsimt.thread_idx {ascsimt.scope = 52 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %279 = arith.extsi %278 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg16 = %279 to %arg15 step %c256_i64_9  : i64 {
                      %280 = arith.trunci %arg16 : i64 to i32
                      %281 = arith.addi %arg8, %280 : i32
                      %282 = arith.cmpi ult, %281, %arg8 : i32
                      %283 = arith.extui %282 : i1 to i32
                      %284 = arith.addi %arg9, %283 : i32
                      %285 = arith.divui %281, %c16384_i32 : i32
                      %286 = arith.remui %284, %c16384_i32 : i32
                      %287 = arith.muli %286, %c262144_i32 : i32
                      %288 = arith.addi %285, %287 : i32
                      %289 = arith.divui %284, %c16384_i32 : i32
                      %290 = arith.addi %arg10, %288 : i32
                      %291 = arith.addi %arg11, %289 : i32
                      %292 = arith.cmpi ult, %290, %arg10 : i32
                      %293 = arith.extui %292 : i1 to i32
                      %294 = arith.addi %291, %293 : i32
                      %295 = arith.remui %281, %c16384_i32 : i32
                      %296:4 = scf.for %arg17 = %c0_i32_15 to %c10_i32 step %c1_i32_14 iter_args(%arg18 = %c0_i32_15, %arg19 = %295, %arg20 = %290, %arg21 = %294) -> (i32, i32, i32, i32)  : i32 {
                        %305 = arith.muli %arg18, %c-766435501_i32 : i32
                        %306 = "ascsimt.bitcast"(%arg18) : (i32) -> ui32
                        %307 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %308 = ascsimt.math_binary %306, %307, "mulhi" : ui32, ui32 -> ui32
                        %309 = "ascsimt.bitcast"(%308) : (ui32) -> i32
                        %310 = arith.muli %arg20, %c-845247145_i32 : i32
                        %311 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                        %312 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %313 = ascsimt.math_binary %311, %312, "mulhi" : ui32, ui32 -> ui32
                        %314 = "ascsimt.bitcast"(%313) : (ui32) -> i32
                        %315 = arith.muli %arg17, %c-1640531527_i32_17 : i32
                        %316 = arith.addi %arg12, %315 : i32
                        %317 = arith.muli %arg17, %c-1150833019_i32_16 : i32
                        %318 = arith.addi %arg13, %317 : i32
                        %319 = arith.xori %314, %arg19 : i32
                        %320 = arith.xori %319, %316 : i32
                        %321 = arith.xori %309, %arg21 : i32
                        %322 = arith.xori %321, %318 : i32
                        scf.yield %320, %310, %322, %305 : i32, i32, i32, i32
                      }
                      %297 = arith.muli %arg16, %c4_i64_13 : i64
                      %298 = arith.index_cast %297 : i64 to index
                      ascsimt.store %arg14[%298], %296#0 : memref<8192xi32, 1>, i32
                      %299 = arith.addi %297, %c1_i64_12 : i64
                      %300 = arith.index_cast %299 : i64 to index
                      ascsimt.store %arg14[%300], %296#1 : memref<8192xi32, 1>, i32
                      %301 = arith.addi %297, %c2_i64_11 : i64
                      %302 = arith.index_cast %301 : i64 to index
                      ascsimt.store %arg14[%302], %296#2 : memref<8192xi32, 1>, i32
                      %303 = arith.addi %297, %c3_i64_10 : i64
                      %304 = arith.index_cast %303 : i64 to index
                      ascsimt.store %arg14[%304], %296#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %273 = arith.trunci %27 : i64 to i32
                %274 = arith.divsi %273, %c2_i32 : i32
                %275 = arith.trunci %274 : i32 to i8
                %276 = arith.trunci %219 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                %277 = ascscalar.bitcast_memref %129 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                ascvec.vf(%275, %alloc_5, %129, %277, %alloc_7, %alloc_4, %276 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui8, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, i32) {
                ^bb0(%arg8: i8, %arg9: memref<2048xui8, 1>, %arg10: memref<32768xbf16, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<8192xi32, 1>, %arg13: memref<32768xui8, 1>, %arg14: i32):
                  %c512_i32 = arith.constant 512 : i32
                  %c1_i32_9 = arith.constant 1 : i32
                  %c2139062143_i32 = arith.constant 2139062143 : i32
                  %c-8323200_i32 = arith.constant -8323200 : i32
                  %c123_i8 = arith.constant 123 : i8
                  %c-1_i8 = arith.constant -1 : i8
                  %c60_i8 = arith.constant 60 : i8
                  %cst = arith.constant 6.10351526E-5 : f32
                  %c64_i32 = arith.constant 64 : i32
                  %c0_i16 = arith.constant 0 : i16
                  %c0_i32_10 = arith.constant 0 : i32
                  %c254_i32 = arith.constant 254 : i32
                  %c-15_i32 = arith.constant -15 : i32
                  %c15_i32 = arith.constant 15 : i32
                  %c255_i32 = arith.constant 255 : i32
                  %c32767_i16 = arith.constant 32767 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c0_11 = arith.constant 0 : index
                  %c32_i8 = arith.constant 32 : i8
                  %c65535_i32 = arith.constant 65535 : i32
                  %c3_i8 = arith.constant 3 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c-128_i8 = arith.constant -128 : i8
                  %278 = ascvec.create_mask : !ascvec.mask
                  %279 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %280 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %281 = ascvec.duplicate_scalar %c-128_i8, %280 : i8 -> <ui8>
                  %282 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %283 = ascvec.duplicate_scalar %c3_i8, %280 : i8 -> <ui8>
                  %284 = ascvec.and %282, %283, %280 : <ui8>, <ui8> -> <ui8>
                  %285 = ascvec.shl %284, 6, %280 : <ui8> -> <ui8>
                  %286 = ascvec.shr %282, 2, %280 : <ui8> -> <ui8>
                  %287 = ascvec.or %285, %286, %280 : <ui8>, <ui8> -> <ui8>
                  %288 = ascvec.duplicate_scalar %c65535_i32, %278 : i32 -> <ui32>
                  %289 = ascvec.cmp_lt_scalar %282, %c32_i8, %280 : <ui8>, i8 -> !ascvec.mask
                  %290 = ascvec.cmp_lt_scalar %284, %arg8, %280 : <ui8>, i8 -> !ascvec.mask
                  %291 = ascvec.mask_and %289, %290, %280
                  ascvec.store_unalign_begin %arg9 : memref<2048xui8, 1>
                  scf.for %arg15 = %c0_i32_10 to %arg14 step %c1_i32_9  : i32 {
                    %292 = arith.muli %arg15, %c256_i32 : i32
                    %293 = arith.index_cast %292 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg10[%293] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %294 = ascvec.duplicate_scalar %c32767_i16, %279 : i16 -> <ui16>
                    %295 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %296 = ascvec.and %295, %294, %279 : <ui16>, <ui16> -> <ui16>
                    %297 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %298 = ascvec.and %297, %294, %279 : <ui16>, <ui16> -> <ui16>
                    %299 = ascvec.max %296, %298, %279 : <ui16>
                    %300 = ascvec.reduce_max_datablock %299, %279 : <ui16>
                    %301 = ascvec.unpack %300 {part = "lower"} : <ui16> -> <ui32>
                    %302 = ascvec.shl %301, 16, %278 : <ui32> -> <ui32>
                    %303 = arith.muli %arg15, %c512_i32 : i32
                    %304 = arith.index_cast %303 : i32 to index
                    %result0_13, %result1_14 = ascvec.loadalign_deintlv %arg11[%304] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                    %305 = ascvec.and %result1_14, %281, %280 : <ui8>, <ui8> -> <ui8>
                    %306 = ascvec.reinterpret %305 : <ui8> -> <ui32>
                    %307 = ascvec.shr %302, 23, %278 : <ui32> -> <ui32>
                    %308 = ascvec.cmp_ge_scalar %307, %c255_i32, %278 : <ui32>, i32 -> !ascvec.mask
                    %309 = ascvec.max_scalar %307, %c15_i32, %278 : <ui32>, i32
                    %310 = ascvec.add_scalar %309, %c-15_i32, %278 : <ui32>, i32
                    %311 = ascvec.duplicate_scalar %c255_i32, %308 merge %310 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                    %312 = ascvec.duplicate_scalar %c254_i32, %278 : i32 -> <ui32>
                    %313 = ascvec.sub %312, %311, %278 : <ui32>
                    %314 = ascvec.cmp_ne_scalar %311, %c0_i32_10, %278 : <ui32>, i32 -> !ascvec.mask
                    %315 = ascvec.shl %313, 23, %314 : <ui32> -> <ui32>
                    %316 = ascvec.reinterpret %315 : <ui32> -> <f32>
                    %317 = ascvec.reinterpret %316 : <f32> -> <ui32>
                    %318 = ascvec.shr %317, 16, %278 : <ui32> -> <ui32>
                    %319 = ascvec.pack %318 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                    %320 = ascvec.arange %c0_i16 : i16 -> <ui16>
                    %321 = ascvec.shr %320, 4, %279 : <ui16> -> <ui16>
                    %322 = ascvec.gather_reg %319, %321 : <ui16>, <ui16> -> <ui16>
                    %323 = ascvec.reinterpret %322 : <ui16> -> <bf16>
                    %324 = ascvec.reinterpret %296 : <ui16> -> <bf16>
                    %325 = ascvec.mul %324, %323, %279 : <bf16>
                    %326 = ascvec.reinterpret %298 : <ui16> -> <bf16>
                    %327 = ascvec.mul %326, %323, %279 : <bf16>
                    %328 = ascvec.cast %325, %279 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %329 = ascvec.cast %325, %279 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %330 = ascvec.cast %327, %279 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %331 = ascvec.cast %327, %279 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %332 = ascvec.reinterpret %323 : <bf16> -> <ui32>
                    %333 = arith.muli %arg15, %c64_i32 : i32
                    %334 = arith.index_cast %333 : i32 to index
                    %335 = ascvec.loadalign %arg12[%334] : memref<8192xi32, 1>, index -> <i32>
                    %336 = ascvec.reinterpret %335 : <i32> -> <ui32>
                    %337 = ascvec.shl %336, 8, %278 : <ui32> -> <ui32>
                    %338 = ascvec.shr %336, 24, %278 : <ui32> -> <ui32>
                    %339 = ascvec.or %337, %338, %278 : <ui32>, <ui32> -> <ui32>
                    %340 = ascvec.shr %336, 16, %278 : <ui32> -> <ui32>
                    %341 = ascvec.and %339, %288, %278 : <ui32>, <ui32> -> <ui32>
                    %342 = ascvec.and %336, %288, %278 : <ui32>, <ui32> -> <ui32>
                    %343 = ascvec.shr %339, 16, %278 : <ui32> -> <ui32>
                    %344 = ascvec.reinterpret %328 : <f32> -> <ui32>
                    %345 = ascvec.reinterpret %344 : <ui32> -> <f32>
                    %346 = ascvec.max_scalar %345, %cst, %278 : <f32>, f32
                    %347 = ascvec.add %345, %346, %278 : <f32>
                    %348 = ascvec.reinterpret %347 : <f32> -> <ui32>
                    %349 = ascvec.shr %348, 5, %278 : <ui32> -> <ui32>
                    %350 = ascvec.add %349, %340, %278 : <ui32>
                    %351 = ascvec.reinterpret %330 : <f32> -> <ui32>
                    %352 = ascvec.reinterpret %351 : <ui32> -> <f32>
                    %353 = ascvec.max_scalar %352, %cst, %278 : <f32>, f32
                    %354 = ascvec.add %352, %353, %278 : <f32>
                    %355 = ascvec.reinterpret %354 : <f32> -> <ui32>
                    %356 = ascvec.shr %355, 5, %278 : <ui32> -> <ui32>
                    %357 = ascvec.add %356, %341, %278 : <ui32>
                    %358 = ascvec.reinterpret %329 : <f32> -> <ui32>
                    %359 = ascvec.reinterpret %358 : <ui32> -> <f32>
                    %360 = ascvec.max_scalar %359, %cst, %278 : <f32>, f32
                    %361 = ascvec.add %359, %360, %278 : <f32>
                    %362 = ascvec.reinterpret %361 : <f32> -> <ui32>
                    %363 = ascvec.shr %362, 5, %278 : <ui32> -> <ui32>
                    %364 = ascvec.add %363, %342, %278 : <ui32>
                    %365 = ascvec.reinterpret %331 : <f32> -> <ui32>
                    %366 = ascvec.reinterpret %365 : <ui32> -> <f32>
                    %367 = ascvec.max_scalar %366, %cst, %278 : <f32>, f32
                    %368 = ascvec.add %366, %367, %278 : <f32>
                    %369 = ascvec.reinterpret %368 : <f32> -> <ui32>
                    %370 = ascvec.shr %369, 5, %278 : <ui32> -> <ui32>
                    %371 = ascvec.add %370, %343, %278 : <ui32>
                    %372 = ascvec.reinterpret %350 : <ui32> -> <ui16>
                    %373 = ascvec.reinterpret %357 : <ui32> -> <ui16>
                    %result0_15, %result1_16 = ascvec.deintlv %372, %373 : <ui16>, <ui16> -> <ui16>, <ui16>
                    %374 = ascvec.reinterpret %364 : <ui32> -> <ui16>
                    %375 = ascvec.reinterpret %371 : <ui32> -> <ui16>
                    %result0_17, %result1_18 = ascvec.deintlv %374, %375 : <ui16>, <ui16> -> <ui16>, <ui16>
                    %376 = ascvec.reinterpret %result1_16 : <ui16> -> <ui8>
                    %377 = ascvec.reinterpret %result1_18 : <ui16> -> <ui8>
                    %result0_19, %result1_20 = ascvec.deintlv %376, %377 : <ui8>, <ui8> -> <ui8>, <ui8>
                    %378 = ascvec.gather_reg %result0_19, %287 : <ui8>, <ui8> -> <ui8>
                    %379 = ascvec.reinterpret %378 : <ui8> -> <ui32>
                    %380 = ascvec.reinterpret %379 : <ui32> -> <ui8>
                    %381 = ascvec.add_scalar %380, %c60_i8, %280 : <ui8>, i8
                    %382 = ascvec.cmp_ne_scalar %381, %c-1_i8, %280 : <ui8>, i8 -> !ascvec.mask
                    %383 = ascvec.min_scalar %381, %c123_i8, %382 : <ui8>, i8
                    %384 = ascvec.reinterpret %383 : <ui8> -> <ui32>
                    %385 = ascvec.or %384, %306, %278 : <ui32>, <ui32> -> <ui32>
                    %386 = ascvec.reinterpret %332 : <ui32> -> <ui32>
                    %387 = ascvec.cmp_eq_scalar %386, %c-8323200_i32, %278 : <ui32>, i32 -> !ascvec.mask
                    %388 = ascvec.duplicate_scalar %c2139062143_i32, %387 merge %385 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                    %389 = ascvec.reinterpret %388 : <ui32> -> <ui8>
                    ascvec.storealign %arg13[%293], %389, %280 : memref<32768xui8, 1>, index, <ui8>
                    %390 = ascvec.reinterpret %311 : <ui32> -> <ui8>
                    %391 = ascvec.squeeze_store_reg %390, %291 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg9[%c0_11], %391 : memref<2048xui8, 1>, <ui8>
                  }
                  %c0_12 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg15 = %c0_12 to %c1 step %c1 {
                    ascvec.store_unalign_post %arg9[%c0_11] : memref<2048xui8, 1>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              }
            }
          } else {
            %206 = arith.muli %88, %7 : i64
            %207 = arith.addi %206, %91 : i64
            %208 = arith.remsi %arg3, %c256_i64 : i64
            %209 = arith.cmpi ne, %208, %c0_i64 : i64
            %210 = arith.cmpi slt, %208, %c0_i64 : i64
            %211 = arith.andi %209, %210 : i1
            %212 = arith.select %211, %c256_i64, %c0_i64 : i64
            %213 = arith.addi %208, %212 overflow<nsw> : i64
            %214 = arith.cmpi eq, %213, %c0_i64 : i64
            scf.if %214 {
              %215 = arith.addi %94, %c63_i64 : i64
              %216 = arith.remsi %215, %c64_i64 : i64
              %217 = arith.cmpi ne, %216, %c0_i64 : i64
              %218 = arith.cmpi slt, %216, %c0_i64 : i64
              %219 = arith.andi %217, %218 : i1
              %220 = arith.divsi %215, %c64_i64 : i64
              %221 = arith.extui %219 : i1 to i64
              %222 = arith.subi %220, %221 : i64
              %223 = arith.addi %94, %c1023_i64 : i64
              %224 = arith.remsi %223, %c1024_i64 : i64
              %225 = arith.cmpi ne, %224, %c0_i64 : i64
              %226 = arith.cmpi slt, %224, %c0_i64 : i64
              %227 = arith.andi %225, %226 : i1
              %228 = arith.divsi %223, %c1024_i64 : i64
              %229 = arith.extui %227 : i1 to i64
              %230 = arith.subi %228, %229 : i64
              %231 = arith.trunci %arg4 : i64 to i32
              %232 = arith.extui %231 : i32 to i64
              %233 = arith.subi %arg4, %232 : i64
              %234 = arith.remsi %233, %c4294967296_i64 : i64
              %235 = arith.cmpi ne, %234, %c0_i64 : i64
              %236 = arith.cmpi slt, %234, %c0_i64 : i64
              %237 = arith.andi %235, %236 : i1
              %238 = arith.divsi %233, %c4294967296_i64 : i64
              %239 = arith.extui %237 : i1 to i64
              %240 = arith.subi %238, %239 : i64
              %241 = arith.trunci %240 : i64 to i32
              %242 = arith.divui %207, %c32_i64 : i64
              %243 = arith.trunci %242 : i64 to i32
              %244 = arith.extui %243 : i32 to i64
              %245 = arith.subi %242, %244 : i64
              %246 = arith.remsi %245, %c4294967296_i64 : i64
              %247 = arith.cmpi ne, %246, %c0_i64 : i64
              %248 = arith.cmpi slt, %246, %c0_i64 : i64
              %249 = arith.andi %247, %248 : i1
              %250 = arith.divsi %245, %c4294967296_i64 : i64
              %251 = arith.extui %249 : i1 to i64
              %252 = arith.subi %250, %251 : i64
              %253 = arith.trunci %252 : i64 to i32
              %254 = arith.trunci %arg5 : i64 to i32
              %255 = arith.extui %254 : i32 to i64
              %256 = arith.subi %arg5, %255 : i64
              %257 = arith.remsi %256, %c4294967296_i64 : i64
              %258 = arith.cmpi ne, %257, %c0_i64 : i64
              %259 = arith.cmpi slt, %257, %c0_i64 : i64
              %260 = arith.andi %258, %259 : i1
              %261 = arith.divsi %256, %c4294967296_i64 : i64
              %262 = arith.extui %260 : i1 to i64
              %263 = arith.subi %261, %262 : i64
              %264 = arith.trunci %263 : i64 to i32
              %265 = arith.divui %254, %c4_i32 : i32
              %266 = arith.remui %264, %c4_i32 : i32
              %267 = arith.muli %266, %c1073741824_i32 : i32
              %268 = arith.addi %265, %267 : i32
              %269 = arith.divui %264, %c4_i32 : i32
              %270 = arith.muli %230, %c32_i64 : i64
              %271 = arith.divui %242, %c16384_i64 : i64
              %272 = arith.addi %242, %270 : i64
              %273 = arith.subi %272, %c1_i64 : i64
              %274 = arith.divui %273, %c16384_i64 : i64
              %275 = arith.cmpi eq, %271, %274 : i64
              scf.if %275 {
                %281 = arith.trunci %271 : i64 to i32
                %282 = arith.extui %281 : i32 to i64
                %283 = arith.subi %271, %282 : i64
                %284 = arith.remsi %283, %c4294967296_i64 : i64
                %285 = arith.cmpi ne, %284, %c0_i64 : i64
                %286 = arith.cmpi slt, %284, %c0_i64 : i64
                %287 = arith.andi %285, %286 : i1
                %288 = arith.divsi %283, %c4294967296_i64 : i64
                %289 = arith.extui %287 : i1 to i64
                %290 = arith.subi %288, %289 : i64
                %291 = arith.trunci %290 : i64 to i32
                %292 = arith.addi %268, %281 : i32
                %293 = arith.addi %269, %291 : i32
                %294 = arith.cmpi ult, %292, %268 : i32
                %295 = arith.extui %294 : i1 to i32
                %296 = arith.addi %293, %295 : i32
                %297 = arith.extui %292 : i32 to i64
                %298 = arith.muli %297, %c3449720151_i64 : i64
                %299 = arith.trunci %298 : i64 to i32
                %300 = arith.extui %299 : i32 to i64
                %301 = arith.subi %298, %300 : i64
                %302 = arith.remsi %301, %c4294967296_i64 : i64
                %303 = arith.cmpi ne, %302, %c0_i64 : i64
                %304 = arith.cmpi slt, %302, %c0_i64 : i64
                %305 = arith.andi %303, %304 : i1
                %306 = arith.divsi %301, %c4294967296_i64 : i64
                %307 = arith.extui %305 : i1 to i64
                %308 = arith.subi %306, %307 : i64
                %309 = arith.trunci %308 : i64 to i32
                %310 = arith.xori %296, %241 : i32
                %311 = arith.extui %310 : i32 to i64
                %312 = arith.muli %311, %c3449720151_i64 : i64
                %313 = arith.trunci %312 : i64 to i32
                %314 = arith.extui %313 : i32 to i64
                %315 = arith.subi %312, %314 : i64
                %316 = arith.remsi %315, %c4294967296_i64 : i64
                %317 = arith.cmpi ne, %316, %c0_i64 : i64
                %318 = arith.cmpi slt, %316, %c0_i64 : i64
                %319 = arith.andi %317, %318 : i1
                %320 = arith.divsi %315, %c4294967296_i64 : i64
                %321 = arith.extui %319 : i1 to i64
                %322 = arith.subi %320, %321 : i64
                %323 = arith.trunci %322 : i64 to i32
                %324 = arith.addi %231, %c-1640531527_i32 : i32
                %325 = arith.addi %241, %c-1150833019_i32 : i32
                %326 = arith.xori %323, %299 : i32
                %327 = arith.xori %326, %324 : i32
                %328 = arith.extui %327 : i32 to i64
                %329 = arith.muli %328, %c3528531795_i64 : i64
                %330 = arith.trunci %329 : i64 to i32
                %331 = arith.extui %330 : i32 to i64
                %332 = arith.subi %329, %331 : i64
                %333 = arith.remsi %332, %c4294967296_i64 : i64
                %334 = arith.cmpi ne, %333, %c0_i64 : i64
                %335 = arith.cmpi slt, %333, %c0_i64 : i64
                %336 = arith.andi %334, %335 : i1
                %337 = arith.divsi %332, %c4294967296_i64 : i64
                %338 = arith.extui %336 : i1 to i64
                %339 = arith.subi %337, %338 : i64
                %340 = arith.trunci %339 : i64 to i32
                %341 = arith.xori %309, %231 : i32
                %342 = arith.addi %231, %c1013904242_i32 : i32
                %343 = arith.xori %313, %342 : i32
                %344 = arith.addi %241, %c1993301258_i32 : i32
                %345 = arith.xori %340, %344 : i32
                ascsimt.vf(%243, %341, %325, %343, %345, %231, %241, %330, %alloc_7, %270 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: memref<8192xi32, 1>, %arg17: i64):
                  %c256_i64_9 = arith.constant 256 : i64
                  %c3_i64_10 = arith.constant 3 : i64
                  %c2_i64_11 = arith.constant 2 : i64
                  %c1_i64_12 = arith.constant 1 : i64
                  %c4_i64_13 = arith.constant 4 : i64
                  %c1_i32_14 = arith.constant 1 : i32
                  %c10_i32 = arith.constant 10 : i32
                  %c3_i32 = arith.constant 3 : i32
                  %c-1150833019_i32_15 = arith.constant -1150833019 : i32
                  %c-1640531527_i32_16 = arith.constant -1640531527 : i32
                  %c-845247145_i32 = arith.constant -845247145 : i32
                  %c-766435501_i32 = arith.constant -766435501 : i32
                  %c16384_i32 = arith.constant 16384 : i32
                  %346 = ascsimt.thread_idx {ascsimt.scope = 54 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                  %347 = arith.extsi %346 {cannir.vf_threads = 256 : i64} : i32 to i64
                  scf.for %arg18 = %347 to %arg17 step %c256_i64_9  : i64 {
                    %348 = arith.trunci %arg18 : i64 to i32
                    %349 = arith.addi %arg8, %348 : i32
                    %350 = arith.remui %349, %c16384_i32 : i32
                    %351 = arith.xori %350, %arg9 : i32
                    %352 = arith.muli %351, %c-766435501_i32 : i32
                    %353 = "ascsimt.bitcast"(%351) : (i32) -> ui32
                    %354 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                    %355 = ascsimt.math_binary %353, %354, "mulhi" : ui32, ui32 -> ui32
                    %356 = "ascsimt.bitcast"(%355) : (ui32) -> i32
                    %357 = arith.xori %356, %arg10 : i32
                    %358 = arith.muli %357, %c-845247145_i32 : i32
                    %359 = "ascsimt.bitcast"(%357) : (i32) -> ui32
                    %360 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                    %361 = ascsimt.math_binary %359, %360, "mulhi" : ui32, ui32 -> ui32
                    %362 = "ascsimt.bitcast"(%361) : (ui32) -> i32
                    %363 = arith.xori %362, %arg11 : i32
                    %364 = arith.xori %352, %arg12 : i32
                    %365:4 = scf.for %arg19 = %c3_i32 to %c10_i32 step %c1_i32_14 iter_args(%arg20 = %363, %arg21 = %358, %arg22 = %364, %arg23 = %arg15) -> (i32, i32, i32, i32)  : i32 {
                      %374 = arith.muli %arg20, %c-766435501_i32 : i32
                      %375 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                      %376 = ascsimt.math_binary %375, %354, "mulhi" : ui32, ui32 -> ui32
                      %377 = "ascsimt.bitcast"(%376) : (ui32) -> i32
                      %378 = arith.muli %arg22, %c-845247145_i32 : i32
                      %379 = "ascsimt.bitcast"(%arg22) : (i32) -> ui32
                      %380 = ascsimt.math_binary %379, %360, "mulhi" : ui32, ui32 -> ui32
                      %381 = "ascsimt.bitcast"(%380) : (ui32) -> i32
                      %382 = arith.muli %arg19, %c-1640531527_i32_16 : i32
                      %383 = arith.addi %arg13, %382 : i32
                      %384 = arith.muli %arg19, %c-1150833019_i32_15 : i32
                      %385 = arith.addi %arg14, %384 : i32
                      %386 = arith.xori %381, %arg21 : i32
                      %387 = arith.xori %386, %383 : i32
                      %388 = arith.xori %377, %arg23 : i32
                      %389 = arith.xori %388, %385 : i32
                      scf.yield %387, %378, %389, %374 : i32, i32, i32, i32
                    }
                    %366 = arith.muli %arg18, %c4_i64_13 : i64
                    %367 = arith.index_cast %366 : i64 to index
                    ascsimt.store %arg16[%367], %365#0 : memref<8192xi32, 1>, i32
                    %368 = arith.addi %366, %c1_i64_12 : i64
                    %369 = arith.index_cast %368 : i64 to index
                    ascsimt.store %arg16[%369], %365#1 : memref<8192xi32, 1>, i32
                    %370 = arith.addi %366, %c2_i64_11 : i64
                    %371 = arith.index_cast %370 : i64 to index
                    ascsimt.store %arg16[%371], %365#2 : memref<8192xi32, 1>, i32
                    %372 = arith.addi %366, %c3_i64_10 : i64
                    %373 = arith.index_cast %372 : i64 to index
                    ascsimt.store %arg16[%373], %365#3 : memref<8192xi32, 1>, i32
                  } {cannir.vf_threads = 256 : i64}
                } {threads = 256 : i64}
              } else {
                ascsimt.vf(%243, %253, %268, %269, %231, %241, %alloc_7, %270 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                ^bb0(%arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: memref<8192xi32, 1>, %arg15: i64):
                  %c256_i64_9 = arith.constant 256 : i64
                  %c3_i64_10 = arith.constant 3 : i64
                  %c2_i64_11 = arith.constant 2 : i64
                  %c1_i64_12 = arith.constant 1 : i64
                  %c4_i64_13 = arith.constant 4 : i64
                  %c1_i32_14 = arith.constant 1 : i32
                  %c10_i32 = arith.constant 10 : i32
                  %c0_i32_15 = arith.constant 0 : i32
                  %c-1150833019_i32_16 = arith.constant -1150833019 : i32
                  %c-1640531527_i32_17 = arith.constant -1640531527 : i32
                  %c-845247145_i32 = arith.constant -845247145 : i32
                  %c-766435501_i32 = arith.constant -766435501 : i32
                  %c262144_i32 = arith.constant 262144 : i32
                  %c16384_i32 = arith.constant 16384 : i32
                  %281 = ascsimt.thread_idx {ascsimt.scope = 55 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                  %282 = arith.extsi %281 {cannir.vf_threads = 256 : i64} : i32 to i64
                  scf.for %arg16 = %282 to %arg15 step %c256_i64_9  : i64 {
                    %283 = arith.trunci %arg16 : i64 to i32
                    %284 = arith.addi %arg8, %283 : i32
                    %285 = arith.cmpi ult, %284, %arg8 : i32
                    %286 = arith.extui %285 : i1 to i32
                    %287 = arith.addi %arg9, %286 : i32
                    %288 = arith.divui %284, %c16384_i32 : i32
                    %289 = arith.remui %287, %c16384_i32 : i32
                    %290 = arith.muli %289, %c262144_i32 : i32
                    %291 = arith.addi %288, %290 : i32
                    %292 = arith.divui %287, %c16384_i32 : i32
                    %293 = arith.addi %arg10, %291 : i32
                    %294 = arith.addi %arg11, %292 : i32
                    %295 = arith.cmpi ult, %293, %arg10 : i32
                    %296 = arith.extui %295 : i1 to i32
                    %297 = arith.addi %294, %296 : i32
                    %298 = arith.remui %284, %c16384_i32 : i32
                    %299:4 = scf.for %arg17 = %c0_i32_15 to %c10_i32 step %c1_i32_14 iter_args(%arg18 = %c0_i32_15, %arg19 = %298, %arg20 = %293, %arg21 = %297) -> (i32, i32, i32, i32)  : i32 {
                      %308 = arith.muli %arg18, %c-766435501_i32 : i32
                      %309 = "ascsimt.bitcast"(%arg18) : (i32) -> ui32
                      %310 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %311 = ascsimt.math_binary %309, %310, "mulhi" : ui32, ui32 -> ui32
                      %312 = "ascsimt.bitcast"(%311) : (ui32) -> i32
                      %313 = arith.muli %arg20, %c-845247145_i32 : i32
                      %314 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                      %315 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %316 = ascsimt.math_binary %314, %315, "mulhi" : ui32, ui32 -> ui32
                      %317 = "ascsimt.bitcast"(%316) : (ui32) -> i32
                      %318 = arith.muli %arg17, %c-1640531527_i32_17 : i32
                      %319 = arith.addi %arg12, %318 : i32
                      %320 = arith.muli %arg17, %c-1150833019_i32_16 : i32
                      %321 = arith.addi %arg13, %320 : i32
                      %322 = arith.xori %317, %arg19 : i32
                      %323 = arith.xori %322, %319 : i32
                      %324 = arith.xori %312, %arg21 : i32
                      %325 = arith.xori %324, %321 : i32
                      scf.yield %323, %313, %325, %308 : i32, i32, i32, i32
                    }
                    %300 = arith.muli %arg16, %c4_i64_13 : i64
                    %301 = arith.index_cast %300 : i64 to index
                    ascsimt.store %arg14[%301], %299#0 : memref<8192xi32, 1>, i32
                    %302 = arith.addi %300, %c1_i64_12 : i64
                    %303 = arith.index_cast %302 : i64 to index
                    ascsimt.store %arg14[%303], %299#1 : memref<8192xi32, 1>, i32
                    %304 = arith.addi %300, %c2_i64_11 : i64
                    %305 = arith.index_cast %304 : i64 to index
                    ascsimt.store %arg14[%305], %299#2 : memref<8192xi32, 1>, i32
                    %306 = arith.addi %300, %c3_i64_10 : i64
                    %307 = arith.index_cast %306 : i64 to index
                    ascsimt.store %arg14[%307], %299#3 : memref<8192xi32, 1>, i32
                  } {cannir.vf_threads = 256 : i64}
                } {threads = 256 : i64}
              }
              %276 = arith.trunci %222 : i64 to i32
              %277 = arith.trunci %94 : i64 to i32
              %278 = arith.trunci %27 : i64 to i32
              %279 = arith.muli %276, %c2_i32 : i32
              %280 = arith.cmpi eq, %278, %c4_i32 : i32
              ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%94, %129, %alloc_7, %alloc_4, %alloc_6, %276, %277, %279, %280, %alloc_5 : i64, memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, memref<1024xui32, 1>, i32, i32, i32, i1, memref<2048xui8, 1>) {
              ^bb0(%arg8: i64, %arg9: memref<32768xbf16, 1>, %arg10: memref<8192xi32, 1>, %arg11: memref<32768xui8, 1>, %arg12: memref<1024xui32, 1>, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i1, %arg17: memref<2048xui8, 1>):
                %c2139095040_i32 = arith.constant 2139095040 : i32
                %c127_i32 = arith.constant 127 : i32
                %c123_i32 = arith.constant 123 : i32
                %c947912704_i32 = arith.constant 947912704 : i32
                %cst = arith.constant 4.2949673E+9 : f32
                %c29360128_i32 = arith.constant 29360128 : i32
                %c128_i32 = arith.constant 128 : i32
                %c65535_i32 = arith.constant 65535 : i32
                %c2_i32_9 = arith.constant 2 : i32
                %c7_i32 = arith.constant 7 : i32
                %c1_i32_10 = arith.constant 1 : i32
                %c4_i32_11 = arith.constant 4 : i32
                %c2139160576_i32 = arith.constant 2139160576 : i32
                %c254_i32 = arith.constant 254 : i32
                %c-15_i32 = arith.constant -15 : i32
                %c15_i32 = arith.constant 15 : i32
                %c255_i32 = arith.constant 255 : i32
                %c2147483647_i32 = arith.constant 2147483647 : i32
                %cst_12 = arith.constant 0.000000e+00 : f32
                %c0_13 = arith.constant 0 : index
                %c64_i32 = arith.constant 64 : i32
                %c32_i32 = arith.constant 32 : i32
                %c0_i32_14 = arith.constant 0 : i32
                %281 = ascvec.create_mask : !ascvec.mask
                %282 = ascvec.arange %c0_i32_14 : i32 -> <ui32>
                %283 = ascvec.cmp_lt_scalar %282, %c32_i32, %281 : <ui32>, i32 -> !ascvec.mask
                %284 = ascvec.cmp_ge_scalar %282, %c32_i32, %281 : <ui32>, i32 -> !ascvec.mask
                scf.for %arg18 = %c0_i32_14 to %arg13 step %c1_i32_10  : i32 {
                  %286 = arith.muli %arg18, %c64_i32 : i32
                  %287 = arith.trunci %arg8 : i64 to i32
                  %288 = arith.subi %287, %286 : i32
                  %mask, %remaining_out = ascvec.update_mask %288 : !ascvec.mask
                  %289 = arith.index_cast %286 : i32 to index
                  %290 = ascvec.loadalign_unpack %arg9[%289] {mode = "b16_to_b32"} : memref<32768xbf16, 1>, index -> <bf16>
                  %291 = ascvec.cast %290, %281 {rounding = 6 : i32} : <bf16> -> <f32>
                  %292 = ascvec.duplicate_scalar %cst_12, %281 : f32 -> <f32>
                  %293 = ascvec.select %291, %292, %mask : <f32>
                  %294 = ascvec.reinterpret %293 : <f32> -> <ui32>
                  %295 = ascvec.duplicate_scalar %c2147483647_i32, %281 : i32 -> <ui32>
                  %296 = ascvec.and %294, %295, %281 : <ui32>, <ui32> -> <ui32>
                  %297 = ascvec.reduce_max %296, %283 : <ui32>
                  %298 = ascvec.duplicate %297, %281 : <ui32> -> <ui32>
                  %299 = ascvec.reduce_max %296, %284 : <ui32>
                  %300 = ascvec.duplicate %299, %281 : <ui32> -> <ui32>
                  %301 = ascvec.select %298, %300, %283 : <ui32>
                  %302 = ascvec.shr %301, 23, %281 : <ui32> -> <ui32>
                  %303 = ascvec.cmp_ge_scalar %302, %c255_i32, %281 : <ui32>, i32 -> !ascvec.mask
                  %304 = ascvec.max_scalar %302, %c15_i32, %281 : <ui32>, i32
                  %305 = ascvec.add_scalar %304, %c-15_i32, %281 : <ui32>, i32
                  %306 = ascvec.duplicate_scalar %c255_i32, %303 merge %305 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %307 = ascvec.duplicate_scalar %c254_i32, %281 : i32 -> <ui32>
                  %308 = ascvec.sub %307, %306, %281 : <ui32>
                  %309 = ascvec.cmp_ne_scalar %306, %c0_i32_14, %281 : <ui32>, i32 -> !ascvec.mask
                  %310 = ascvec.shl %308, 23, %309 : <ui32> -> <ui32>
                  %311 = ascvec.duplicate_scalar %c2139160576_i32, %303 merge %310 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %312 = ascvec.reinterpret %311 : <ui32> -> <f32>
                  %313 = ascvec.mul %293, %312, %281 : <f32>
                  %314 = ascvec.cast %313, %281 : <f32> -> <bf16>
                  %315 = ascvec.cast %314, %281 {rounding = 6 : i32} : <bf16> -> <f32>
                  %316 = arith.remsi %arg18, %c4_i32_11 : i32
                  %317 = arith.cmpi ne, %316, %c0_i32_14 : i32
                  %318 = arith.cmpi slt, %316, %c0_i32_14 : i32
                  %319 = arith.andi %317, %318 : i1
                  %320 = arith.select %319, %c4_i32_11, %c0_i32_14 : i32
                  %321 = arith.addi %316, %320 overflow<nsw> : i32
                  %322 = arith.muli %321, %c64_i32 overflow<nsw> : i32
                  %323 = ascvec.add_scalar %282, %322, %281 : <ui32>, i32
                  %324 = arith.divsi %arg18, %c4_i32_11 : i32
                  %325 = arith.extui %319 : i1 to i32
                  %326 = arith.subi %324, %325 : i32
                  %327 = ascvec.shr %323, 3, %281 : <ui32> -> <ui32>
                  %328 = arith.muli %326, %c32_i32 : i32
                  %329 = ascvec.add_scalar %327, %328, %281 : <ui32>, i32
                  %330 = ascvec.gather %arg10[%c0_13], %329, %281 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                  %331 = ascvec.reinterpret %330 : <i32> -> <ui32>
                  %332 = ascvec.duplicate_scalar %c7_i32, %281 : i32 -> <ui32>
                  %333 = ascvec.and %323, %332, %281 : <ui32>, <ui32> -> <ui32>
                  %334 = ascvec.shl %331, 11, %281 : <ui32> -> <ui32>
                  %335 = ascvec.shr %331, 21, %281 : <ui32> -> <ui32>
                  %336 = ascvec.or %334, %335, %281 : <ui32>, <ui32> -> <ui32>
                  %337 = ascvec.shr %336, 3, %281 : <ui32> -> <ui32>
                  %338 = ascvec.shl %336, 29, %281 : <ui32> -> <ui32>
                  %339 = ascvec.or %337, %338, %281 : <ui32>, <ui32> -> <ui32>
                  %340 = ascvec.duplicate_scalar %c1_i32_10, %281 : i32 -> <ui32>
                  %341 = ascvec.and %333, %340, %281 : <ui32>, <ui32> -> <ui32>
                  %342 = ascvec.cmp_ne_scalar %341, %c0_i32_14, %281 : <ui32>, i32 -> !ascvec.mask
                  %343 = ascvec.select %339, %336, %342 : <ui32>
                  %344 = ascvec.shr %343, 8, %281 : <ui32> -> <ui32>
                  %345 = ascvec.shl %343, 24, %281 : <ui32> -> <ui32>
                  %346 = ascvec.or %344, %345, %281 : <ui32>, <ui32> -> <ui32>
                  %347 = ascvec.duplicate_scalar %c2_i32_9, %281 : i32 -> <ui32>
                  %348 = ascvec.and %333, %347, %281 : <ui32>, <ui32> -> <ui32>
                  %349 = ascvec.cmp_ne_scalar %348, %c0_i32_14, %281 : <ui32>, i32 -> !ascvec.mask
                  %350 = ascvec.select %346, %343, %349 : <ui32>
                  %351 = ascvec.shr %350, 16, %281 : <ui32> -> <ui32>
                  %352 = ascvec.shl %350, 16, %281 : <ui32> -> <ui32>
                  %353 = ascvec.or %351, %352, %281 : <ui32>, <ui32> -> <ui32>
                  %354 = ascvec.duplicate_scalar %c4_i32_11, %281 : i32 -> <ui32>
                  %355 = ascvec.and %333, %354, %281 : <ui32>, <ui32> -> <ui32>
                  %356 = ascvec.cmp_ne_scalar %355, %c0_i32_14, %281 : <ui32>, i32 -> !ascvec.mask
                  %357 = ascvec.select %353, %350, %356 : <ui32>
                  %358 = ascvec.duplicate_scalar %c65535_i32, %281 : i32 -> <ui32>
                  %359 = ascvec.and %357, %358, %281 : <ui32>, <ui32> -> <ui32>
                  %360 = ascvec.reinterpret %315 : <f32> -> <ui32>
                  %361 = ascvec.shr %360, 24, %281 : <ui32> -> <ui32>
                  %362 = ascvec.duplicate_scalar %c128_i32, %281 : i32 -> <ui32>
                  %363 = ascvec.and %361, %362, %281 : <ui32>, <ui32> -> <ui32>
                  %364 = ascvec.and %360, %295, %281 : <ui32>, <ui32> -> <ui32>
                  %365 = ascvec.shr %364, 5, %281 : <ui32> -> <ui32>
                  %366 = ascvec.duplicate_scalar %c29360128_i32, %281 : i32 -> <ui32>
                  %367 = ascvec.sub %365, %366, %281 : <ui32>
                  %368 = ascvec.reinterpret %364 : <ui32> -> <f32>
                  %369 = ascvec.mul_scalar %368, %cst, %281 : <f32>, f32
                  %370 = ascvec.cast %369, %281 {rounding = 4 : i32} : <f32> -> <i32>
                  %371 = ascvec.reinterpret %370 : <i32> -> <ui32>
                  %372 = ascvec.cmp_ge_scalar %364, %c947912704_i32, %281 : <ui32>, i32 -> !ascvec.mask
                  %373 = ascvec.select %367, %371, %372 : <ui32>
                  %374 = ascvec.and %359, %358, %281 : <ui32>, <ui32> -> <ui32>
                  %375 = ascvec.add %373, %374, %281 : <ui32>
                  %376 = ascvec.shr %375, 16, %281 : <ui32> -> <ui32>
                  %377 = ascvec.min_scalar %376, %c123_i32, %281 : <ui32>, i32
                  %378 = ascvec.or %377, %363, %281 : <ui32>, <ui32> -> <ui32>
                  %379 = ascvec.duplicate_scalar %c127_i32, %281 : i32 -> <ui32>
                  %380 = ascvec.cmp_gt_scalar %364, %c2139095040_i32, %281 : <ui32>, i32 -> !ascvec.mask
                  %381 = ascvec.select %379, %378, %380 : <ui32>
                  ascvec.storealign_pack %arg11[%289], %381, %mask {mode = "b32_to_b8"} : memref<32768xui8, 1>, index, <ui32>
                  %382 = arith.muli %arg18, %c2_i32_9 : i32
                  %383 = arith.index_cast %382 : i32 to index
                  ascvec.storealign_1st %arg12[%383], %306 : memref<1024xui32, 1>, index, <ui32>
                  %384 = ascvec.duplicate_scalar %c32_i32, %281 : i32 -> <ui32>
                  %385 = ascvec.gather_reg %306, %384 : <ui32>, <ui32> -> <ui32>
                  %386 = arith.addi %382, %c1_i32_10 : i32
                  %387 = arith.index_cast %386 : i32 to index
                  ascvec.storealign_1st %arg12[%387], %385 : memref<1024xui32, 1>, index, <ui32>
                }
                ascvec.mem_bar vst_vld
                %285 = scf.if %arg16 -> (i32) {
                  %286 = arith.remsi %arg14, %c32_i32 : i32
                  %287 = arith.cmpi ne, %286, %c0_i32_14 : i32
                  %288 = arith.cmpi slt, %286, %c0_i32_14 : i32
                  %289 = arith.andi %287, %288 : i1
                  %290 = arith.divsi %arg14, %c32_i32 : i32
                  %291 = arith.extui %289 : i1 to i32
                  %292 = arith.subi %290, %291 : i32
                  %293 = arith.muli %292, %c2_i32_9 : i32
                  scf.yield %293 : i32
                } else {
                  scf.yield %arg15 : i32
                }
                scf.for %arg18 = %c0_i32_14 to %285 step %c64_i32  : i32 {
                  %286 = arith.subi %285, %arg18 : i32
                  %mask, %remaining_out = ascvec.update_mask %286 : !ascvec.mask
                  %287 = ascvec.add_scalar %282, %arg18, %281 : <ui32>, i32
                  scf.if %arg16 {
                    %288 = ascvec.shr %287, 1, %281 : <ui32> -> <ui32>
                    %289 = ascvec.gather %arg12[%c0_13], %288, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %290 = ascvec.duplicate_scalar %c1_i32_10, %281 : i32 -> <ui32>
                    %291 = ascvec.and %282, %290, %281 : <ui32>, <ui32> -> <ui32>
                    %292 = ascvec.cmp_eq_scalar %291, %c0_i32_14, %281 : <ui32>, i32 -> !ascvec.mask
                    %293 = ascvec.duplicate_scalar %c0_i32_14, %281 : i32 -> <ui32>
                    %294 = ascvec.select %289, %293, %292 : <ui32>
                    %295 = arith.index_cast %arg18 : i32 to index
                    %c0_15 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg19 = %c0_15 to %c1 step %c1 {
                      ascvec.storealign_pack %arg17[%295], %294, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  } else {
                    %288 = ascvec.gather %arg12[%c0_13], %287, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %289 = arith.index_cast %arg18 : i32 to index
                    %c0_15 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg19 = %c0_15 to %c1 step %c1 {
                      ascvec.storealign_pack %arg17[%289], %288, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  }
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
            } else {
              %215 = arith.addi %94, %c63_i64 : i64
              %216 = arith.remsi %215, %c64_i64 : i64
              %217 = arith.cmpi ne, %216, %c0_i64 : i64
              %218 = arith.cmpi slt, %216, %c0_i64 : i64
              %219 = arith.andi %217, %218 : i1
              %220 = arith.divsi %215, %c64_i64 : i64
              %221 = arith.extui %219 : i1 to i64
              %222 = arith.subi %220, %221 : i64
              %223 = arith.addi %94, %c1023_i64 : i64
              %224 = arith.remsi %223, %c1024_i64 : i64
              %225 = arith.cmpi ne, %224, %c0_i64 : i64
              %226 = arith.cmpi slt, %224, %c0_i64 : i64
              %227 = arith.andi %225, %226 : i1
              %228 = arith.divsi %223, %c1024_i64 : i64
              %229 = arith.extui %227 : i1 to i64
              %230 = arith.subi %228, %229 : i64
              %231 = arith.trunci %arg4 : i64 to i32
              %232 = arith.extui %231 : i32 to i64
              %233 = arith.subi %arg4, %232 : i64
              %234 = arith.remsi %233, %c4294967296_i64 : i64
              %235 = arith.cmpi ne, %234, %c0_i64 : i64
              %236 = arith.cmpi slt, %234, %c0_i64 : i64
              %237 = arith.andi %235, %236 : i1
              %238 = arith.divsi %233, %c4294967296_i64 : i64
              %239 = arith.extui %237 : i1 to i64
              %240 = arith.subi %238, %239 : i64
              %241 = arith.trunci %240 : i64 to i32
              %242 = arith.remsi %207, %c16_i64 : i64
              %243 = arith.cmpi ne, %242, %c0_i64 : i64
              %244 = arith.cmpi slt, %242, %c0_i64 : i64
              %245 = arith.andi %243, %244 : i1
              %246 = arith.divsi %207, %c16_i64 : i64
              %247 = arith.extui %245 : i1 to i64
              %248 = arith.subi %246, %247 : i64
              %249 = arith.remsi %arg3, %c16_i64 : i64
              %250 = arith.cmpi ne, %249, %c0_i64 : i64
              %251 = arith.cmpi slt, %249, %c0_i64 : i64
              %252 = arith.andi %250, %251 : i1
              %253 = arith.divsi %arg3, %c16_i64 : i64
              %254 = arith.extui %252 : i1 to i64
              %255 = arith.subi %253, %254 : i64
              %256 = arith.remsi %248, %255 : i64
              %257 = arith.cmpi ne, %256, %c0_i64 : i64
              %258 = arith.cmpi slt, %256, %c0_i64 : i64
              %259 = arith.cmpi slt, %255, %c0_i64 : i64
              %260 = arith.xori %258, %259 : i1
              %261 = arith.andi %257, %260 : i1
              %262 = arith.divsi %248, %255 : i64
              %263 = arith.extui %261 : i1 to i64
              %264 = arith.subi %262, %263 : i64
              %265 = arith.select %261, %255, %c0_i64 : i64
              %266 = arith.addi %256, %265 : i64
              %267 = arith.trunci %264 : i64 to i32
              %268 = arith.extui %267 : i32 to i64
              %269 = arith.subi %264, %268 : i64
              %270 = arith.remsi %269, %c4294967296_i64 : i64
              %271 = arith.cmpi ne, %270, %c0_i64 : i64
              %272 = arith.cmpi slt, %270, %c0_i64 : i64
              %273 = arith.andi %271, %272 : i1
              %274 = arith.divsi %269, %c4294967296_i64 : i64
              %275 = arith.extui %273 : i1 to i64
              %276 = arith.subi %274, %275 : i64
              %277 = arith.trunci %276 : i64 to i32
              %278 = arith.trunci %266 : i64 to i32
              %279 = arith.extui %278 : i32 to i64
              %280 = arith.subi %266, %279 : i64
              %281 = arith.remsi %280, %c4294967296_i64 : i64
              %282 = arith.cmpi ne, %281, %c0_i64 : i64
              %283 = arith.cmpi slt, %281, %c0_i64 : i64
              %284 = arith.andi %282, %283 : i1
              %285 = arith.divsi %280, %c4294967296_i64 : i64
              %286 = arith.extui %284 : i1 to i64
              %287 = arith.subi %285, %286 : i64
              %288 = arith.trunci %287 : i64 to i32
              %289 = arith.remsi %c64_i64, %255 : i64
              %290 = arith.cmpi ne, %289, %c0_i64 : i64
              %291 = arith.cmpi slt, %289, %c0_i64 : i64
              %292 = arith.xori %291, %259 : i1
              %293 = arith.andi %290, %292 : i1
              %294 = arith.divsi %c64_i64, %255 : i64
              %295 = arith.extui %293 : i1 to i64
              %296 = arith.subi %294, %295 : i64
              %297 = arith.select %293, %255, %c0_i64 : i64
              %298 = arith.addi %289, %297 : i64
              %299 = arith.trunci %230 : i64 to i32
              ascvec.vf(%255, %arg5, %231, %241, %alloc_7, %298, %296, %299, %267, %277, %278, %288 : i64, i64, i32, i32, memref<8192xi32, 1>, i64, i64, i32, i32, i32, i32, i32) {
              ^bb0(%arg8: i64, %arg9: i64, %arg10: i32, %arg11: i32, %arg12: memref<8192xi32, 1>, %arg13: i64, %arg14: i64, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32):
                %c1_i32_9 = arith.constant 1 : i32
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
                %c1993301258_i32_10 = arith.constant 1993301258 : i32
                %c1013904242_i32_11 = arith.constant 1013904242 : i32
                %c-1150833019_i32_12 = arith.constant -1150833019 : i32
                %c-1640531527_i32_13 = arith.constant -1640531527 : i32
                %c-845247145_i32 = arith.constant -845247145 : i32
                %c-766435501_i32 = arith.constant -766435501 : i32
                %c16383_i32 = arith.constant 16383 : i32
                %c1073741824_i32_14 = arith.constant 1073741824 : i32
                %c4_i32_15 = arith.constant 4 : i32
                %c0_i64_16 = arith.constant 0 : i64
                %c0_i32_17 = arith.constant 0 : i32
                %c256_i32 = arith.constant 256 : i32
                %c4294967296_i64_18 = arith.constant 4294967296 : i64
                %305:4 = scf.for %arg20 = %c0_i32_17 to %arg15 step %c1_i32_9 iter_args(%arg21 = %arg16, %arg22 = %arg17, %arg23 = %arg18, %arg24 = %arg19) -> (i32, i32, i32, i32)  : i32 {
                  %306 = arith.extui %arg21 : i32 to i64
                  %307 = arith.extui %arg22 : i32 to i64
                  %308 = arith.muli %307, %c4294967296_i64_18 : i64
                  %309 = arith.addi %306, %308 : i64
                  %310 = arith.extui %arg23 : i32 to i64
                  %311 = arith.extui %arg24 : i32 to i64
                  %312 = arith.muli %311, %c4294967296_i64_18 : i64
                  %313 = arith.addi %310, %312 : i64
                  %314 = arith.muli %arg20, %c256_i32 : i32
                  %315 = ascvec.create_mask : !ascvec.mask
                  %316 = ascvec.arange %c0_i32_17 : i32 -> <ui32>
                  %317 = arith.muli %309, %arg8 : i64
                  %318 = arith.addi %317, %313 : i64
                  %319 = arith.trunci %318 : i64 to i32
                  %320 = arith.extui %319 : i32 to i64
                  %321 = arith.subi %318, %320 : i64
                  %322 = arith.remsi %321, %c4294967296_i64_18 : i64
                  %323 = arith.cmpi ne, %322, %c0_i64_16 : i64
                  %324 = arith.cmpi slt, %322, %c0_i64_16 : i64
                  %325 = arith.andi %323, %324 : i1
                  %326 = arith.divsi %321, %c4294967296_i64_18 : i64
                  %327 = arith.extui %325 : i1 to i64
                  %328 = arith.subi %326, %327 : i64
                  %329 = arith.trunci %328 : i64 to i32
                  %330 = ascvec.duplicate_scalar %319, %315 : i32 -> <ui32>
                  %331 = ascvec.duplicate_scalar %329, %315 : i32 -> <ui32>
                  %332 = ascvec.duplicate_scalar %c0_i32_17, %315 : i32 -> <ui32>
                  %carry, %dst = ascvec.addc %330, %316, %315 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %carry_19, %dst_20 = ascvec.addc %331, %332, %carry, %315 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %333 = arith.trunci %arg9 : i64 to i32
                  %334 = arith.extui %333 : i32 to i64
                  %335 = arith.subi %arg9, %334 : i64
                  %336 = arith.remsi %335, %c4294967296_i64_18 : i64
                  %337 = arith.cmpi ne, %336, %c0_i64_16 : i64
                  %338 = arith.cmpi slt, %336, %c0_i64_16 : i64
                  %339 = arith.andi %337, %338 : i1
                  %340 = arith.divsi %335, %c4294967296_i64_18 : i64
                  %341 = arith.extui %339 : i1 to i64
                  %342 = arith.subi %340, %341 : i64
                  %343 = arith.trunci %342 : i64 to i32
                  %344 = arith.divui %333, %c4_i32_15 : i32
                  %345 = arith.remui %343, %c4_i32_15 : i32
                  %346 = arith.muli %345, %c1073741824_i32_14 : i32
                  %347 = arith.addi %344, %346 : i32
                  %348 = arith.divui %343, %c4_i32_15 : i32
                  %349 = ascvec.shr %dst, 14, %315 : <ui32> -> <ui32>
                  %350 = ascvec.shl %dst_20, 18, %315 : <ui32> -> <ui32>
                  %351 = ascvec.or %349, %350, %315 : <ui32>, <ui32> -> <ui32>
                  %352 = ascvec.shr %dst_20, 14, %315 : <ui32> -> <ui32>
                  %353 = ascvec.duplicate_scalar %347, %315 : i32 -> <ui32>
                  %354 = ascvec.duplicate_scalar %348, %315 : i32 -> <ui32>
                  %carry_21, %dst_22 = ascvec.addc %353, %351, %315 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %carry_23, %dst_24 = ascvec.addc %354, %352, %carry_21, %315 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %355 = ascvec.duplicate_scalar %c16383_i32, %315 : i32 -> <ui32>
                  %356 = ascvec.and %dst, %355, %315 : <ui32>, <ui32> -> <ui32>
                  %357 = ascvec.duplicate_scalar %c-766435501_i32, %315 : i32 -> <ui32>
                  %358 = ascvec.duplicate_scalar %c-845247145_i32, %315 : i32 -> <ui32>
                  %dst0, %dst1 = ascvec.mull %332, %357, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_25, %dst1_26 = ascvec.mull %dst_22, %358, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %359 = ascvec.xor %dst1_26, %356, %315 : <ui32>, <ui32> -> <ui32>
                  %360 = ascvec.xor %dst1, %dst_24, %315 : <ui32>, <ui32> -> <ui32>
                  %361 = ascvec.duplicate_scalar %arg10, %315 : i32 -> <ui32>
                  %362 = ascvec.duplicate_scalar %arg11, %315 : i32 -> <ui32>
                  %363 = ascvec.xor %359, %361, %315 : <ui32>, <ui32> -> <ui32>
                  %364 = ascvec.xor %360, %362, %315 : <ui32>, <ui32> -> <ui32>
                  %dst0_27, %dst1_28 = ascvec.mull %363, %357, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_29, %dst1_30 = ascvec.mull %364, %358, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %365 = ascvec.xor %dst1_30, %dst0_25, %315 : <ui32>, <ui32> -> <ui32>
                  %366 = ascvec.xor %dst1_28, %dst0, %315 : <ui32>, <ui32> -> <ui32>
                  %367 = arith.addi %arg10, %c-1640531527_i32_13 : i32
                  %368 = ascvec.duplicate_scalar %367, %315 : i32 -> <ui32>
                  %369 = arith.addi %arg11, %c-1150833019_i32_12 : i32
                  %370 = ascvec.duplicate_scalar %369, %315 : i32 -> <ui32>
                  %371 = ascvec.xor %365, %368, %315 : <ui32>, <ui32> -> <ui32>
                  %372 = ascvec.xor %366, %370, %315 : <ui32>, <ui32> -> <ui32>
                  %dst0_31, %dst1_32 = ascvec.mull %371, %357, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_33, %dst1_34 = ascvec.mull %372, %358, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %373 = ascvec.xor %dst1_34, %dst0_29, %315 : <ui32>, <ui32> -> <ui32>
                  %374 = ascvec.xor %dst1_32, %dst0_27, %315 : <ui32>, <ui32> -> <ui32>
                  %375 = arith.addi %arg10, %c1013904242_i32_11 : i32
                  %376 = ascvec.duplicate_scalar %375, %315 : i32 -> <ui32>
                  %377 = arith.addi %arg11, %c1993301258_i32_10 : i32
                  %378 = ascvec.duplicate_scalar %377, %315 : i32 -> <ui32>
                  %379 = ascvec.xor %373, %376, %315 : <ui32>, <ui32> -> <ui32>
                  %380 = ascvec.xor %374, %378, %315 : <ui32>, <ui32> -> <ui32>
                  %dst0_35, %dst1_36 = ascvec.mull %379, %357, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_37, %dst1_38 = ascvec.mull %380, %358, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %381 = ascvec.xor %dst1_38, %dst0_33, %315 : <ui32>, <ui32> -> <ui32>
                  %382 = ascvec.xor %dst1_36, %dst0_31, %315 : <ui32>, <ui32> -> <ui32>
                  %383 = arith.addi %arg10, %c-626627285_i32 : i32
                  %384 = ascvec.duplicate_scalar %383, %315 : i32 -> <ui32>
                  %385 = arith.addi %arg11, %c842468239_i32 : i32
                  %386 = ascvec.duplicate_scalar %385, %315 : i32 -> <ui32>
                  %387 = ascvec.xor %381, %384, %315 : <ui32>, <ui32> -> <ui32>
                  %388 = ascvec.xor %382, %386, %315 : <ui32>, <ui32> -> <ui32>
                  %dst0_39, %dst1_40 = ascvec.mull %387, %357, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_41, %dst1_42 = ascvec.mull %388, %358, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %389 = ascvec.xor %dst1_42, %dst0_37, %315 : <ui32>, <ui32> -> <ui32>
                  %390 = ascvec.xor %dst1_40, %dst0_35, %315 : <ui32>, <ui32> -> <ui32>
                  %391 = arith.addi %arg10, %c2027808484_i32 : i32
                  %392 = ascvec.duplicate_scalar %391, %315 : i32 -> <ui32>
                  %393 = arith.addi %arg11, %c-308364780_i32 : i32
                  %394 = ascvec.duplicate_scalar %393, %315 : i32 -> <ui32>
                  %395 = ascvec.xor %389, %392, %315 : <ui32>, <ui32> -> <ui32>
                  %396 = ascvec.xor %390, %394, %315 : <ui32>, <ui32> -> <ui32>
                  %dst0_43, %dst1_44 = ascvec.mull %395, %357, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_45, %dst1_46 = ascvec.mull %396, %358, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %397 = ascvec.xor %dst1_46, %dst0_41, %315 : <ui32>, <ui32> -> <ui32>
                  %398 = ascvec.xor %dst1_44, %dst0_39, %315 : <ui32>, <ui32> -> <ui32>
                  %399 = arith.addi %arg10, %c387276957_i32 : i32
                  %400 = ascvec.duplicate_scalar %399, %315 : i32 -> <ui32>
                  %401 = arith.addi %arg11, %c-1459197799_i32 : i32
                  %402 = ascvec.duplicate_scalar %401, %315 : i32 -> <ui32>
                  %403 = ascvec.xor %397, %400, %315 : <ui32>, <ui32> -> <ui32>
                  %404 = ascvec.xor %398, %402, %315 : <ui32>, <ui32> -> <ui32>
                  %dst0_47, %dst1_48 = ascvec.mull %403, %357, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_49, %dst1_50 = ascvec.mull %404, %358, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %405 = ascvec.xor %dst1_50, %dst0_45, %315 : <ui32>, <ui32> -> <ui32>
                  %406 = ascvec.xor %dst1_48, %dst0_43, %315 : <ui32>, <ui32> -> <ui32>
                  %407 = arith.addi %arg10, %c-1253254570_i32 : i32
                  %408 = ascvec.duplicate_scalar %407, %315 : i32 -> <ui32>
                  %409 = arith.addi %arg11, %c1684936478_i32 : i32
                  %410 = ascvec.duplicate_scalar %409, %315 : i32 -> <ui32>
                  %411 = ascvec.xor %405, %408, %315 : <ui32>, <ui32> -> <ui32>
                  %412 = ascvec.xor %406, %410, %315 : <ui32>, <ui32> -> <ui32>
                  %dst0_51, %dst1_52 = ascvec.mull %411, %357, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_53, %dst1_54 = ascvec.mull %412, %358, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %413 = ascvec.xor %dst1_54, %dst0_49, %315 : <ui32>, <ui32> -> <ui32>
                  %414 = ascvec.xor %dst1_52, %dst0_47, %315 : <ui32>, <ui32> -> <ui32>
                  %415 = arith.addi %arg10, %c1401181199_i32 : i32
                  %416 = ascvec.duplicate_scalar %415, %315 : i32 -> <ui32>
                  %417 = arith.addi %arg11, %c534103459_i32 : i32
                  %418 = ascvec.duplicate_scalar %417, %315 : i32 -> <ui32>
                  %419 = ascvec.xor %413, %416, %315 : <ui32>, <ui32> -> <ui32>
                  %420 = ascvec.xor %414, %418, %315 : <ui32>, <ui32> -> <ui32>
                  %dst0_55, %dst1_56 = ascvec.mull %419, %357, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_57, %dst1_58 = ascvec.mull %420, %358, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %421 = ascvec.xor %dst1_58, %dst0_53, %315 : <ui32>, <ui32> -> <ui32>
                  %422 = ascvec.xor %dst1_56, %dst0_51, %315 : <ui32>, <ui32> -> <ui32>
                  %423 = arith.addi %arg10, %c-239350328_i32 : i32
                  %424 = ascvec.duplicate_scalar %423, %315 : i32 -> <ui32>
                  %425 = arith.addi %arg11, %c-616729560_i32 : i32
                  %426 = ascvec.duplicate_scalar %425, %315 : i32 -> <ui32>
                  %427 = ascvec.xor %421, %424, %315 : <ui32>, <ui32> -> <ui32>
                  %428 = ascvec.xor %422, %426, %315 : <ui32>, <ui32> -> <ui32>
                  %dst0_59, %dst1_60 = ascvec.mull %427, %357, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_61, %dst1_62 = ascvec.mull %428, %358, %315 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %429 = ascvec.xor %dst1_62, %dst0_57, %315 : <ui32>, <ui32> -> <ui32>
                  %430 = ascvec.xor %dst1_60, %dst0_55, %315 : <ui32>, <ui32> -> <ui32>
                  %431 = arith.addi %arg10, %c-1879881855_i32 : i32
                  %432 = ascvec.duplicate_scalar %431, %315 : i32 -> <ui32>
                  %433 = arith.addi %arg11, %c-1767562579_i32 : i32
                  %434 = ascvec.duplicate_scalar %433, %315 : i32 -> <ui32>
                  %435 = ascvec.xor %429, %432, %315 : <ui32>, <ui32> -> <ui32>
                  %436 = ascvec.xor %430, %434, %315 : <ui32>, <ui32> -> <ui32>
                  %437 = ascvec.reinterpret %435 : <ui32> -> <i32>
                  %438 = arith.index_cast %314 : i32 to index
                  ascvec.storealign %arg12[%438], %437, %315 : memref<8192xi32, 1>, index, <i32>
                  %439 = arith.addi %314, %c64_i32 : i32
                  %440 = ascvec.reinterpret %dst0_61 : <ui32> -> <i32>
                  %441 = arith.index_cast %439 : i32 to index
                  ascvec.storealign %arg12[%441], %440, %315 : memref<8192xi32, 1>, index, <i32>
                  %442 = arith.addi %314, %c128_i32 : i32
                  %443 = ascvec.reinterpret %436 : <ui32> -> <i32>
                  %444 = arith.index_cast %442 : i32 to index
                  ascvec.storealign %arg12[%444], %443, %315 : memref<8192xi32, 1>, index, <i32>
                  %445 = arith.addi %314, %c192_i32 : i32
                  %446 = ascvec.reinterpret %dst0_59 : <ui32> -> <i32>
                  %447 = arith.index_cast %445 : i32 to index
                  ascvec.storealign %arg12[%447], %446, %315 : memref<8192xi32, 1>, index, <i32>
                  %448 = arith.addi %313, %arg13 : i64
                  %449 = arith.cmpi sge, %448, %arg8 : i64
                  %450 = arith.extui %449 : i1 to i64
                  %451 = arith.addi %309, %arg14 : i64
                  %452 = arith.addi %451, %450 : i64
                  %453 = arith.muli %450, %arg8 : i64
                  %454 = arith.subi %448, %453 : i64
                  %455 = arith.trunci %452 : i64 to i32
                  %456 = arith.extui %455 : i32 to i64
                  %457 = arith.subi %452, %456 : i64
                  %458 = arith.remsi %457, %c4294967296_i64_18 : i64
                  %459 = arith.cmpi ne, %458, %c0_i64_16 : i64
                  %460 = arith.cmpi slt, %458, %c0_i64_16 : i64
                  %461 = arith.andi %459, %460 : i1
                  %462 = arith.divsi %457, %c4294967296_i64_18 : i64
                  %463 = arith.extui %461 : i1 to i64
                  %464 = arith.subi %462, %463 : i64
                  %465 = arith.trunci %464 : i64 to i32
                  %466 = arith.trunci %454 : i64 to i32
                  %467 = arith.extui %466 : i32 to i64
                  %468 = arith.subi %454, %467 : i64
                  %469 = arith.remsi %468, %c4294967296_i64_18 : i64
                  %470 = arith.cmpi ne, %469, %c0_i64_16 : i64
                  %471 = arith.cmpi slt, %469, %c0_i64_16 : i64
                  %472 = arith.andi %470, %471 : i1
                  %473 = arith.divsi %468, %c4294967296_i64_18 : i64
                  %474 = arith.extui %472 : i1 to i64
                  %475 = arith.subi %473, %474 : i64
                  %476 = arith.trunci %475 : i64 to i32
                  scf.yield %455, %465, %466, %476 : i32, i32, i32, i32
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              %300 = arith.trunci %222 : i64 to i32
              %301 = arith.trunci %94 : i64 to i32
              %302 = arith.trunci %27 : i64 to i32
              %303 = arith.muli %300, %c2_i32 : i32
              %304 = arith.cmpi eq, %302, %c4_i32 : i32
              ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%94, %129, %alloc_7, %alloc_4, %alloc_6, %300, %301, %303, %304, %alloc_5 : i64, memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, memref<1024xui32, 1>, i32, i32, i32, i1, memref<2048xui8, 1>) {
              ^bb0(%arg8: i64, %arg9: memref<32768xbf16, 1>, %arg10: memref<8192xi32, 1>, %arg11: memref<32768xui8, 1>, %arg12: memref<1024xui32, 1>, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i1, %arg17: memref<2048xui8, 1>):
                %c2_i32_9 = arith.constant 2 : i32
                %c2139095040_i32 = arith.constant 2139095040 : i32
                %c127_i32 = arith.constant 127 : i32
                %c123_i32 = arith.constant 123 : i32
                %c947912704_i32 = arith.constant 947912704 : i32
                %cst = arith.constant 4.2949673E+9 : f32
                %c29360128_i32 = arith.constant 29360128 : i32
                %c128_i32 = arith.constant 128 : i32
                %c65535_i32 = arith.constant 65535 : i32
                %c4_i32_10 = arith.constant 4 : i32
                %c256_i32 = arith.constant 256 : i32
                %c1_i32_11 = arith.constant 1 : i32
                %c16_i32 = arith.constant 16 : i32
                %c3_i32 = arith.constant 3 : i32
                %c2139160576_i32 = arith.constant 2139160576 : i32
                %c254_i32 = arith.constant 254 : i32
                %c-15_i32 = arith.constant -15 : i32
                %c15_i32 = arith.constant 15 : i32
                %c255_i32 = arith.constant 255 : i32
                %c2147483647_i32 = arith.constant 2147483647 : i32
                %cst_12 = arith.constant 0.000000e+00 : f32
                %c0_13 = arith.constant 0 : index
                %c64_i32 = arith.constant 64 : i32
                %c32_i32 = arith.constant 32 : i32
                %c0_i32_14 = arith.constant 0 : i32
                %305 = ascvec.create_mask : !ascvec.mask
                %306 = ascvec.arange %c0_i32_14 : i32 -> <ui32>
                %307 = ascvec.cmp_lt_scalar %306, %c32_i32, %305 : <ui32>, i32 -> !ascvec.mask
                %308 = ascvec.cmp_ge_scalar %306, %c32_i32, %305 : <ui32>, i32 -> !ascvec.mask
                scf.for %arg18 = %c0_i32_14 to %arg13 step %c1_i32_11  : i32 {
                  %310 = arith.muli %arg18, %c64_i32 : i32
                  %311 = arith.trunci %arg8 : i64 to i32
                  %312 = arith.subi %311, %310 : i32
                  %mask, %remaining_out = ascvec.update_mask %312 : !ascvec.mask
                  %313 = arith.index_cast %310 : i32 to index
                  %314 = ascvec.loadalign_unpack %arg9[%313] {mode = "b16_to_b32"} : memref<32768xbf16, 1>, index -> <bf16>
                  %315 = ascvec.cast %314, %305 {rounding = 6 : i32} : <bf16> -> <f32>
                  %316 = ascvec.duplicate_scalar %cst_12, %305 : f32 -> <f32>
                  %317 = ascvec.select %315, %316, %mask : <f32>
                  %318 = ascvec.reinterpret %317 : <f32> -> <ui32>
                  %319 = ascvec.duplicate_scalar %c2147483647_i32, %305 : i32 -> <ui32>
                  %320 = ascvec.and %318, %319, %305 : <ui32>, <ui32> -> <ui32>
                  %321 = ascvec.reduce_max %320, %307 : <ui32>
                  %322 = ascvec.duplicate %321, %305 : <ui32> -> <ui32>
                  %323 = ascvec.reduce_max %320, %308 : <ui32>
                  %324 = ascvec.duplicate %323, %305 : <ui32> -> <ui32>
                  %325 = ascvec.select %322, %324, %307 : <ui32>
                  %326 = ascvec.shr %325, 23, %305 : <ui32> -> <ui32>
                  %327 = ascvec.cmp_ge_scalar %326, %c255_i32, %305 : <ui32>, i32 -> !ascvec.mask
                  %328 = ascvec.max_scalar %326, %c15_i32, %305 : <ui32>, i32
                  %329 = ascvec.add_scalar %328, %c-15_i32, %305 : <ui32>, i32
                  %330 = ascvec.duplicate_scalar %c255_i32, %327 merge %329 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %331 = ascvec.duplicate_scalar %c254_i32, %305 : i32 -> <ui32>
                  %332 = ascvec.sub %331, %330, %305 : <ui32>
                  %333 = ascvec.cmp_ne_scalar %330, %c0_i32_14, %305 : <ui32>, i32 -> !ascvec.mask
                  %334 = ascvec.shl %332, 23, %333 : <ui32> -> <ui32>
                  %335 = ascvec.duplicate_scalar %c2139160576_i32, %327 merge %334 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %336 = ascvec.reinterpret %335 : <ui32> -> <f32>
                  %337 = ascvec.mul %317, %336, %305 : <f32>
                  %338 = ascvec.shr %306, 2, %305 : <ui32> -> <ui32>
                  %339 = ascvec.duplicate_scalar %c3_i32, %305 : i32 -> <ui32>
                  %340 = ascvec.and %338, %339, %305 : <ui32>, <ui32> -> <ui32>
                  %341 = ascvec.shl %340, 6, %305 : <ui32> -> <ui32>
                  %342 = ascvec.shr %306, 4, %305 : <ui32> -> <ui32>
                  %343 = ascvec.add %341, %342, %305 : <ui32>
                  %344 = arith.remsi %arg18, %c16_i32 : i32
                  %345 = arith.cmpi ne, %344, %c0_i32_14 : i32
                  %346 = arith.cmpi slt, %344, %c0_i32_14 : i32
                  %347 = arith.andi %345, %346 : i1
                  %348 = arith.divsi %arg18, %c16_i32 : i32
                  %349 = arith.extui %347 : i1 to i32
                  %350 = arith.subi %348, %349 : i32
                  %351 = arith.muli %350, %c256_i32 : i32
                  %352 = arith.select %347, %c16_i32, %c0_i32_14 : i32
                  %353 = arith.addi %344, %352 overflow<nsw> : i32
                  %354 = arith.muli %353, %c4_i32_10 overflow<nsw> : i32
                  %355 = arith.addi %351, %354 : i32
                  %356 = ascvec.add_scalar %343, %355, %305 : <ui32>, i32
                  %357 = ascvec.gather %arg10[%c0_13], %356, %305 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                  %358 = ascvec.reinterpret %357 : <i32> -> <ui32>
                  %359 = ascvec.shl %358, 8, %305 : <ui32> -> <ui32>
                  %360 = ascvec.shr %358, 24, %305 : <ui32> -> <ui32>
                  %361 = ascvec.or %359, %360, %305 : <ui32>, <ui32> -> <ui32>
                  %362 = ascvec.and %306, %339, %305 : <ui32>, <ui32> -> <ui32>
                  %363 = ascvec.duplicate_scalar %c1_i32_11, %305 : i32 -> <ui32>
                  %364 = ascvec.and %362, %363, %305 : <ui32>, <ui32> -> <ui32>
                  %365 = ascvec.cmp_eq_scalar %364, %c0_i32_14, %305 : <ui32>, i32 -> !ascvec.mask
                  %366 = ascvec.select %358, %361, %365 : <ui32>
                  %367 = ascvec.cmp_eq_scalar %362, %c0_i32_14, %305 : <ui32>, i32 -> !ascvec.mask
                  %368 = ascvec.cmp_eq_scalar %362, %c3_i32, %305 : <ui32>, i32 -> !ascvec.mask
                  %369 = ascvec.mask_or %367, %368, %305
                  %370 = ascvec.shr %366, 16, %305 : <ui32> -> <ui32>
                  %371 = ascvec.duplicate_scalar %c65535_i32, %305 : i32 -> <ui32>
                  %372 = ascvec.and %366, %371, %305 : <ui32>, <ui32> -> <ui32>
                  %373 = ascvec.select %370, %372, %369 : <ui32>
                  %374 = ascvec.reinterpret %337 : <f32> -> <ui32>
                  %375 = ascvec.shr %374, 24, %305 : <ui32> -> <ui32>
                  %376 = ascvec.duplicate_scalar %c128_i32, %305 : i32 -> <ui32>
                  %377 = ascvec.and %375, %376, %305 : <ui32>, <ui32> -> <ui32>
                  %378 = ascvec.and %374, %319, %305 : <ui32>, <ui32> -> <ui32>
                  %379 = ascvec.shr %378, 5, %305 : <ui32> -> <ui32>
                  %380 = ascvec.duplicate_scalar %c29360128_i32, %305 : i32 -> <ui32>
                  %381 = ascvec.sub %379, %380, %305 : <ui32>
                  %382 = ascvec.reinterpret %378 : <ui32> -> <f32>
                  %383 = ascvec.mul_scalar %382, %cst, %305 : <f32>, f32
                  %384 = ascvec.cast %383, %305 {rounding = 4 : i32} : <f32> -> <i32>
                  %385 = ascvec.reinterpret %384 : <i32> -> <ui32>
                  %386 = ascvec.cmp_ge_scalar %378, %c947912704_i32, %305 : <ui32>, i32 -> !ascvec.mask
                  %387 = ascvec.select %381, %385, %386 : <ui32>
                  %388 = ascvec.and %373, %371, %305 : <ui32>, <ui32> -> <ui32>
                  %389 = ascvec.add %387, %388, %305 : <ui32>
                  %390 = ascvec.shr %389, 16, %305 : <ui32> -> <ui32>
                  %391 = ascvec.min_scalar %390, %c123_i32, %305 : <ui32>, i32
                  %392 = ascvec.or %391, %377, %305 : <ui32>, <ui32> -> <ui32>
                  %393 = ascvec.duplicate_scalar %c127_i32, %305 : i32 -> <ui32>
                  %394 = ascvec.cmp_gt_scalar %378, %c2139095040_i32, %305 : <ui32>, i32 -> !ascvec.mask
                  %395 = ascvec.select %393, %392, %394 : <ui32>
                  ascvec.storealign_pack %arg11[%313], %395, %mask {mode = "b32_to_b8"} : memref<32768xui8, 1>, index, <ui32>
                  %396 = arith.muli %arg18, %c2_i32_9 : i32
                  %397 = arith.index_cast %396 : i32 to index
                  ascvec.storealign_1st %arg12[%397], %330 : memref<1024xui32, 1>, index, <ui32>
                  %398 = ascvec.duplicate_scalar %c32_i32, %305 : i32 -> <ui32>
                  %399 = ascvec.gather_reg %330, %398 : <ui32>, <ui32> -> <ui32>
                  %400 = arith.addi %396, %c1_i32_11 : i32
                  %401 = arith.index_cast %400 : i32 to index
                  ascvec.storealign_1st %arg12[%401], %399 : memref<1024xui32, 1>, index, <ui32>
                }
                ascvec.mem_bar vst_vld
                %309 = scf.if %arg16 -> (i32) {
                  %310 = arith.remsi %arg14, %c32_i32 : i32
                  %311 = arith.cmpi ne, %310, %c0_i32_14 : i32
                  %312 = arith.cmpi slt, %310, %c0_i32_14 : i32
                  %313 = arith.andi %311, %312 : i1
                  %314 = arith.divsi %arg14, %c32_i32 : i32
                  %315 = arith.extui %313 : i1 to i32
                  %316 = arith.subi %314, %315 : i32
                  %317 = arith.muli %316, %c2_i32_9 : i32
                  scf.yield %317 : i32
                } else {
                  scf.yield %arg15 : i32
                }
                scf.for %arg18 = %c0_i32_14 to %309 step %c64_i32  : i32 {
                  %310 = arith.subi %309, %arg18 : i32
                  %mask, %remaining_out = ascvec.update_mask %310 : !ascvec.mask
                  %311 = ascvec.add_scalar %306, %arg18, %305 : <ui32>, i32
                  scf.if %arg16 {
                    %312 = ascvec.shr %311, 1, %305 : <ui32> -> <ui32>
                    %313 = ascvec.gather %arg12[%c0_13], %312, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %314 = ascvec.duplicate_scalar %c1_i32_11, %305 : i32 -> <ui32>
                    %315 = ascvec.and %306, %314, %305 : <ui32>, <ui32> -> <ui32>
                    %316 = ascvec.cmp_eq_scalar %315, %c0_i32_14, %305 : <ui32>, i32 -> !ascvec.mask
                    %317 = ascvec.duplicate_scalar %c0_i32_14, %305 : i32 -> <ui32>
                    %318 = ascvec.select %313, %317, %316 : <ui32>
                    %319 = arith.index_cast %arg18 : i32 to index
                    %c0_15 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg19 = %c0_15 to %c1 step %c1 {
                      ascvec.storealign_pack %arg17[%319], %318, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  } else {
                    %312 = ascvec.gather %arg12[%c0_13], %311, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %313 = arith.index_cast %arg18 : i32 to index
                    %c0_15 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg19 = %c0_15 to %c1 step %c1 {
                      ascvec.storealign_pack %arg17[%313], %312, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  }
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
            }
          }
          %134 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %135 = cannir.extract_shape %134 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
          %136 = cannir.extract_dim %135[0] : !cannir.shape<(-1, -1)>
          %137 = arith.minsi %88, %136 : i64
          %138 = arith.subi %136, %137 : i64
          %139 = arith.minsi %138, %c1_i64 : i64
          %140 = arith.cmpi sle, %88, %136 : i64
          %141 = arith.select %140, %139, %c0_i64 : i64
          %142 = arith.cmpi sgt, %141, %c1_i64 : i64
          %143 = arith.select %142, %c1_i64, %141 : i64
          %144 = cannir.extract_dim %135[1] : !cannir.shape<(-1, -1)>
          %145 = arith.divsi %144, %112 : i64
          %146 = arith.minsi %90, %145 : i64
          %147 = arith.muli %146, %65 : i64
          %148 = arith.subi %144, %147 : i64
          %149 = arith.minsi %65, %148 : i64
          %150 = arith.cmpi sle, %90, %145 : i64
          %151 = arith.select %150, %149, %c0_i64 : i64
          %152 = arith.cmpi sgt, %151, %c32768_i64 : i64
          %153 = arith.select %152, %c32768_i64, %151 : i64
          %154 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %155 = cannir.extract_stride %154 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %156 = cannir.extract_dim %155[0] : !cannir.stride<(-1, 1)>
          %157 = arith.muli %88, %156 : i64
          %158 = arith.addi %157, %91 : i64
          %159 = arith.muli %158, %c8_i64 : i64
          %160 = ascscalar.memref_address %alloc_0 : memref<1xui8>
          %161 = arith.divui %159, %c8_i64 : i64
          %162 = arith.addi %160, %161 : i64
          %163 = ascscalar.bind_gm_ptr %162 : memref<1xui8>
          %164 = ascscalar.bitcast_memref %alloc_4 byte_offset = %c0_i64 : memref<32768xui8, 1> -> memref<?xui8, 1>
          %165 = cannir.extract_stride %134 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %166 = cannir.extract_dim %165[0] : !cannir.stride<(-1, 1)>
          %167 = arith.cmpi ugt, %143, %c0_i64 : i64
          %168 = arith.cmpi ugt, %153, %c0_i64 : i64
          %169 = arith.andi %167, %168 : i1
          scf.if %169 {
            ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
            %206 = arith.trunci %143 : i64 to i32
            %207 = arith.trunci %153 : i64 to i32
            ascvec.copy_ub2gm %163[%c0], %164[%c0], %206, %207, %166, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
          }
          %170 = arith.muli %64, %27 : i64
          %171 = cannir.extract_dim %17[0] : !cannir.shape<(-1, -1)>
          %172 = arith.minsi %88, %171 : i64
          %173 = arith.subi %171, %172 : i64
          %174 = arith.minsi %173, %c1_i64 : i64
          %175 = arith.cmpi sle, %88, %171 : i64
          %176 = arith.select %175, %174, %c0_i64 : i64
          %177 = arith.cmpi sgt, %176, %c1_i64 : i64
          %178 = arith.select %177, %c1_i64, %176 : i64
          %179 = arith.maxsi %170, %c1_i64 : i64
          %180 = arith.divsi %18, %179 : i64
          %181 = arith.minsi %90, %180 : i64
          %182 = arith.muli %181, %170 : i64
          %183 = arith.subi %18, %182 : i64
          %184 = arith.minsi %170, %183 : i64
          %185 = arith.cmpi sle, %90, %180 : i64
          %186 = arith.select %185, %184, %c0_i64 : i64
          %187 = arith.cmpi sgt, %186, %c2048_i64 : i64
          %188 = arith.select %187, %c2048_i64, %186 : i64
          %189 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %190 = cannir.extract_stride %189 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %191 = cannir.extract_dim %190[0] : !cannir.stride<(-1, 1)>
          %192 = arith.muli %88, %191 : i64
          %193 = arith.muli %90, %170 : i64
          %194 = arith.addi %192, %193 : i64
          %195 = arith.muli %194, %c8_i64 : i64
          %196 = ascscalar.memref_address %alloc_1 : memref<1xui8>
          %197 = arith.divui %195, %c8_i64 : i64
          %198 = arith.addi %196, %197 : i64
          %199 = ascscalar.bind_gm_ptr %198 : memref<1xui8>
          %200 = ascscalar.bitcast_memref %alloc_5 byte_offset = %c0_i64 : memref<2048xui8, 1> -> memref<?xui8, 1>
          %201 = cannir.extract_stride %16 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %202 = cannir.extract_dim %201[0] : !cannir.stride<(-1, 1)>
          %203 = arith.cmpi ugt, %178, %c0_i64 : i64
          %204 = arith.cmpi ugt, %188, %c0_i64 : i64
          %205 = arith.andi %203, %204 : i1
          scf.if %205 {
            ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
            %206 = arith.trunci %178 : i64 to i32
            %207 = arith.trunci %188 : i64 to i32
            ascvec.copy_ub2gm %199[%c0], %200[%c0], %206, %207, %202, %c2048_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %102 : i64
        }
      }
    }
    return
  }
}
