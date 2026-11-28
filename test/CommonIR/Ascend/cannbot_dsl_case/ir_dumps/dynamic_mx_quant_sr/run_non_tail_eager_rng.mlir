module {
  func.func @run_non_tail_eager_rng_cbc512e164750f345fc0cf9ac158720d(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: i64) attributes {cannbotdsl.contract_digest = "cbc512e164750f345fc0cf9ac158720d306989aec7b5adcea9b39af9c15660b9", cannbotdsl.public_slot_count = 8 : i64, cannbotdsl.slot_dtype_bits = [16, 16, 8, 8, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 4, 1, 1, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 2, 2, 2, 2], cannbotdsl.slot_type_digests = ["d3381cb3edb8f40514b57c6c942d36e84a87723300273978fe7c3f381dbd4f9d", "d3381cb3edb8f40514b57c6c942d36e84a87723300273978fe7c3f381dbd4f9d", "e3a7bc6c4b8cb98d68cf0f892959846f4e8f11c27f2c9e430aec968ae281159e", "9332f9464639b7e579b015309e81477f71c4c6705898c0b7ab38269d65f6d71a", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel_run_non_tail_0(%arg5) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg6, %arg7] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3>, name = "_DynamicMxQuantSrKernel_run_non_tail", op_type = "_DynamicMxQuantSrKernel_run_non_tail", roles = array<i64: 0, 0, 1, 1>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, i64, i64
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel_run_non_tail_0(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: i64, %arg5: i64, %arg6: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel_run_non_tail_0", cannir.resource.ids_prepared} {
    %c1024_i32 = arith.constant 1024 : i32
    %c64_i64 = arith.constant 64 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c16384_i64 = arith.constant 16384 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c1023_i64 = arith.constant 1023 : i64
    %c4294967296_i64 = arith.constant 4294967296 : i64
    %c16_i64 = arith.constant 16 : i64
    %c4_i32 = arith.constant 4 : i32
    %c1073741824_i32 = arith.constant 1073741824 : i32
    %c3449720151_i64 = arith.constant 3449720151 : i64
    %c-1640531527_i32 = arith.constant -1640531527 : i32
    %c-1150833019_i32 = arith.constant -1150833019 : i32
    %c3528531795_i64 = arith.constant 3528531795 : i64
    %c1013904242_i32 = arith.constant 1013904242 : i32
    %c1993301258_i32 = arith.constant 1993301258 : i32
    %c16384_i32 = arith.constant 16384 : i32
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0 = arith.constant 0 : index
    %c63_i64 = arith.constant 63 : i64
    %c8_i64 = arith.constant 8 : i64
    %c255_i64 = arith.constant 255 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %false = arith.constant false
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xbf16>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xui8>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<1xui8>
    %alloc_3 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xbf16, 1>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xbf16, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<16384xui8, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 81920 : i64} : memref<16384xui8, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 98304 : i64} : memref<1024xui8, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 99328 : i64} : memref<1024xui8, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 100352 : i64} : memref<512xui32, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 102400 : i64} : memref<4096xi32, 1>
    %0 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
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
    %11 = arith.muli %10, %c2_i64 : i64
    %12 = arith.muli %11, %9 : i64
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
    %25 = arith.addi %24, %c255_i64 : i64
    %26 = arith.remsi %25, %c256_i64 : i64
    %27 = arith.cmpi ne, %26, %c0_i64 : i64
    %28 = arith.cmpi slt, %26, %c0_i64 : i64
    %29 = arith.andi %27, %28 : i1
    %30 = arith.divsi %25, %c256_i64 : i64
    %31 = arith.extui %29 : i1 to i64
    %32 = arith.subi %30, %31 : i64
    %33 = arith.addi %24, %32 : i64
    %34 = arith.subi %33, %c1_i64 : i64
    %35 = arith.remsi %34, %32 : i64
    %36 = arith.cmpi ne, %35, %c0_i64 : i64
    %37 = arith.cmpi slt, %35, %c0_i64 : i64
    %38 = arith.cmpi slt, %32, %c0_i64 : i64
    %39 = arith.xori %37, %38 : i1
    %40 = arith.andi %36, %39 : i1
    %41 = arith.divsi %34, %32 : i64
    %42 = arith.extui %40 : i1 to i64
    %43 = arith.subi %41, %42 : i64
    %44 = arith.cmpi sge, %43, %c4_i64 : i64
    %45 = scf.if %44 -> (i64) {
      %61 = arith.addi %43, %c3_i64 : i64
      %62 = arith.remsi %61, %c4_i64 : i64
      %63 = arith.cmpi ne, %62, %c0_i64 : i64
      %64 = arith.cmpi slt, %62, %c0_i64 : i64
      %65 = arith.andi %63, %64 : i1
      %66 = arith.divsi %61, %c4_i64 : i64
      %67 = arith.extui %65 : i1 to i64
      %68 = arith.subi %66, %67 : i64
      %69 = arith.muli %68, %c4_i64 : i64
      scf.yield %69 : i64
    } else {
      scf.yield %43 : i64
    }
    %46 = arith.muli %45, %c64_i64 : i64
    %47 = arith.addi %9, %45 : i64
    %48 = arith.subi %47, %c1_i64 : i64
    %49 = arith.remsi %48, %45 : i64
    %50 = arith.cmpi ne, %49, %c0_i64 : i64
    %51 = arith.cmpi slt, %49, %c0_i64 : i64
    %52 = arith.cmpi slt, %45, %c0_i64 : i64
    %53 = arith.xori %51, %52 : i1
    %54 = arith.andi %50, %53 : i1
    %55 = arith.divsi %48, %45 : i64
    %56 = arith.extui %54 : i1 to i64
    %57 = arith.subi %55, %56 : i64
    %58 = arith.muli %11, %57 : i64
    %59 = ascscalar.get_block_idx
    %60:2 = scf.for %arg7 = %59 to %58 step %13 iter_args(%arg8 = %c0_i64, %arg9 = %c0_i64) -> (i64, i64)  : i64 {
      %61 = arith.remsi %arg7, %57 : i64
      %62 = arith.cmpi ne, %61, %c0_i64 : i64
      %63 = arith.cmpi slt, %61, %c0_i64 : i64
      %64 = arith.cmpi slt, %57, %c0_i64 : i64
      %65 = arith.xori %63, %64 : i1
      %66 = arith.andi %62, %65 : i1
      %67 = arith.divsi %arg7, %57 : i64
      %68 = arith.extui %66 : i1 to i64
      %69 = arith.subi %67, %68 : i64
      %70 = arith.remsi %69, %c2_i64 : i64
      %71 = arith.cmpi ne, %70, %c0_i64 : i64
      %72 = arith.cmpi slt, %70, %c0_i64 : i64
      %73 = arith.andi %71, %72 : i1
      %74 = arith.divsi %69, %c2_i64 : i64
      %75 = arith.extui %73 : i1 to i64
      %76 = arith.subi %74, %75 : i64
      %77 = arith.select %73, %c2_i64, %c0_i64 : i64
      %78 = arith.addi %70, %77 overflow<nsw> : i64
      %79 = arith.select %66, %57, %c0_i64 : i64
      %80 = arith.addi %61, %79 : i64
      %81 = arith.muli %80, %46 : i64
      %82 = arith.subi %2, %81 : i64
      %83 = arith.cmpi sgt, %82, %46 : i64
      %84 = arith.select %83, %46, %82 : i64
      %85 = arith.cmpi eq, %78, %c0_i64 : i64
      %86 = scf.if %85 -> (i64) {
        %188 = arith.cmpi eq, %arg8, %c0_i64 : i64
        %189 = ascscalar.memref_address %alloc_3 : memref<16384xbf16, 1>
        %190 = ascscalar.memref_address %alloc_4 : memref<16384xbf16, 1>
        %191 = arith.select %188, %189, %190 : i64
        %192 = arith.cmpi ne, %arg8, %c0_i64 : i64
        %193 = arith.extui %192 : i1 to i64
        %194 = arith.addi %arg8, %c1_i64 overflow<nsw> : i64
        %195 = arith.remui %194, %c2_i64 : i64
        %196 = arith.minsi %76, %10 : i64
        %197 = arith.subi %10, %196 : i64
        %198 = arith.minsi %197, %c1_i64 : i64
        %199 = arith.cmpi sle, %76, %10 : i64
        %200 = arith.select %199, %198, %c0_i64 : i64
        %201 = arith.maxsi %46, %c1_i64 : i64
        %202 = arith.divsi %2, %201 : i64
        %203 = arith.minsi %80, %202 : i64
        %204 = arith.muli %203, %46 : i64
        %205 = arith.subi %2, %204 : i64
        %206 = arith.minsi %46, %205 : i64
        %207 = arith.cmpi sle, %80, %202 : i64
        %208 = arith.select %207, %206, %c0_i64 : i64
        %209 = cannir.extract_stride %0 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
        %210 = cannir.extract_dim %209[0] : !cannir.stride<(-1, 2)>
        %211 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
        %212 = cannir.extract_stride %211 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
        %213 = cannir.extract_dim %212[0] : !cannir.stride<(-1, 2)>
        %214 = arith.muli %76, %213 : i64
        %215 = arith.muli %81, %c2_i64 : i64
        %216 = arith.addi %214, %215 : i64
        %217 = arith.muli %216, %c16_i64 : i64
        %218 = ascscalar.memref_address %alloc : memref<1xbf16>
        %219 = arith.divui %217, %c8_i64 : i64
        %220 = arith.addi %218, %219 : i64
        %221 = ascscalar.bind_gm_ptr %220 : memref<1xbf16>
        %222 = ascscalar.bind_local_ptr %191 : memref<1xbf16, 1>
        %223 = arith.cmpi ugt, %200, %c0_i64 : i64
        %224 = arith.cmpi ugt, %208, %c0_i64 : i64
        %225 = arith.andi %223, %224 : i1
        scf.if %225 {
          ascvec.get_buf mte2, %193, 0 {cannir.buffer.generated_sync}
          %226 = arith.trunci %208 : i64 to i32
          %227 = arith.trunci %200 : i64 to i32
          ascvec.copy_gm2ub_nddma %222[%c0], %221[%c0], %226, %227, %c1_i32, %c1_i32, %c1_i32, %c2_i64, %210, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c16384_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 : memref<1xbf16, 1>, memref<1xbf16>
          ascvec.release_buf mte2, %193, 0 {cannir.buffer.generated_sync}
        }
        scf.yield %195 : i64
      } else {
        %188 = arith.cmpi eq, %arg8, %c0_i64 : i64
        %189 = ascscalar.memref_address %alloc_3 : memref<16384xbf16, 1>
        %190 = ascscalar.memref_address %alloc_4 : memref<16384xbf16, 1>
        %191 = arith.select %188, %189, %190 : i64
        %192 = arith.cmpi ne, %arg8, %c0_i64 : i64
        %193 = arith.extui %192 : i1 to i64
        %194 = arith.addi %arg8, %c1_i64 overflow<nsw> : i64
        %195 = arith.remui %194, %c2_i64 : i64
        %196 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
        %197 = cannir.extract_shape %196 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, -1)>
        %198 = cannir.extract_dim %197[0] : !cannir.shape<(-1, -1)>
        %199 = arith.minsi %76, %198 : i64
        %200 = arith.subi %198, %199 : i64
        %201 = arith.minsi %200, %c1_i64 : i64
        %202 = arith.cmpi sle, %76, %198 : i64
        %203 = arith.select %202, %201, %c0_i64 : i64
        %204 = cannir.extract_dim %197[1] : !cannir.shape<(-1, -1)>
        %205 = arith.maxsi %46, %c1_i64 : i64
        %206 = arith.divsi %204, %205 : i64
        %207 = arith.minsi %80, %206 : i64
        %208 = arith.muli %207, %46 : i64
        %209 = arith.subi %204, %208 : i64
        %210 = arith.minsi %46, %209 : i64
        %211 = arith.cmpi sle, %80, %206 : i64
        %212 = arith.select %211, %210, %c0_i64 : i64
        %213 = cannir.extract_stride %196 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
        %214 = cannir.extract_dim %213[0] : !cannir.stride<(-1, 2)>
        %215 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
        %216 = cannir.extract_stride %215 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
        %217 = cannir.extract_dim %216[0] : !cannir.stride<(-1, 2)>
        %218 = arith.muli %76, %217 : i64
        %219 = arith.muli %81, %c2_i64 : i64
        %220 = arith.addi %218, %219 : i64
        %221 = arith.muli %220, %c16_i64 : i64
        %222 = ascscalar.memref_address %alloc_0 : memref<1xbf16>
        %223 = arith.divui %221, %c8_i64 : i64
        %224 = arith.addi %222, %223 : i64
        %225 = ascscalar.bind_gm_ptr %224 : memref<1xbf16>
        %226 = ascscalar.bind_local_ptr %191 : memref<1xbf16, 1>
        %227 = arith.cmpi ugt, %203, %c0_i64 : i64
        %228 = arith.cmpi ugt, %212, %c0_i64 : i64
        %229 = arith.andi %227, %228 : i1
        scf.if %229 {
          ascvec.get_buf mte2, %193, 0 {cannir.buffer.generated_sync}
          %230 = arith.trunci %212 : i64 to i32
          %231 = arith.trunci %203 : i64 to i32
          ascvec.copy_gm2ub_nddma %226[%c0], %225[%c0], %230, %231, %c1_i32, %c1_i32, %c1_i32, %c2_i64, %214, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c16384_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 : memref<1xbf16, 1>, memref<1xbf16>
          ascvec.release_buf mte2, %193, 0 {cannir.buffer.generated_sync}
        }
        scf.yield %195 : i64
      }
      %87 = arith.cmpi eq, %arg9, %c0_i64 : i64
      %88 = ascscalar.memref_address %alloc_3 : memref<16384xbf16, 1>
      %89 = ascscalar.memref_address %alloc_4 : memref<16384xbf16, 1>
      %90 = arith.select %87, %88, %89 : i64
      %91 = ascscalar.bind_local_ptr %90 : memref<16384xbf16, 1>
      %92 = arith.cmpi ne, %arg9, %c0_i64 : i64
      %93 = arith.extui %92 : i1 to i64
      %94 = arith.addi %arg9, %c1_i64 overflow<nsw> : i64
      %95 = arith.remui %94, %c2_i64 : i64
      %96 = ascscalar.memref_address %alloc_5 : memref<16384xui8, 1>
      %97 = ascscalar.memref_address %alloc_6 : memref<16384xui8, 1>
      %98 = arith.select %87, %96, %97 : i64
      %99 = ascscalar.bind_local_ptr %98 : memref<16384xui8, 1>
      %100 = arith.select %87, %c2_i64, %c3_i64 : i64
      %101 = ascscalar.memref_address %alloc_7 : memref<1024xui8, 1>
      %102 = ascscalar.memref_address %alloc_8 : memref<1024xui8, 1>
      %103 = arith.select %87, %101, %102 : i64
      %104 = ascscalar.bind_local_ptr %103 : memref<1024xui8, 1>
      %105 = arith.select %87, %c4_i64, %c5_i64 : i64
      %106 = arith.muli %69, %2 : i64
      %107 = arith.addi %106, %81 : i64
      %108 = arith.cmpi sge, %84, %c256_i64 : i64
      %109 = arith.remsi %84, %c256_i64 : i64
      %110 = arith.cmpi eq, %109, %c0_i64 : i64
      %111 = arith.select %108, %110, %false : i1
      scf.if %111 {
        %188 = arith.divsi %84, %c256_i64 : i64
        %189 = arith.addi %84, %c1023_i64 : i64
        %190 = arith.remsi %189, %c1024_i64 : i64
        %191 = arith.cmpi ne, %190, %c0_i64 : i64
        %192 = arith.cmpi slt, %190, %c0_i64 : i64
        %193 = arith.andi %191, %192 : i1
        %194 = arith.divsi %189, %c1024_i64 : i64
        %195 = arith.extui %193 : i1 to i64
        %196 = arith.subi %194, %195 : i64
        %197 = arith.trunci %arg5 : i64 to i32
        %198 = arith.extui %197 : i32 to i64
        %199 = arith.subi %arg5, %198 : i64
        %200 = arith.remsi %199, %c4294967296_i64 : i64
        %201 = arith.cmpi ne, %200, %c0_i64 : i64
        %202 = arith.cmpi slt, %200, %c0_i64 : i64
        %203 = arith.andi %201, %202 : i1
        %204 = arith.divsi %199, %c4294967296_i64 : i64
        %205 = arith.extui %203 : i1 to i64
        %206 = arith.subi %204, %205 : i64
        %207 = arith.trunci %206 : i64 to i32
        %208 = arith.divui %107, %c16_i64 : i64
        %209 = arith.trunci %208 : i64 to i32
        %210 = arith.extui %209 : i32 to i64
        %211 = arith.subi %208, %210 : i64
        %212 = arith.remsi %211, %c4294967296_i64 : i64
        %213 = arith.cmpi ne, %212, %c0_i64 : i64
        %214 = arith.cmpi slt, %212, %c0_i64 : i64
        %215 = arith.andi %213, %214 : i1
        %216 = arith.divsi %211, %c4294967296_i64 : i64
        %217 = arith.extui %215 : i1 to i64
        %218 = arith.subi %216, %217 : i64
        %219 = arith.trunci %218 : i64 to i32
        %220 = arith.trunci %arg6 : i64 to i32
        %221 = arith.extui %220 : i32 to i64
        %222 = arith.subi %arg6, %221 : i64
        %223 = arith.remsi %222, %c4294967296_i64 : i64
        %224 = arith.cmpi ne, %223, %c0_i64 : i64
        %225 = arith.cmpi slt, %223, %c0_i64 : i64
        %226 = arith.andi %224, %225 : i1
        %227 = arith.divsi %222, %c4294967296_i64 : i64
        %228 = arith.extui %226 : i1 to i64
        %229 = arith.subi %227, %228 : i64
        %230 = arith.trunci %229 : i64 to i32
        %231 = arith.divui %220, %c4_i32 : i32
        %232 = arith.remui %230, %c4_i32 : i32
        %233 = arith.muli %232, %c1073741824_i32 : i32
        %234 = arith.addi %231, %233 : i32
        %235 = arith.divui %230, %c4_i32 : i32
        %236 = arith.muli %196, %c64_i64 : i64
        %237 = arith.divui %208, %c16384_i64 : i64
        %238 = arith.addi %208, %236 : i64
        %239 = arith.subi %238, %c1_i64 : i64
        %240 = arith.divui %239, %c16384_i64 : i64
        %241 = arith.cmpi eq, %237, %240 : i64
        scf.if %241 {
          %244 = arith.trunci %237 : i64 to i32
          %245 = arith.extui %244 : i32 to i64
          %246 = arith.subi %237, %245 : i64
          %247 = arith.remsi %246, %c4294967296_i64 : i64
          %248 = arith.cmpi ne, %247, %c0_i64 : i64
          %249 = arith.cmpi slt, %247, %c0_i64 : i64
          %250 = arith.andi %248, %249 : i1
          %251 = arith.divsi %246, %c4294967296_i64 : i64
          %252 = arith.extui %250 : i1 to i64
          %253 = arith.subi %251, %252 : i64
          %254 = arith.trunci %253 : i64 to i32
          %255 = arith.addi %234, %244 : i32
          %256 = arith.addi %235, %254 : i32
          %257 = arith.cmpi ult, %255, %234 : i32
          %258 = arith.extui %257 : i1 to i32
          %259 = arith.addi %256, %258 : i32
          %260 = arith.extui %255 : i32 to i64
          %261 = arith.muli %260, %c3449720151_i64 : i64
          %262 = arith.trunci %261 : i64 to i32
          %263 = arith.extui %262 : i32 to i64
          %264 = arith.subi %261, %263 : i64
          %265 = arith.remsi %264, %c4294967296_i64 : i64
          %266 = arith.cmpi ne, %265, %c0_i64 : i64
          %267 = arith.cmpi slt, %265, %c0_i64 : i64
          %268 = arith.andi %266, %267 : i1
          %269 = arith.divsi %264, %c4294967296_i64 : i64
          %270 = arith.extui %268 : i1 to i64
          %271 = arith.subi %269, %270 : i64
          %272 = arith.trunci %271 : i64 to i32
          %273 = arith.xori %259, %207 : i32
          %274 = arith.extui %273 : i32 to i64
          %275 = arith.muli %274, %c3449720151_i64 : i64
          %276 = arith.trunci %275 : i64 to i32
          %277 = arith.extui %276 : i32 to i64
          %278 = arith.subi %275, %277 : i64
          %279 = arith.remsi %278, %c4294967296_i64 : i64
          %280 = arith.cmpi ne, %279, %c0_i64 : i64
          %281 = arith.cmpi slt, %279, %c0_i64 : i64
          %282 = arith.andi %280, %281 : i1
          %283 = arith.divsi %278, %c4294967296_i64 : i64
          %284 = arith.extui %282 : i1 to i64
          %285 = arith.subi %283, %284 : i64
          %286 = arith.trunci %285 : i64 to i32
          %287 = arith.addi %197, %c-1640531527_i32 : i32
          %288 = arith.addi %207, %c-1150833019_i32 : i32
          %289 = arith.xori %286, %262 : i32
          %290 = arith.xori %289, %287 : i32
          %291 = arith.extui %290 : i32 to i64
          %292 = arith.muli %291, %c3528531795_i64 : i64
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
          %304 = arith.xori %272, %197 : i32
          %305 = arith.addi %197, %c1013904242_i32 : i32
          %306 = arith.xori %276, %305 : i32
          %307 = arith.addi %207, %c1993301258_i32 : i32
          %308 = arith.xori %303, %307 : i32
          ascsimt.vf(%209, %304, %288, %306, %308, %197, %207, %293, %alloc_10, %236 : i32, i32, i32, i32, i32, i32, i32, i32, memref<4096xi32, 1>, i64) {
          ^bb0(%arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: memref<4096xi32, 1>, %arg19: i64):
            %c256_i64_11 = arith.constant 256 : i64
            %c3_i64_12 = arith.constant 3 : i64
            %c2_i64_13 = arith.constant 2 : i64
            %c1_i64_14 = arith.constant 1 : i64
            %c4_i64_15 = arith.constant 4 : i64
            %c1_i32_16 = arith.constant 1 : i32
            %c10_i32 = arith.constant 10 : i32
            %c3_i32 = arith.constant 3 : i32
            %c-1150833019_i32_17 = arith.constant -1150833019 : i32
            %c-1640531527_i32_18 = arith.constant -1640531527 : i32
            %c-845247145_i32 = arith.constant -845247145 : i32
            %c-766435501_i32 = arith.constant -766435501 : i32
            %c16384_i32_19 = arith.constant 16384 : i32
            %309 = ascsimt.thread_idx {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
            %310 = arith.extsi %309 {cannir.vf_threads = 256 : i64} : i32 to i64
            scf.for %arg20 = %310 to %arg19 step %c256_i64_11  : i64 {
              %311 = arith.trunci %arg20 : i64 to i32
              %312 = arith.addi %arg10, %311 : i32
              %313 = arith.remui %312, %c16384_i32_19 : i32
              %314 = arith.xori %313, %arg11 : i32
              %315 = arith.muli %314, %c-766435501_i32 : i32
              %316 = "ascsimt.bitcast"(%314) : (i32) -> ui32
              %317 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %318 = ascsimt.math_binary %316, %317, "mulhi" : ui32, ui32 -> ui32
              %319 = "ascsimt.bitcast"(%318) : (ui32) -> i32
              %320 = arith.xori %319, %arg12 : i32
              %321 = arith.muli %320, %c-845247145_i32 : i32
              %322 = "ascsimt.bitcast"(%320) : (i32) -> ui32
              %323 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %324 = ascsimt.math_binary %322, %323, "mulhi" : ui32, ui32 -> ui32
              %325 = "ascsimt.bitcast"(%324) : (ui32) -> i32
              %326 = arith.xori %325, %arg13 : i32
              %327 = arith.xori %315, %arg14 : i32
              %328:4 = scf.for %arg21 = %c3_i32 to %c10_i32 step %c1_i32_16 iter_args(%arg22 = %326, %arg23 = %321, %arg24 = %327, %arg25 = %arg17) -> (i32, i32, i32, i32)  : i32 {
                %337 = arith.muli %arg22, %c-766435501_i32 : i32
                %338 = "ascsimt.bitcast"(%arg22) : (i32) -> ui32
                %339 = ascsimt.math_binary %338, %317, "mulhi" : ui32, ui32 -> ui32
                %340 = "ascsimt.bitcast"(%339) : (ui32) -> i32
                %341 = arith.muli %arg24, %c-845247145_i32 : i32
                %342 = "ascsimt.bitcast"(%arg24) : (i32) -> ui32
                %343 = ascsimt.math_binary %342, %323, "mulhi" : ui32, ui32 -> ui32
                %344 = "ascsimt.bitcast"(%343) : (ui32) -> i32
                %345 = arith.muli %arg21, %c-1640531527_i32_18 : i32
                %346 = arith.addi %arg15, %345 : i32
                %347 = arith.muli %arg21, %c-1150833019_i32_17 : i32
                %348 = arith.addi %arg16, %347 : i32
                %349 = arith.xori %344, %arg23 : i32
                %350 = arith.xori %349, %346 : i32
                %351 = arith.xori %340, %arg25 : i32
                %352 = arith.xori %351, %348 : i32
                scf.yield %350, %341, %352, %337 : i32, i32, i32, i32
              }
              %329 = arith.muli %arg20, %c4_i64_15 : i64
              %330 = arith.index_cast %329 : i64 to index
              ascsimt.store %arg18[%330], %328#0 : memref<4096xi32, 1>, i32
              %331 = arith.addi %329, %c1_i64_14 : i64
              %332 = arith.index_cast %331 : i64 to index
              ascsimt.store %arg18[%332], %328#1 : memref<4096xi32, 1>, i32
              %333 = arith.addi %329, %c2_i64_13 : i64
              %334 = arith.index_cast %333 : i64 to index
              ascsimt.store %arg18[%334], %328#2 : memref<4096xi32, 1>, i32
              %335 = arith.addi %329, %c3_i64_12 : i64
              %336 = arith.index_cast %335 : i64 to index
              ascsimt.store %arg18[%336], %328#3 : memref<4096xi32, 1>, i32
            } {cannir.vf_threads = 256 : i64}
          } {threads = 256 : i64}
        } else {
          ascsimt.vf(%209, %219, %234, %235, %197, %207, %alloc_10, %236 : i32, i32, i32, i32, i32, i32, memref<4096xi32, 1>, i64) {
          ^bb0(%arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: memref<4096xi32, 1>, %arg17: i64):
            %c256_i64_11 = arith.constant 256 : i64
            %c3_i64_12 = arith.constant 3 : i64
            %c2_i64_13 = arith.constant 2 : i64
            %c1_i64_14 = arith.constant 1 : i64
            %c4_i64_15 = arith.constant 4 : i64
            %c1_i32_16 = arith.constant 1 : i32
            %c10_i32 = arith.constant 10 : i32
            %c0_i32_17 = arith.constant 0 : i32
            %c-1150833019_i32_18 = arith.constant -1150833019 : i32
            %c-1640531527_i32_19 = arith.constant -1640531527 : i32
            %c-845247145_i32 = arith.constant -845247145 : i32
            %c-766435501_i32 = arith.constant -766435501 : i32
            %c262144_i32 = arith.constant 262144 : i32
            %c16384_i32_20 = arith.constant 16384 : i32
            %244 = ascsimt.thread_idx {ascsimt.scope = 2 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
            %245 = arith.extsi %244 {cannir.vf_threads = 256 : i64} : i32 to i64
            scf.for %arg18 = %245 to %arg17 step %c256_i64_11  : i64 {
              %246 = arith.trunci %arg18 : i64 to i32
              %247 = arith.addi %arg10, %246 : i32
              %248 = arith.cmpi ult, %247, %arg10 : i32
              %249 = arith.extui %248 : i1 to i32
              %250 = arith.addi %arg11, %249 : i32
              %251 = arith.divui %247, %c16384_i32_20 : i32
              %252 = arith.remui %250, %c16384_i32_20 : i32
              %253 = arith.muli %252, %c262144_i32 : i32
              %254 = arith.addi %251, %253 : i32
              %255 = arith.divui %250, %c16384_i32_20 : i32
              %256 = arith.addi %arg12, %254 : i32
              %257 = arith.addi %arg13, %255 : i32
              %258 = arith.cmpi ult, %256, %arg12 : i32
              %259 = arith.extui %258 : i1 to i32
              %260 = arith.addi %257, %259 : i32
              %261 = arith.remui %247, %c16384_i32_20 : i32
              %262:4 = scf.for %arg19 = %c0_i32_17 to %c10_i32 step %c1_i32_16 iter_args(%arg20 = %c0_i32_17, %arg21 = %261, %arg22 = %256, %arg23 = %260) -> (i32, i32, i32, i32)  : i32 {
                %271 = arith.muli %arg20, %c-766435501_i32 : i32
                %272 = "ascsimt.bitcast"(%arg20) : (i32) -> ui32
                %273 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                %274 = ascsimt.math_binary %272, %273, "mulhi" : ui32, ui32 -> ui32
                %275 = "ascsimt.bitcast"(%274) : (ui32) -> i32
                %276 = arith.muli %arg22, %c-845247145_i32 : i32
                %277 = "ascsimt.bitcast"(%arg22) : (i32) -> ui32
                %278 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                %279 = ascsimt.math_binary %277, %278, "mulhi" : ui32, ui32 -> ui32
                %280 = "ascsimt.bitcast"(%279) : (ui32) -> i32
                %281 = arith.muli %arg19, %c-1640531527_i32_19 : i32
                %282 = arith.addi %arg14, %281 : i32
                %283 = arith.muli %arg19, %c-1150833019_i32_18 : i32
                %284 = arith.addi %arg15, %283 : i32
                %285 = arith.xori %280, %arg21 : i32
                %286 = arith.xori %285, %282 : i32
                %287 = arith.xori %275, %arg23 : i32
                %288 = arith.xori %287, %284 : i32
                scf.yield %286, %276, %288, %271 : i32, i32, i32, i32
              }
              %263 = arith.muli %arg18, %c4_i64_15 : i64
              %264 = arith.index_cast %263 : i64 to index
              ascsimt.store %arg16[%264], %262#0 : memref<4096xi32, 1>, i32
              %265 = arith.addi %263, %c1_i64_14 : i64
              %266 = arith.index_cast %265 : i64 to index
              ascsimt.store %arg16[%266], %262#1 : memref<4096xi32, 1>, i32
              %267 = arith.addi %263, %c2_i64_13 : i64
              %268 = arith.index_cast %267 : i64 to index
              ascsimt.store %arg16[%268], %262#2 : memref<4096xi32, 1>, i32
              %269 = arith.addi %263, %c3_i64_12 : i64
              %270 = arith.index_cast %269 : i64 to index
              ascsimt.store %arg16[%270], %262#3 : memref<4096xi32, 1>, i32
            } {cannir.vf_threads = 256 : i64}
          } {threads = 256 : i64}
        }
        %242 = arith.trunci %188 : i64 to i32
        ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %93, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
        %243 = ascscalar.bitcast_memref %91 byte_offset = %c0_i64 : memref<16384xbf16, 1> -> memref<?xui8, 1>
        ascvec.vf(%104, %91, %243, %alloc_10, %99, %242 : memref<1024xui8, 1>, memref<16384xbf16, 1>, memref<?xui8, 1>, memref<4096xi32, 1>, memref<16384xui8, 1>, i32) {
        ^bb0(%arg10: memref<1024xui8, 1>, %arg11: memref<16384xbf16, 1>, %arg12: memref<?xui8, 1>, %arg13: memref<4096xi32, 1>, %arg14: memref<16384xui8, 1>, %arg15: i32):
          %c512_i32 = arith.constant 512 : i32
          %c1_i32_11 = arith.constant 1 : i32
          %c2139062143_i32 = arith.constant 2139062143 : i32
          %c-8323200_i32 = arith.constant -8323200 : i32
          %c123_i8 = arith.constant 123 : i8
          %c-1_i8 = arith.constant -1 : i8
          %c60_i8 = arith.constant 60 : i8
          %cst = arith.constant 6.10351526E-5 : f32
          %c64_i32 = arith.constant 64 : i32
          %c0_i16 = arith.constant 0 : i16
          %c0_i32_12 = arith.constant 0 : i32
          %c254_i32 = arith.constant 254 : i32
          %c-15_i32 = arith.constant -15 : i32
          %c15_i32 = arith.constant 15 : i32
          %c255_i32 = arith.constant 255 : i32
          %c32767_i16 = arith.constant 32767 : i16
          %c256_i32 = arith.constant 256 : i32
          %c0_13 = arith.constant 0 : index
          %c1_i8 = arith.constant 1 : i8
          %c32_i8 = arith.constant 32 : i8
          %c65535_i32 = arith.constant 65535 : i32
          %c3_i8 = arith.constant 3 : i8
          %c0_i8 = arith.constant 0 : i8
          %c-128_i8 = arith.constant -128 : i8
          %244 = ascvec.create_mask : !ascvec.mask
          %245 = ascvec.create_mask elem_bits 16 : !ascvec.mask
          %246 = ascvec.create_mask elem_bits 8 : !ascvec.mask
          %247 = ascvec.duplicate_scalar %c-128_i8, %246 : i8 -> <ui8>
          %248 = ascvec.arange %c0_i8 : i8 -> <ui8>
          %249 = ascvec.duplicate_scalar %c3_i8, %246 : i8 -> <ui8>
          %250 = ascvec.and %248, %249, %246 : <ui8>, <ui8> -> <ui8>
          %251 = ascvec.shl %250, 6, %246 : <ui8> -> <ui8>
          %252 = ascvec.shr %248, 2, %246 : <ui8> -> <ui8>
          %253 = ascvec.or %251, %252, %246 : <ui8>, <ui8> -> <ui8>
          %254 = ascvec.duplicate_scalar %c65535_i32, %244 : i32 -> <ui32>
          %255 = ascvec.cmp_lt_scalar %248, %c32_i8, %246 : <ui8>, i8 -> !ascvec.mask
          %256 = ascvec.cmp_lt_scalar %250, %c1_i8, %246 : <ui8>, i8 -> !ascvec.mask
          %257 = ascvec.mask_and %255, %256, %246
          ascvec.store_unalign_begin %arg10 : memref<1024xui8, 1>
          scf.for %arg16 = %c0_i32_12 to %arg15 step %c1_i32_11  : i32 {
            %258 = arith.muli %arg16, %c256_i32 : i32
            %259 = arith.index_cast %258 : i32 to index
            %result0, %result1 = ascvec.loadalign_deintlv %arg11[%259] {dintlv_width = "b16"} : memref<16384xbf16, 1>, index -> <bf16>, <bf16>
            %260 = ascvec.duplicate_scalar %c32767_i16, %245 : i16 -> <ui16>
            %261 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
            %262 = ascvec.and %261, %260, %245 : <ui16>, <ui16> -> <ui16>
            %263 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
            %264 = ascvec.and %263, %260, %245 : <ui16>, <ui16> -> <ui16>
            %265 = ascvec.max %262, %264, %245 : <ui16>
            %266 = ascvec.reduce_max_datablock %265, %245 : <ui16>
            %267 = ascvec.unpack %266 {part = "lower"} : <ui16> -> <ui32>
            %268 = ascvec.shl %267, 16, %244 : <ui32> -> <ui32>
            %269 = arith.muli %arg16, %c512_i32 : i32
            %270 = arith.index_cast %269 : i32 to index
            %result0_15, %result1_16 = ascvec.loadalign_deintlv %arg12[%270] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
            %271 = ascvec.and %result1_16, %247, %246 : <ui8>, <ui8> -> <ui8>
            %272 = ascvec.reinterpret %271 : <ui8> -> <ui32>
            %273 = ascvec.shr %268, 23, %244 : <ui32> -> <ui32>
            %274 = ascvec.cmp_ge_scalar %273, %c255_i32, %244 : <ui32>, i32 -> !ascvec.mask
            %275 = ascvec.max_scalar %273, %c15_i32, %244 : <ui32>, i32
            %276 = ascvec.add_scalar %275, %c-15_i32, %244 : <ui32>, i32
            %277 = ascvec.duplicate_scalar %c255_i32, %274 merge %276 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %278 = ascvec.duplicate_scalar %c254_i32, %244 : i32 -> <ui32>
            %279 = ascvec.sub %278, %277, %244 : <ui32>
            %280 = ascvec.cmp_ne_scalar %277, %c0_i32_12, %244 : <ui32>, i32 -> !ascvec.mask
            %281 = ascvec.shl %279, 23, %280 : <ui32> -> <ui32>
            %282 = ascvec.reinterpret %281 : <ui32> -> <f32>
            %283 = ascvec.reinterpret %282 : <f32> -> <ui32>
            %284 = ascvec.shr %283, 16, %244 : <ui32> -> <ui32>
            %285 = ascvec.pack %284 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %286 = ascvec.arange %c0_i16 : i16 -> <ui16>
            %287 = ascvec.shr %286, 4, %245 : <ui16> -> <ui16>
            %288 = ascvec.gather_reg %285, %287 : <ui16>, <ui16> -> <ui16>
            %289 = ascvec.reinterpret %288 : <ui16> -> <bf16>
            %290 = ascvec.reinterpret %262 : <ui16> -> <bf16>
            %291 = ascvec.mul %290, %289, %245 : <bf16>
            %292 = ascvec.reinterpret %264 : <ui16> -> <bf16>
            %293 = ascvec.mul %292, %289, %245 : <bf16>
            %294 = ascvec.cast %291, %245 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
            %295 = ascvec.cast %291, %245 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
            %296 = ascvec.cast %293, %245 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
            %297 = ascvec.cast %293, %245 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
            %298 = ascvec.reinterpret %289 : <bf16> -> <ui32>
            %299 = arith.muli %arg16, %c64_i32 : i32
            %300 = arith.index_cast %299 : i32 to index
            %301 = ascvec.loadalign %arg13[%300] : memref<4096xi32, 1>, index -> <i32>
            %302 = ascvec.reinterpret %301 : <i32> -> <ui32>
            %303 = ascvec.shl %302, 8, %244 : <ui32> -> <ui32>
            %304 = ascvec.shr %302, 24, %244 : <ui32> -> <ui32>
            %305 = ascvec.or %303, %304, %244 : <ui32>, <ui32> -> <ui32>
            %306 = ascvec.shr %302, 16, %244 : <ui32> -> <ui32>
            %307 = ascvec.and %305, %254, %244 : <ui32>, <ui32> -> <ui32>
            %308 = ascvec.and %302, %254, %244 : <ui32>, <ui32> -> <ui32>
            %309 = ascvec.shr %305, 16, %244 : <ui32> -> <ui32>
            %310 = ascvec.reinterpret %294 : <f32> -> <ui32>
            %311 = ascvec.reinterpret %310 : <ui32> -> <f32>
            %312 = ascvec.max_scalar %311, %cst, %244 : <f32>, f32
            %313 = ascvec.add %311, %312, %244 : <f32>
            %314 = ascvec.reinterpret %313 : <f32> -> <ui32>
            %315 = ascvec.shr %314, 5, %244 : <ui32> -> <ui32>
            %316 = ascvec.add %315, %306, %244 : <ui32>
            %317 = ascvec.reinterpret %296 : <f32> -> <ui32>
            %318 = ascvec.reinterpret %317 : <ui32> -> <f32>
            %319 = ascvec.max_scalar %318, %cst, %244 : <f32>, f32
            %320 = ascvec.add %318, %319, %244 : <f32>
            %321 = ascvec.reinterpret %320 : <f32> -> <ui32>
            %322 = ascvec.shr %321, 5, %244 : <ui32> -> <ui32>
            %323 = ascvec.add %322, %307, %244 : <ui32>
            %324 = ascvec.reinterpret %295 : <f32> -> <ui32>
            %325 = ascvec.reinterpret %324 : <ui32> -> <f32>
            %326 = ascvec.max_scalar %325, %cst, %244 : <f32>, f32
            %327 = ascvec.add %325, %326, %244 : <f32>
            %328 = ascvec.reinterpret %327 : <f32> -> <ui32>
            %329 = ascvec.shr %328, 5, %244 : <ui32> -> <ui32>
            %330 = ascvec.add %329, %308, %244 : <ui32>
            %331 = ascvec.reinterpret %297 : <f32> -> <ui32>
            %332 = ascvec.reinterpret %331 : <ui32> -> <f32>
            %333 = ascvec.max_scalar %332, %cst, %244 : <f32>, f32
            %334 = ascvec.add %332, %333, %244 : <f32>
            %335 = ascvec.reinterpret %334 : <f32> -> <ui32>
            %336 = ascvec.shr %335, 5, %244 : <ui32> -> <ui32>
            %337 = ascvec.add %336, %309, %244 : <ui32>
            %338 = ascvec.reinterpret %316 : <ui32> -> <ui16>
            %339 = ascvec.reinterpret %323 : <ui32> -> <ui16>
            %result0_17, %result1_18 = ascvec.deintlv %338, %339 : <ui16>, <ui16> -> <ui16>, <ui16>
            %340 = ascvec.reinterpret %330 : <ui32> -> <ui16>
            %341 = ascvec.reinterpret %337 : <ui32> -> <ui16>
            %result0_19, %result1_20 = ascvec.deintlv %340, %341 : <ui16>, <ui16> -> <ui16>, <ui16>
            %342 = ascvec.reinterpret %result1_18 : <ui16> -> <ui8>
            %343 = ascvec.reinterpret %result1_20 : <ui16> -> <ui8>
            %result0_21, %result1_22 = ascvec.deintlv %342, %343 : <ui8>, <ui8> -> <ui8>, <ui8>
            %344 = ascvec.gather_reg %result0_21, %253 : <ui8>, <ui8> -> <ui8>
            %345 = ascvec.reinterpret %344 : <ui8> -> <ui32>
            %346 = ascvec.reinterpret %345 : <ui32> -> <ui8>
            %347 = ascvec.add_scalar %346, %c60_i8, %246 : <ui8>, i8
            %348 = ascvec.cmp_ne_scalar %347, %c-1_i8, %246 : <ui8>, i8 -> !ascvec.mask
            %349 = ascvec.min_scalar %347, %c123_i8, %348 : <ui8>, i8
            %350 = ascvec.reinterpret %349 : <ui8> -> <ui32>
            %351 = ascvec.or %350, %272, %244 : <ui32>, <ui32> -> <ui32>
            %352 = ascvec.reinterpret %298 : <ui32> -> <ui32>
            %353 = ascvec.cmp_eq_scalar %352, %c-8323200_i32, %244 : <ui32>, i32 -> !ascvec.mask
            %354 = ascvec.duplicate_scalar %c2139062143_i32, %353 merge %351 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %355 = ascvec.reinterpret %354 : <ui32> -> <ui8>
            ascvec.storealign %arg14[%259], %355, %246 : memref<16384xui8, 1>, index, <ui8>
            %356 = ascvec.reinterpret %277 : <ui32> -> <ui8>
            %357 = ascvec.squeeze_store_reg %356, %257 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg10[%c0_13], %357 : memref<1024xui8, 1>, <ui8>
          }
          %c0_14 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg16 = %c0_14 to %c1 step %c1 {
            ascvec.store_unalign_post %arg10[%c0_13] : memref<1024xui8, 1>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %93, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
      } else {
        %188 = arith.addi %84, %c63_i64 : i64
        %189 = arith.remsi %188, %c64_i64 : i64
        %190 = arith.cmpi ne, %189, %c0_i64 : i64
        %191 = arith.cmpi slt, %189, %c0_i64 : i64
        %192 = arith.andi %190, %191 : i1
        %193 = arith.divsi %188, %c64_i64 : i64
        %194 = arith.extui %192 : i1 to i64
        %195 = arith.subi %193, %194 : i64
        %196 = arith.addi %84, %c1023_i64 : i64
        %197 = arith.remsi %196, %c1024_i64 : i64
        %198 = arith.cmpi ne, %197, %c0_i64 : i64
        %199 = arith.cmpi slt, %197, %c0_i64 : i64
        %200 = arith.andi %198, %199 : i1
        %201 = arith.divsi %196, %c1024_i64 : i64
        %202 = arith.extui %200 : i1 to i64
        %203 = arith.subi %201, %202 : i64
        %204 = arith.trunci %arg5 : i64 to i32
        %205 = arith.extui %204 : i32 to i64
        %206 = arith.subi %arg5, %205 : i64
        %207 = arith.remsi %206, %c4294967296_i64 : i64
        %208 = arith.cmpi ne, %207, %c0_i64 : i64
        %209 = arith.cmpi slt, %207, %c0_i64 : i64
        %210 = arith.andi %208, %209 : i1
        %211 = arith.divsi %206, %c4294967296_i64 : i64
        %212 = arith.extui %210 : i1 to i64
        %213 = arith.subi %211, %212 : i64
        %214 = arith.trunci %213 : i64 to i32
        %215 = arith.remsi %107, %c16_i64 : i64
        %216 = arith.cmpi ne, %215, %c0_i64 : i64
        %217 = arith.cmpi slt, %215, %c0_i64 : i64
        %218 = arith.andi %216, %217 : i1
        %219 = arith.divsi %107, %c16_i64 : i64
        %220 = arith.extui %218 : i1 to i64
        %221 = arith.subi %219, %220 : i64
        %222 = arith.remsi %arg4, %c16_i64 : i64
        %223 = arith.cmpi ne, %222, %c0_i64 : i64
        %224 = arith.cmpi slt, %222, %c0_i64 : i64
        %225 = arith.andi %223, %224 : i1
        %226 = arith.divsi %arg4, %c16_i64 : i64
        %227 = arith.extui %225 : i1 to i64
        %228 = arith.subi %226, %227 : i64
        %229 = arith.remsi %221, %228 : i64
        %230 = arith.cmpi ne, %229, %c0_i64 : i64
        %231 = arith.cmpi slt, %229, %c0_i64 : i64
        %232 = arith.cmpi slt, %228, %c0_i64 : i64
        %233 = arith.xori %231, %232 : i1
        %234 = arith.andi %230, %233 : i1
        %235 = arith.divsi %221, %228 : i64
        %236 = arith.extui %234 : i1 to i64
        %237 = arith.subi %235, %236 : i64
        %238 = arith.select %234, %228, %c0_i64 : i64
        %239 = arith.addi %229, %238 : i64
        %240 = arith.trunci %237 : i64 to i32
        %241 = arith.extui %240 : i32 to i64
        %242 = arith.subi %237, %241 : i64
        %243 = arith.remsi %242, %c4294967296_i64 : i64
        %244 = arith.cmpi ne, %243, %c0_i64 : i64
        %245 = arith.cmpi slt, %243, %c0_i64 : i64
        %246 = arith.andi %244, %245 : i1
        %247 = arith.divsi %242, %c4294967296_i64 : i64
        %248 = arith.extui %246 : i1 to i64
        %249 = arith.subi %247, %248 : i64
        %250 = arith.trunci %249 : i64 to i32
        %251 = arith.trunci %239 : i64 to i32
        %252 = arith.extui %251 : i32 to i64
        %253 = arith.subi %239, %252 : i64
        %254 = arith.remsi %253, %c4294967296_i64 : i64
        %255 = arith.cmpi ne, %254, %c0_i64 : i64
        %256 = arith.cmpi slt, %254, %c0_i64 : i64
        %257 = arith.andi %255, %256 : i1
        %258 = arith.divsi %253, %c4294967296_i64 : i64
        %259 = arith.extui %257 : i1 to i64
        %260 = arith.subi %258, %259 : i64
        %261 = arith.trunci %260 : i64 to i32
        %262 = arith.remsi %c64_i64, %228 : i64
        %263 = arith.cmpi ne, %262, %c0_i64 : i64
        %264 = arith.cmpi slt, %262, %c0_i64 : i64
        %265 = arith.xori %264, %232 : i1
        %266 = arith.andi %263, %265 : i1
        %267 = arith.divsi %c64_i64, %228 : i64
        %268 = arith.extui %266 : i1 to i64
        %269 = arith.subi %267, %268 : i64
        %270 = arith.select %266, %228, %c0_i64 : i64
        %271 = arith.addi %262, %270 : i64
        %272 = arith.trunci %203 : i64 to i32
        ascvec.vf(%228, %arg6, %204, %214, %alloc_10, %271, %269, %272, %240, %250, %251, %261 : i64, i64, i32, i32, memref<4096xi32, 1>, i64, i64, i32, i32, i32, i32, i32) {
        ^bb0(%arg10: i64, %arg11: i64, %arg12: i32, %arg13: i32, %arg14: memref<4096xi32, 1>, %arg15: i64, %arg16: i64, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32):
          %c1_i32_11 = arith.constant 1 : i32
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
          %c1993301258_i32_12 = arith.constant 1993301258 : i32
          %c1013904242_i32_13 = arith.constant 1013904242 : i32
          %c-1150833019_i32_14 = arith.constant -1150833019 : i32
          %c-1640531527_i32_15 = arith.constant -1640531527 : i32
          %c-845247145_i32 = arith.constant -845247145 : i32
          %c-766435501_i32 = arith.constant -766435501 : i32
          %c16383_i32 = arith.constant 16383 : i32
          %c1073741824_i32_16 = arith.constant 1073741824 : i32
          %c4_i32_17 = arith.constant 4 : i32
          %c0_i64_18 = arith.constant 0 : i64
          %c0_i32_19 = arith.constant 0 : i32
          %c256_i32 = arith.constant 256 : i32
          %c4294967296_i64_20 = arith.constant 4294967296 : i64
          %275:4 = scf.for %arg22 = %c0_i32_19 to %arg17 step %c1_i32_11 iter_args(%arg23 = %arg18, %arg24 = %arg19, %arg25 = %arg20, %arg26 = %arg21) -> (i32, i32, i32, i32)  : i32 {
            %276 = arith.extui %arg23 : i32 to i64
            %277 = arith.extui %arg24 : i32 to i64
            %278 = arith.muli %277, %c4294967296_i64_20 : i64
            %279 = arith.addi %276, %278 : i64
            %280 = arith.extui %arg25 : i32 to i64
            %281 = arith.extui %arg26 : i32 to i64
            %282 = arith.muli %281, %c4294967296_i64_20 : i64
            %283 = arith.addi %280, %282 : i64
            %284 = arith.muli %arg22, %c256_i32 : i32
            %285 = ascvec.create_mask : !ascvec.mask
            %286 = ascvec.arange %c0_i32_19 : i32 -> <ui32>
            %287 = arith.muli %279, %arg10 : i64
            %288 = arith.addi %287, %283 : i64
            %289 = arith.trunci %288 : i64 to i32
            %290 = arith.extui %289 : i32 to i64
            %291 = arith.subi %288, %290 : i64
            %292 = arith.remsi %291, %c4294967296_i64_20 : i64
            %293 = arith.cmpi ne, %292, %c0_i64_18 : i64
            %294 = arith.cmpi slt, %292, %c0_i64_18 : i64
            %295 = arith.andi %293, %294 : i1
            %296 = arith.divsi %291, %c4294967296_i64_20 : i64
            %297 = arith.extui %295 : i1 to i64
            %298 = arith.subi %296, %297 : i64
            %299 = arith.trunci %298 : i64 to i32
            %300 = ascvec.duplicate_scalar %289, %285 : i32 -> <ui32>
            %301 = ascvec.duplicate_scalar %299, %285 : i32 -> <ui32>
            %302 = ascvec.duplicate_scalar %c0_i32_19, %285 : i32 -> <ui32>
            %carry, %dst = ascvec.addc %300, %286, %285 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %carry_21, %dst_22 = ascvec.addc %301, %302, %carry, %285 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %303 = arith.trunci %arg11 : i64 to i32
            %304 = arith.extui %303 : i32 to i64
            %305 = arith.subi %arg11, %304 : i64
            %306 = arith.remsi %305, %c4294967296_i64_20 : i64
            %307 = arith.cmpi ne, %306, %c0_i64_18 : i64
            %308 = arith.cmpi slt, %306, %c0_i64_18 : i64
            %309 = arith.andi %307, %308 : i1
            %310 = arith.divsi %305, %c4294967296_i64_20 : i64
            %311 = arith.extui %309 : i1 to i64
            %312 = arith.subi %310, %311 : i64
            %313 = arith.trunci %312 : i64 to i32
            %314 = arith.divui %303, %c4_i32_17 : i32
            %315 = arith.remui %313, %c4_i32_17 : i32
            %316 = arith.muli %315, %c1073741824_i32_16 : i32
            %317 = arith.addi %314, %316 : i32
            %318 = arith.divui %313, %c4_i32_17 : i32
            %319 = ascvec.shr %dst, 14, %285 : <ui32> -> <ui32>
            %320 = ascvec.shl %dst_22, 18, %285 : <ui32> -> <ui32>
            %321 = ascvec.or %319, %320, %285 : <ui32>, <ui32> -> <ui32>
            %322 = ascvec.shr %dst_22, 14, %285 : <ui32> -> <ui32>
            %323 = ascvec.duplicate_scalar %317, %285 : i32 -> <ui32>
            %324 = ascvec.duplicate_scalar %318, %285 : i32 -> <ui32>
            %carry_23, %dst_24 = ascvec.addc %323, %321, %285 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %carry_25, %dst_26 = ascvec.addc %324, %322, %carry_23, %285 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %325 = ascvec.duplicate_scalar %c16383_i32, %285 : i32 -> <ui32>
            %326 = ascvec.and %dst, %325, %285 : <ui32>, <ui32> -> <ui32>
            %327 = ascvec.duplicate_scalar %c-766435501_i32, %285 : i32 -> <ui32>
            %328 = ascvec.duplicate_scalar %c-845247145_i32, %285 : i32 -> <ui32>
            %dst0, %dst1 = ascvec.mull %302, %327, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_27, %dst1_28 = ascvec.mull %dst_24, %328, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %329 = ascvec.xor %dst1_28, %326, %285 : <ui32>, <ui32> -> <ui32>
            %330 = ascvec.xor %dst1, %dst_26, %285 : <ui32>, <ui32> -> <ui32>
            %331 = ascvec.duplicate_scalar %arg12, %285 : i32 -> <ui32>
            %332 = ascvec.duplicate_scalar %arg13, %285 : i32 -> <ui32>
            %333 = ascvec.xor %329, %331, %285 : <ui32>, <ui32> -> <ui32>
            %334 = ascvec.xor %330, %332, %285 : <ui32>, <ui32> -> <ui32>
            %dst0_29, %dst1_30 = ascvec.mull %333, %327, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_31, %dst1_32 = ascvec.mull %334, %328, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %335 = ascvec.xor %dst1_32, %dst0_27, %285 : <ui32>, <ui32> -> <ui32>
            %336 = ascvec.xor %dst1_30, %dst0, %285 : <ui32>, <ui32> -> <ui32>
            %337 = arith.addi %arg12, %c-1640531527_i32_15 : i32
            %338 = ascvec.duplicate_scalar %337, %285 : i32 -> <ui32>
            %339 = arith.addi %arg13, %c-1150833019_i32_14 : i32
            %340 = ascvec.duplicate_scalar %339, %285 : i32 -> <ui32>
            %341 = ascvec.xor %335, %338, %285 : <ui32>, <ui32> -> <ui32>
            %342 = ascvec.xor %336, %340, %285 : <ui32>, <ui32> -> <ui32>
            %dst0_33, %dst1_34 = ascvec.mull %341, %327, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_35, %dst1_36 = ascvec.mull %342, %328, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %343 = ascvec.xor %dst1_36, %dst0_31, %285 : <ui32>, <ui32> -> <ui32>
            %344 = ascvec.xor %dst1_34, %dst0_29, %285 : <ui32>, <ui32> -> <ui32>
            %345 = arith.addi %arg12, %c1013904242_i32_13 : i32
            %346 = ascvec.duplicate_scalar %345, %285 : i32 -> <ui32>
            %347 = arith.addi %arg13, %c1993301258_i32_12 : i32
            %348 = ascvec.duplicate_scalar %347, %285 : i32 -> <ui32>
            %349 = ascvec.xor %343, %346, %285 : <ui32>, <ui32> -> <ui32>
            %350 = ascvec.xor %344, %348, %285 : <ui32>, <ui32> -> <ui32>
            %dst0_37, %dst1_38 = ascvec.mull %349, %327, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_39, %dst1_40 = ascvec.mull %350, %328, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %351 = ascvec.xor %dst1_40, %dst0_35, %285 : <ui32>, <ui32> -> <ui32>
            %352 = ascvec.xor %dst1_38, %dst0_33, %285 : <ui32>, <ui32> -> <ui32>
            %353 = arith.addi %arg12, %c-626627285_i32 : i32
            %354 = ascvec.duplicate_scalar %353, %285 : i32 -> <ui32>
            %355 = arith.addi %arg13, %c842468239_i32 : i32
            %356 = ascvec.duplicate_scalar %355, %285 : i32 -> <ui32>
            %357 = ascvec.xor %351, %354, %285 : <ui32>, <ui32> -> <ui32>
            %358 = ascvec.xor %352, %356, %285 : <ui32>, <ui32> -> <ui32>
            %dst0_41, %dst1_42 = ascvec.mull %357, %327, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_43, %dst1_44 = ascvec.mull %358, %328, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %359 = ascvec.xor %dst1_44, %dst0_39, %285 : <ui32>, <ui32> -> <ui32>
            %360 = ascvec.xor %dst1_42, %dst0_37, %285 : <ui32>, <ui32> -> <ui32>
            %361 = arith.addi %arg12, %c2027808484_i32 : i32
            %362 = ascvec.duplicate_scalar %361, %285 : i32 -> <ui32>
            %363 = arith.addi %arg13, %c-308364780_i32 : i32
            %364 = ascvec.duplicate_scalar %363, %285 : i32 -> <ui32>
            %365 = ascvec.xor %359, %362, %285 : <ui32>, <ui32> -> <ui32>
            %366 = ascvec.xor %360, %364, %285 : <ui32>, <ui32> -> <ui32>
            %dst0_45, %dst1_46 = ascvec.mull %365, %327, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_47, %dst1_48 = ascvec.mull %366, %328, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %367 = ascvec.xor %dst1_48, %dst0_43, %285 : <ui32>, <ui32> -> <ui32>
            %368 = ascvec.xor %dst1_46, %dst0_41, %285 : <ui32>, <ui32> -> <ui32>
            %369 = arith.addi %arg12, %c387276957_i32 : i32
            %370 = ascvec.duplicate_scalar %369, %285 : i32 -> <ui32>
            %371 = arith.addi %arg13, %c-1459197799_i32 : i32
            %372 = ascvec.duplicate_scalar %371, %285 : i32 -> <ui32>
            %373 = ascvec.xor %367, %370, %285 : <ui32>, <ui32> -> <ui32>
            %374 = ascvec.xor %368, %372, %285 : <ui32>, <ui32> -> <ui32>
            %dst0_49, %dst1_50 = ascvec.mull %373, %327, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_51, %dst1_52 = ascvec.mull %374, %328, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %375 = ascvec.xor %dst1_52, %dst0_47, %285 : <ui32>, <ui32> -> <ui32>
            %376 = ascvec.xor %dst1_50, %dst0_45, %285 : <ui32>, <ui32> -> <ui32>
            %377 = arith.addi %arg12, %c-1253254570_i32 : i32
            %378 = ascvec.duplicate_scalar %377, %285 : i32 -> <ui32>
            %379 = arith.addi %arg13, %c1684936478_i32 : i32
            %380 = ascvec.duplicate_scalar %379, %285 : i32 -> <ui32>
            %381 = ascvec.xor %375, %378, %285 : <ui32>, <ui32> -> <ui32>
            %382 = ascvec.xor %376, %380, %285 : <ui32>, <ui32> -> <ui32>
            %dst0_53, %dst1_54 = ascvec.mull %381, %327, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_55, %dst1_56 = ascvec.mull %382, %328, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %383 = ascvec.xor %dst1_56, %dst0_51, %285 : <ui32>, <ui32> -> <ui32>
            %384 = ascvec.xor %dst1_54, %dst0_49, %285 : <ui32>, <ui32> -> <ui32>
            %385 = arith.addi %arg12, %c1401181199_i32 : i32
            %386 = ascvec.duplicate_scalar %385, %285 : i32 -> <ui32>
            %387 = arith.addi %arg13, %c534103459_i32 : i32
            %388 = ascvec.duplicate_scalar %387, %285 : i32 -> <ui32>
            %389 = ascvec.xor %383, %386, %285 : <ui32>, <ui32> -> <ui32>
            %390 = ascvec.xor %384, %388, %285 : <ui32>, <ui32> -> <ui32>
            %dst0_57, %dst1_58 = ascvec.mull %389, %327, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_59, %dst1_60 = ascvec.mull %390, %328, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %391 = ascvec.xor %dst1_60, %dst0_55, %285 : <ui32>, <ui32> -> <ui32>
            %392 = ascvec.xor %dst1_58, %dst0_53, %285 : <ui32>, <ui32> -> <ui32>
            %393 = arith.addi %arg12, %c-239350328_i32 : i32
            %394 = ascvec.duplicate_scalar %393, %285 : i32 -> <ui32>
            %395 = arith.addi %arg13, %c-616729560_i32 : i32
            %396 = ascvec.duplicate_scalar %395, %285 : i32 -> <ui32>
            %397 = ascvec.xor %391, %394, %285 : <ui32>, <ui32> -> <ui32>
            %398 = ascvec.xor %392, %396, %285 : <ui32>, <ui32> -> <ui32>
            %dst0_61, %dst1_62 = ascvec.mull %397, %327, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_63, %dst1_64 = ascvec.mull %398, %328, %285 : <ui32>, <ui32> -> <ui32>, <ui32>
            %399 = ascvec.xor %dst1_64, %dst0_59, %285 : <ui32>, <ui32> -> <ui32>
            %400 = ascvec.xor %dst1_62, %dst0_57, %285 : <ui32>, <ui32> -> <ui32>
            %401 = arith.addi %arg12, %c-1879881855_i32 : i32
            %402 = ascvec.duplicate_scalar %401, %285 : i32 -> <ui32>
            %403 = arith.addi %arg13, %c-1767562579_i32 : i32
            %404 = ascvec.duplicate_scalar %403, %285 : i32 -> <ui32>
            %405 = ascvec.xor %399, %402, %285 : <ui32>, <ui32> -> <ui32>
            %406 = ascvec.xor %400, %404, %285 : <ui32>, <ui32> -> <ui32>
            %407 = ascvec.reinterpret %405 : <ui32> -> <i32>
            %408 = arith.index_cast %284 : i32 to index
            ascvec.storealign %arg14[%408], %407, %285 : memref<4096xi32, 1>, index, <i32>
            %409 = arith.addi %284, %c64_i32 : i32
            %410 = ascvec.reinterpret %dst0_63 : <ui32> -> <i32>
            %411 = arith.index_cast %409 : i32 to index
            ascvec.storealign %arg14[%411], %410, %285 : memref<4096xi32, 1>, index, <i32>
            %412 = arith.addi %284, %c128_i32 : i32
            %413 = ascvec.reinterpret %406 : <ui32> -> <i32>
            %414 = arith.index_cast %412 : i32 to index
            ascvec.storealign %arg14[%414], %413, %285 : memref<4096xi32, 1>, index, <i32>
            %415 = arith.addi %284, %c192_i32 : i32
            %416 = ascvec.reinterpret %dst0_61 : <ui32> -> <i32>
            %417 = arith.index_cast %415 : i32 to index
            ascvec.storealign %arg14[%417], %416, %285 : memref<4096xi32, 1>, index, <i32>
            %418 = arith.addi %283, %arg15 : i64
            %419 = arith.cmpi sge, %418, %arg10 : i64
            %420 = arith.extui %419 : i1 to i64
            %421 = arith.addi %279, %arg16 : i64
            %422 = arith.addi %421, %420 : i64
            %423 = arith.muli %420, %arg10 : i64
            %424 = arith.subi %418, %423 : i64
            %425 = arith.trunci %422 : i64 to i32
            %426 = arith.extui %425 : i32 to i64
            %427 = arith.subi %422, %426 : i64
            %428 = arith.remsi %427, %c4294967296_i64_20 : i64
            %429 = arith.cmpi ne, %428, %c0_i64_18 : i64
            %430 = arith.cmpi slt, %428, %c0_i64_18 : i64
            %431 = arith.andi %429, %430 : i1
            %432 = arith.divsi %427, %c4294967296_i64_20 : i64
            %433 = arith.extui %431 : i1 to i64
            %434 = arith.subi %432, %433 : i64
            %435 = arith.trunci %434 : i64 to i32
            %436 = arith.trunci %424 : i64 to i32
            %437 = arith.extui %436 : i32 to i64
            %438 = arith.subi %424, %437 : i64
            %439 = arith.remsi %438, %c4294967296_i64_20 : i64
            %440 = arith.cmpi ne, %439, %c0_i64_18 : i64
            %441 = arith.cmpi slt, %439, %c0_i64_18 : i64
            %442 = arith.andi %440, %441 : i1
            %443 = arith.divsi %438, %c4294967296_i64_20 : i64
            %444 = arith.extui %442 : i1 to i64
            %445 = arith.subi %443, %444 : i64
            %446 = arith.trunci %445 : i64 to i32
            scf.yield %425, %435, %436, %446 : i32, i32, i32, i32
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %273 = arith.trunci %195 : i64 to i32
        %274 = arith.muli %273, %c2_i32 : i32
        ascvec.get_buf v, %93, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%84, %91, %alloc_10, %99, %alloc_9, %273, %274, %104 : i64, memref<16384xbf16, 1>, memref<4096xi32, 1>, memref<16384xui8, 1>, memref<512xui32, 1>, i32, i32, memref<1024xui8, 1>) {
        ^bb0(%arg10: i64, %arg11: memref<16384xbf16, 1>, %arg12: memref<4096xi32, 1>, %arg13: memref<16384xui8, 1>, %arg14: memref<512xui32, 1>, %arg15: i32, %arg16: i32, %arg17: memref<1024xui8, 1>):
          %c2_i32_11 = arith.constant 2 : i32
          %c2139095040_i32 = arith.constant 2139095040 : i32
          %c127_i32 = arith.constant 127 : i32
          %c123_i32 = arith.constant 123 : i32
          %c947912704_i32 = arith.constant 947912704 : i32
          %cst = arith.constant 4.2949673E+9 : f32
          %c29360128_i32 = arith.constant 29360128 : i32
          %c128_i32 = arith.constant 128 : i32
          %c65535_i32 = arith.constant 65535 : i32
          %c4_i32_12 = arith.constant 4 : i32
          %c256_i32 = arith.constant 256 : i32
          %c1_i32_13 = arith.constant 1 : i32
          %c16_i32 = arith.constant 16 : i32
          %c3_i32 = arith.constant 3 : i32
          %c2139160576_i32 = arith.constant 2139160576 : i32
          %c254_i32 = arith.constant 254 : i32
          %c-15_i32 = arith.constant -15 : i32
          %c15_i32 = arith.constant 15 : i32
          %c255_i32 = arith.constant 255 : i32
          %c2147483647_i32 = arith.constant 2147483647 : i32
          %cst_14 = arith.constant 0.000000e+00 : f32
          %c0_15 = arith.constant 0 : index
          %c64_i32 = arith.constant 64 : i32
          %c32_i32 = arith.constant 32 : i32
          %c0_i32_16 = arith.constant 0 : i32
          %275 = ascvec.create_mask : !ascvec.mask
          %276 = ascvec.arange %c0_i32_16 : i32 -> <ui32>
          %277 = ascvec.cmp_lt_scalar %276, %c32_i32, %275 : <ui32>, i32 -> !ascvec.mask
          %278 = ascvec.cmp_ge_scalar %276, %c32_i32, %275 : <ui32>, i32 -> !ascvec.mask
          scf.for %arg18 = %c0_i32_16 to %arg15 step %c1_i32_13  : i32 {
            %279 = arith.muli %arg18, %c64_i32 : i32
            %280 = arith.trunci %arg10 : i64 to i32
            %281 = arith.subi %280, %279 : i32
            %mask, %remaining_out = ascvec.update_mask %281 : !ascvec.mask
            %282 = arith.index_cast %279 : i32 to index
            %283 = ascvec.loadalign_unpack %arg11[%282] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %284 = ascvec.cast %283, %275 {rounding = 6 : i32} : <bf16> -> <f32>
            %285 = ascvec.duplicate_scalar %cst_14, %275 : f32 -> <f32>
            %286 = ascvec.select %284, %285, %mask : <f32>
            %287 = ascvec.reinterpret %286 : <f32> -> <ui32>
            %288 = ascvec.duplicate_scalar %c2147483647_i32, %275 : i32 -> <ui32>
            %289 = ascvec.and %287, %288, %275 : <ui32>, <ui32> -> <ui32>
            %290 = ascvec.reduce_max %289, %277 : <ui32>
            %291 = ascvec.duplicate %290, %275 : <ui32> -> <ui32>
            %292 = ascvec.reduce_max %289, %278 : <ui32>
            %293 = ascvec.duplicate %292, %275 : <ui32> -> <ui32>
            %294 = ascvec.select %291, %293, %277 : <ui32>
            %295 = ascvec.shr %294, 23, %275 : <ui32> -> <ui32>
            %296 = ascvec.cmp_ge_scalar %295, %c255_i32, %275 : <ui32>, i32 -> !ascvec.mask
            %297 = ascvec.max_scalar %295, %c15_i32, %275 : <ui32>, i32
            %298 = ascvec.add_scalar %297, %c-15_i32, %275 : <ui32>, i32
            %299 = ascvec.duplicate_scalar %c255_i32, %296 merge %298 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %300 = ascvec.duplicate_scalar %c254_i32, %275 : i32 -> <ui32>
            %301 = ascvec.sub %300, %299, %275 : <ui32>
            %302 = ascvec.cmp_ne_scalar %299, %c0_i32_16, %275 : <ui32>, i32 -> !ascvec.mask
            %303 = ascvec.shl %301, 23, %302 : <ui32> -> <ui32>
            %304 = ascvec.duplicate_scalar %c2139160576_i32, %296 merge %303 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %305 = ascvec.reinterpret %304 : <ui32> -> <f32>
            %306 = ascvec.mul %286, %305, %275 : <f32>
            %307 = ascvec.shr %276, 2, %275 : <ui32> -> <ui32>
            %308 = ascvec.duplicate_scalar %c3_i32, %275 : i32 -> <ui32>
            %309 = ascvec.and %307, %308, %275 : <ui32>, <ui32> -> <ui32>
            %310 = ascvec.shl %309, 6, %275 : <ui32> -> <ui32>
            %311 = ascvec.shr %276, 4, %275 : <ui32> -> <ui32>
            %312 = ascvec.add %310, %311, %275 : <ui32>
            %313 = arith.remsi %arg18, %c16_i32 : i32
            %314 = arith.cmpi ne, %313, %c0_i32_16 : i32
            %315 = arith.cmpi slt, %313, %c0_i32_16 : i32
            %316 = arith.andi %314, %315 : i1
            %317 = arith.divsi %arg18, %c16_i32 : i32
            %318 = arith.extui %316 : i1 to i32
            %319 = arith.subi %317, %318 : i32
            %320 = arith.muli %319, %c256_i32 : i32
            %321 = arith.select %316, %c16_i32, %c0_i32_16 : i32
            %322 = arith.addi %313, %321 overflow<nsw> : i32
            %323 = arith.muli %322, %c4_i32_12 overflow<nsw> : i32
            %324 = arith.addi %320, %323 : i32
            %325 = ascvec.add_scalar %312, %324, %275 : <ui32>, i32
            %326 = ascvec.gather %arg12[%c0_15], %325, %275 : memref<4096xi32, 1>, <ui32>, !ascvec.mask -> <i32>
            %327 = ascvec.reinterpret %326 : <i32> -> <ui32>
            %328 = ascvec.shl %327, 8, %275 : <ui32> -> <ui32>
            %329 = ascvec.shr %327, 24, %275 : <ui32> -> <ui32>
            %330 = ascvec.or %328, %329, %275 : <ui32>, <ui32> -> <ui32>
            %331 = ascvec.and %276, %308, %275 : <ui32>, <ui32> -> <ui32>
            %332 = ascvec.duplicate_scalar %c1_i32_13, %275 : i32 -> <ui32>
            %333 = ascvec.and %331, %332, %275 : <ui32>, <ui32> -> <ui32>
            %334 = ascvec.cmp_eq_scalar %333, %c0_i32_16, %275 : <ui32>, i32 -> !ascvec.mask
            %335 = ascvec.select %327, %330, %334 : <ui32>
            %336 = ascvec.cmp_eq_scalar %331, %c0_i32_16, %275 : <ui32>, i32 -> !ascvec.mask
            %337 = ascvec.cmp_eq_scalar %331, %c3_i32, %275 : <ui32>, i32 -> !ascvec.mask
            %338 = ascvec.mask_or %336, %337, %275
            %339 = ascvec.shr %335, 16, %275 : <ui32> -> <ui32>
            %340 = ascvec.duplicate_scalar %c65535_i32, %275 : i32 -> <ui32>
            %341 = ascvec.and %335, %340, %275 : <ui32>, <ui32> -> <ui32>
            %342 = ascvec.select %339, %341, %338 : <ui32>
            %343 = ascvec.reinterpret %306 : <f32> -> <ui32>
            %344 = ascvec.shr %343, 24, %275 : <ui32> -> <ui32>
            %345 = ascvec.duplicate_scalar %c128_i32, %275 : i32 -> <ui32>
            %346 = ascvec.and %344, %345, %275 : <ui32>, <ui32> -> <ui32>
            %347 = ascvec.and %343, %288, %275 : <ui32>, <ui32> -> <ui32>
            %348 = ascvec.shr %347, 5, %275 : <ui32> -> <ui32>
            %349 = ascvec.duplicate_scalar %c29360128_i32, %275 : i32 -> <ui32>
            %350 = ascvec.sub %348, %349, %275 : <ui32>
            %351 = ascvec.reinterpret %347 : <ui32> -> <f32>
            %352 = ascvec.mul_scalar %351, %cst, %275 : <f32>, f32
            %353 = ascvec.cast %352, %275 {rounding = 4 : i32} : <f32> -> <i32>
            %354 = ascvec.reinterpret %353 : <i32> -> <ui32>
            %355 = ascvec.cmp_ge_scalar %347, %c947912704_i32, %275 : <ui32>, i32 -> !ascvec.mask
            %356 = ascvec.select %350, %354, %355 : <ui32>
            %357 = ascvec.and %342, %340, %275 : <ui32>, <ui32> -> <ui32>
            %358 = ascvec.add %356, %357, %275 : <ui32>
            %359 = ascvec.shr %358, 16, %275 : <ui32> -> <ui32>
            %360 = ascvec.min_scalar %359, %c123_i32, %275 : <ui32>, i32
            %361 = ascvec.or %360, %346, %275 : <ui32>, <ui32> -> <ui32>
            %362 = ascvec.duplicate_scalar %c127_i32, %275 : i32 -> <ui32>
            %363 = ascvec.cmp_gt_scalar %347, %c2139095040_i32, %275 : <ui32>, i32 -> !ascvec.mask
            %364 = ascvec.select %362, %361, %363 : <ui32>
            ascvec.storealign_pack %arg13[%282], %364, %mask {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
            %365 = arith.muli %arg18, %c2_i32_11 : i32
            %366 = arith.index_cast %365 : i32 to index
            ascvec.storealign_1st %arg14[%366], %299 : memref<512xui32, 1>, index, <ui32>
            %367 = ascvec.duplicate_scalar %c32_i32, %275 : i32 -> <ui32>
            %368 = ascvec.gather_reg %299, %367 : <ui32>, <ui32> -> <ui32>
            %369 = arith.addi %365, %c1_i32_13 : i32
            %370 = arith.index_cast %369 : i32 to index
            ascvec.storealign_1st %arg14[%370], %368 : memref<512xui32, 1>, index, <ui32>
          }
          ascvec.mem_bar vst_vld
          scf.for %arg18 = %c0_i32_16 to %arg16 step %c64_i32  : i32 {
            %279 = arith.subi %arg16, %arg18 : i32
            %mask, %remaining_out = ascvec.update_mask %279 : !ascvec.mask
            %280 = ascvec.add_scalar %276, %arg18, %275 : <ui32>, i32
            %281 = ascvec.gather %arg14[%c0_15], %280, %mask : memref<512xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
            %282 = arith.index_cast %arg18 : i32 to index
            ascvec.storealign_pack %arg17[%282], %281, %mask {mode = "b32_to_b8"} : memref<1024xui8, 1>, index, <ui32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %93, 0 {cannir.buffer.generated_sync}
      }
      %112 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %113 = cannir.extract_shape %112 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %114 = cannir.extract_dim %113[0] : !cannir.shape<(-1, -1)>
      %115 = arith.minsi %69, %114 : i64
      %116 = arith.subi %114, %115 : i64
      %117 = arith.minsi %116, %c1_i64 : i64
      %118 = arith.cmpi sle, %69, %114 : i64
      %119 = arith.select %118, %117, %c0_i64 : i64
      %120 = arith.cmpi sgt, %119, %c1_i64 : i64
      %121 = arith.select %120, %c1_i64, %119 : i64
      %122 = cannir.extract_dim %113[1] : !cannir.shape<(-1, -1)>
      %123 = arith.maxsi %46, %c1_i64 : i64
      %124 = arith.divsi %122, %123 : i64
      %125 = arith.minsi %80, %124 : i64
      %126 = arith.muli %125, %46 : i64
      %127 = arith.subi %122, %126 : i64
      %128 = arith.minsi %46, %127 : i64
      %129 = arith.cmpi sle, %80, %124 : i64
      %130 = arith.select %129, %128, %c0_i64 : i64
      %131 = arith.cmpi sgt, %130, %c16384_i64 : i64
      %132 = arith.select %131, %c16384_i64, %130 : i64
      %133 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %134 = cannir.extract_stride %133 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %135 = cannir.extract_dim %134[0] : !cannir.stride<(-1, 1)>
      %136 = arith.muli %69, %135 : i64
      %137 = arith.addi %136, %81 : i64
      %138 = arith.muli %137, %c8_i64 : i64
      %139 = ascscalar.memref_address %alloc_1 : memref<1xui8>
      %140 = arith.divui %138, %c8_i64 : i64
      %141 = arith.addi %139, %140 : i64
      %142 = ascscalar.bind_gm_ptr %141 : memref<1xui8>
      %143 = ascscalar.bind_local_ptr %98 : memref<1xui8, 1>
      %144 = cannir.extract_stride %112 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %145 = cannir.extract_dim %144[0] : !cannir.stride<(-1, 1)>
      %146 = arith.cmpi ugt, %121, %c0_i64 : i64
      %147 = arith.cmpi ugt, %132, %c0_i64 : i64
      %148 = arith.andi %146, %147 : i1
      scf.if %148 {
        ascvec.get_buf mte3, %100, 0 {cannir.buffer.generated_sync}
        %188 = arith.trunci %121 : i64 to i32
        %189 = arith.trunci %132 : i64 to i32
        ascvec.copy_ub2gm %142[%c0], %143[%c0], %188, %189, %145, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %100, 0 {cannir.buffer.generated_sync}
      }
      %149 = arith.muli %45, %c2_i64 : i64
      %150 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %151 = cannir.extract_shape %150 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %152 = cannir.extract_dim %151[0] : !cannir.shape<(-1, -1)>
      %153 = arith.minsi %69, %152 : i64
      %154 = arith.subi %152, %153 : i64
      %155 = arith.minsi %154, %c1_i64 : i64
      %156 = arith.cmpi sle, %69, %152 : i64
      %157 = arith.select %156, %155, %c0_i64 : i64
      %158 = arith.cmpi sgt, %157, %c1_i64 : i64
      %159 = arith.select %158, %c1_i64, %157 : i64
      %160 = cannir.extract_dim %151[1] : !cannir.shape<(-1, -1)>
      %161 = arith.maxsi %149, %c1_i64 : i64
      %162 = arith.divsi %160, %161 : i64
      %163 = arith.minsi %80, %162 : i64
      %164 = arith.muli %163, %149 : i64
      %165 = arith.subi %160, %164 : i64
      %166 = arith.minsi %149, %165 : i64
      %167 = arith.cmpi sle, %80, %162 : i64
      %168 = arith.select %167, %166, %c0_i64 : i64
      %169 = arith.cmpi sgt, %168, %c1024_i64 : i64
      %170 = arith.select %169, %c1024_i64, %168 : i64
      %171 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %172 = cannir.extract_stride %171 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %173 = cannir.extract_dim %172[0] : !cannir.stride<(-1, 1)>
      %174 = arith.muli %69, %173 : i64
      %175 = arith.muli %80, %149 : i64
      %176 = arith.addi %174, %175 : i64
      %177 = arith.muli %176, %c8_i64 : i64
      %178 = ascscalar.memref_address %alloc_2 : memref<1xui8>
      %179 = arith.divui %177, %c8_i64 : i64
      %180 = arith.addi %178, %179 : i64
      %181 = ascscalar.bind_gm_ptr %180 : memref<1xui8>
      %182 = ascscalar.bind_local_ptr %103 : memref<1xui8, 1>
      %183 = cannir.extract_stride %150 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %184 = cannir.extract_dim %183[0] : !cannir.stride<(-1, 1)>
      %185 = arith.cmpi ugt, %159, %c0_i64 : i64
      %186 = arith.cmpi ugt, %170, %c0_i64 : i64
      %187 = arith.andi %185, %186 : i1
      scf.if %187 {
        ascvec.get_buf mte3, %105, 0 {cannir.buffer.generated_sync}
        %188 = arith.trunci %159 : i64 to i32
        %189 = arith.trunci %170 : i64 to i32
        ascvec.copy_ub2gm %181[%c0], %182[%c0], %188, %189, %184, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %105, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %86, %95 : i64, i64
    }
    return
  }
}
