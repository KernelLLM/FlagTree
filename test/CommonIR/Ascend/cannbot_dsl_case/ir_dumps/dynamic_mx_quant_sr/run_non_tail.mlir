module {
  func.func @run_non_tail_88598342f340321269a1508c09508e8b(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: i64, %arg5: i64, %arg6: !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>) attributes {cannbotdsl.contract_digest = "88598342f340321269a1508c09508e8b4fe6ec5dc7256b24d3b61f1ba00c0980", cannbotdsl.public_slot_count = 7 : i64, cannbotdsl.slot_dtype_bits = [16, 16, 8, 8, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 4, 1, 1, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 2, 2, 1], cannbotdsl.slot_type_digests = ["d3381cb3edb8f40514b57c6c942d36e84a87723300273978fe7c3f381dbd4f9d", "d3381cb3edb8f40514b57c6c942d36e84a87723300273978fe7c3f381dbd4f9d", "e3a7bc6c4b8cb98d68cf0f892959846f4e8f11c27f2c9e430aec968ae281159e", "9332f9464639b7e579b015309e81477f71c4c6705898c0b7ab38269d65f6d71a", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "4126671a83041647cd98afed881383228865da287ea44573939f32b1d956765b"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel_run_non_tail_0(%arg5) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg6] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 5>, name = "_DynamicMxQuantSrKernel_run_non_tail", op_type = "_DynamicMxQuantSrKernel_run_non_tail", roles = array<i64: 0, 0, 1, 1, 0>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel_run_non_tail_0(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: i64, %arg5: !cannir.tensor<<gm, i64>, <<(2)>, <(1)>>, <<(2)>, <(1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel_run_non_tail_0", cannir.resource.ids_prepared} {
    %c1024_i32 = arith.constant 1024 : i32
    %c1 = arith.constant 1 : index
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
    %alloc_3 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<2xi64>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xbf16, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xbf16, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<16384xui8, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 81920 : i64} : memref<16384xui8, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 98304 : i64} : memref<1024xui8, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 99328 : i64} : memref<1024xui8, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 100352 : i64} : memref<512xui32, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 102400 : i64} : memref<4096xi32, 1>
    %0 = ascscalar.scalar_ptr %alloc_3[%c0] : memref<2xi64>, index -> <gm, i64>
    %1 = ascscalar.load_dev %0 : !ascscalar.scalar_ptr<gm, i64> -> i64
    %2 = ascscalar.scalar_ptr %alloc_3[%c1] : memref<2xi64>, index -> <gm, i64>
    %3 = ascscalar.load_dev %2 : !ascscalar.scalar_ptr<gm, i64> -> i64
    %4 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
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
    %15 = arith.muli %14, %c2_i64 : i64
    %16 = arith.muli %15, %13 : i64
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
    %29 = arith.addi %28, %c255_i64 : i64
    %30 = arith.remsi %29, %c256_i64 : i64
    %31 = arith.cmpi ne, %30, %c0_i64 : i64
    %32 = arith.cmpi slt, %30, %c0_i64 : i64
    %33 = arith.andi %31, %32 : i1
    %34 = arith.divsi %29, %c256_i64 : i64
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
    %48 = arith.cmpi sge, %47, %c4_i64 : i64
    %49 = scf.if %48 -> (i64) {
      %65 = arith.addi %47, %c3_i64 : i64
      %66 = arith.remsi %65, %c4_i64 : i64
      %67 = arith.cmpi ne, %66, %c0_i64 : i64
      %68 = arith.cmpi slt, %66, %c0_i64 : i64
      %69 = arith.andi %67, %68 : i1
      %70 = arith.divsi %65, %c4_i64 : i64
      %71 = arith.extui %69 : i1 to i64
      %72 = arith.subi %70, %71 : i64
      %73 = arith.muli %72, %c4_i64 : i64
      scf.yield %73 : i64
    } else {
      scf.yield %47 : i64
    }
    %50 = arith.muli %49, %c64_i64 : i64
    %51 = arith.addi %13, %49 : i64
    %52 = arith.subi %51, %c1_i64 : i64
    %53 = arith.remsi %52, %49 : i64
    %54 = arith.cmpi ne, %53, %c0_i64 : i64
    %55 = arith.cmpi slt, %53, %c0_i64 : i64
    %56 = arith.cmpi slt, %49, %c0_i64 : i64
    %57 = arith.xori %55, %56 : i1
    %58 = arith.andi %54, %57 : i1
    %59 = arith.divsi %52, %49 : i64
    %60 = arith.extui %58 : i1 to i64
    %61 = arith.subi %59, %60 : i64
    %62 = arith.muli %15, %61 : i64
    %63 = ascscalar.get_block_idx
    %64:2 = scf.for %arg6 = %63 to %62 step %17 iter_args(%arg7 = %c0_i64, %arg8 = %c0_i64) -> (i64, i64)  : i64 {
      %65 = arith.remsi %arg6, %61 : i64
      %66 = arith.cmpi ne, %65, %c0_i64 : i64
      %67 = arith.cmpi slt, %65, %c0_i64 : i64
      %68 = arith.cmpi slt, %61, %c0_i64 : i64
      %69 = arith.xori %67, %68 : i1
      %70 = arith.andi %66, %69 : i1
      %71 = arith.divsi %arg6, %61 : i64
      %72 = arith.extui %70 : i1 to i64
      %73 = arith.subi %71, %72 : i64
      %74 = arith.remsi %73, %c2_i64 : i64
      %75 = arith.cmpi ne, %74, %c0_i64 : i64
      %76 = arith.cmpi slt, %74, %c0_i64 : i64
      %77 = arith.andi %75, %76 : i1
      %78 = arith.divsi %73, %c2_i64 : i64
      %79 = arith.extui %77 : i1 to i64
      %80 = arith.subi %78, %79 : i64
      %81 = arith.select %77, %c2_i64, %c0_i64 : i64
      %82 = arith.addi %74, %81 overflow<nsw> : i64
      %83 = arith.select %70, %61, %c0_i64 : i64
      %84 = arith.addi %65, %83 : i64
      %85 = arith.muli %84, %50 : i64
      %86 = arith.subi %6, %85 : i64
      %87 = arith.cmpi sgt, %86, %50 : i64
      %88 = arith.select %87, %50, %86 : i64
      %89 = arith.cmpi eq, %82, %c0_i64 : i64
      %90 = scf.if %89 -> (i64) {
        %192 = arith.cmpi eq, %arg7, %c0_i64 : i64
        %193 = ascscalar.memref_address %alloc_4 : memref<16384xbf16, 1>
        %194 = ascscalar.memref_address %alloc_5 : memref<16384xbf16, 1>
        %195 = arith.select %192, %193, %194 : i64
        %196 = arith.cmpi ne, %arg7, %c0_i64 : i64
        %197 = arith.extui %196 : i1 to i64
        %198 = arith.addi %arg7, %c1_i64 overflow<nsw> : i64
        %199 = arith.remui %198, %c2_i64 : i64
        %200 = arith.minsi %80, %14 : i64
        %201 = arith.subi %14, %200 : i64
        %202 = arith.minsi %201, %c1_i64 : i64
        %203 = arith.cmpi sle, %80, %14 : i64
        %204 = arith.select %203, %202, %c0_i64 : i64
        %205 = arith.maxsi %50, %c1_i64 : i64
        %206 = arith.divsi %6, %205 : i64
        %207 = arith.minsi %84, %206 : i64
        %208 = arith.muli %207, %50 : i64
        %209 = arith.subi %6, %208 : i64
        %210 = arith.minsi %50, %209 : i64
        %211 = arith.cmpi sle, %84, %206 : i64
        %212 = arith.select %211, %210, %c0_i64 : i64
        %213 = cannir.extract_stride %4 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
        %214 = cannir.extract_dim %213[0] : !cannir.stride<(-1, 2)>
        %215 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
        %216 = cannir.extract_stride %215 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
        %217 = cannir.extract_dim %216[0] : !cannir.stride<(-1, 2)>
        %218 = arith.muli %80, %217 : i64
        %219 = arith.muli %85, %c2_i64 : i64
        %220 = arith.addi %218, %219 : i64
        %221 = arith.muli %220, %c16_i64 : i64
        %222 = ascscalar.memref_address %alloc : memref<1xbf16>
        %223 = arith.divui %221, %c8_i64 : i64
        %224 = arith.addi %222, %223 : i64
        %225 = ascscalar.bind_gm_ptr %224 : memref<1xbf16>
        %226 = ascscalar.bind_local_ptr %195 : memref<1xbf16, 1>
        %227 = arith.cmpi ugt, %204, %c0_i64 : i64
        %228 = arith.cmpi ugt, %212, %c0_i64 : i64
        %229 = arith.andi %227, %228 : i1
        scf.if %229 {
          ascvec.get_buf mte2, %197, 0 {cannir.buffer.generated_sync}
          %230 = arith.trunci %212 : i64 to i32
          %231 = arith.trunci %204 : i64 to i32
          ascvec.copy_gm2ub_nddma %226[%c0], %225[%c0], %230, %231, %c1_i32, %c1_i32, %c1_i32, %c2_i64, %214, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c16384_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 : memref<1xbf16, 1>, memref<1xbf16>
          ascvec.release_buf mte2, %197, 0 {cannir.buffer.generated_sync}
        }
        scf.yield %199 : i64
      } else {
        %192 = arith.cmpi eq, %arg7, %c0_i64 : i64
        %193 = ascscalar.memref_address %alloc_4 : memref<16384xbf16, 1>
        %194 = ascscalar.memref_address %alloc_5 : memref<16384xbf16, 1>
        %195 = arith.select %192, %193, %194 : i64
        %196 = arith.cmpi ne, %arg7, %c0_i64 : i64
        %197 = arith.extui %196 : i1 to i64
        %198 = arith.addi %arg7, %c1_i64 overflow<nsw> : i64
        %199 = arith.remui %198, %c2_i64 : i64
        %200 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
        %201 = cannir.extract_shape %200 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, -1)>
        %202 = cannir.extract_dim %201[0] : !cannir.shape<(-1, -1)>
        %203 = arith.minsi %80, %202 : i64
        %204 = arith.subi %202, %203 : i64
        %205 = arith.minsi %204, %c1_i64 : i64
        %206 = arith.cmpi sle, %80, %202 : i64
        %207 = arith.select %206, %205, %c0_i64 : i64
        %208 = cannir.extract_dim %201[1] : !cannir.shape<(-1, -1)>
        %209 = arith.maxsi %50, %c1_i64 : i64
        %210 = arith.divsi %208, %209 : i64
        %211 = arith.minsi %84, %210 : i64
        %212 = arith.muli %211, %50 : i64
        %213 = arith.subi %208, %212 : i64
        %214 = arith.minsi %50, %213 : i64
        %215 = arith.cmpi sle, %84, %210 : i64
        %216 = arith.select %215, %214, %c0_i64 : i64
        %217 = cannir.extract_stride %200 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
        %218 = cannir.extract_dim %217[0] : !cannir.stride<(-1, 2)>
        %219 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
        %220 = cannir.extract_stride %219 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
        %221 = cannir.extract_dim %220[0] : !cannir.stride<(-1, 2)>
        %222 = arith.muli %80, %221 : i64
        %223 = arith.muli %85, %c2_i64 : i64
        %224 = arith.addi %222, %223 : i64
        %225 = arith.muli %224, %c16_i64 : i64
        %226 = ascscalar.memref_address %alloc_0 : memref<1xbf16>
        %227 = arith.divui %225, %c8_i64 : i64
        %228 = arith.addi %226, %227 : i64
        %229 = ascscalar.bind_gm_ptr %228 : memref<1xbf16>
        %230 = ascscalar.bind_local_ptr %195 : memref<1xbf16, 1>
        %231 = arith.cmpi ugt, %207, %c0_i64 : i64
        %232 = arith.cmpi ugt, %216, %c0_i64 : i64
        %233 = arith.andi %231, %232 : i1
        scf.if %233 {
          ascvec.get_buf mte2, %197, 0 {cannir.buffer.generated_sync}
          %234 = arith.trunci %216 : i64 to i32
          %235 = arith.trunci %207 : i64 to i32
          ascvec.copy_gm2ub_nddma %230[%c0], %229[%c0], %234, %235, %c1_i32, %c1_i32, %c1_i32, %c2_i64, %218, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c16384_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 : memref<1xbf16, 1>, memref<1xbf16>
          ascvec.release_buf mte2, %197, 0 {cannir.buffer.generated_sync}
        }
        scf.yield %199 : i64
      }
      %91 = arith.cmpi eq, %arg8, %c0_i64 : i64
      %92 = ascscalar.memref_address %alloc_4 : memref<16384xbf16, 1>
      %93 = ascscalar.memref_address %alloc_5 : memref<16384xbf16, 1>
      %94 = arith.select %91, %92, %93 : i64
      %95 = ascscalar.bind_local_ptr %94 : memref<16384xbf16, 1>
      %96 = arith.cmpi ne, %arg8, %c0_i64 : i64
      %97 = arith.extui %96 : i1 to i64
      %98 = arith.addi %arg8, %c1_i64 overflow<nsw> : i64
      %99 = arith.remui %98, %c2_i64 : i64
      %100 = ascscalar.memref_address %alloc_6 : memref<16384xui8, 1>
      %101 = ascscalar.memref_address %alloc_7 : memref<16384xui8, 1>
      %102 = arith.select %91, %100, %101 : i64
      %103 = ascscalar.bind_local_ptr %102 : memref<16384xui8, 1>
      %104 = arith.select %91, %c2_i64, %c3_i64 : i64
      %105 = ascscalar.memref_address %alloc_8 : memref<1024xui8, 1>
      %106 = ascscalar.memref_address %alloc_9 : memref<1024xui8, 1>
      %107 = arith.select %91, %105, %106 : i64
      %108 = ascscalar.bind_local_ptr %107 : memref<1024xui8, 1>
      %109 = arith.select %91, %c4_i64, %c5_i64 : i64
      %110 = arith.muli %73, %6 : i64
      %111 = arith.addi %110, %85 : i64
      %112 = arith.cmpi sge, %88, %c256_i64 : i64
      %113 = arith.remsi %88, %c256_i64 : i64
      %114 = arith.cmpi eq, %113, %c0_i64 : i64
      %115 = arith.select %112, %114, %false : i1
      scf.if %115 {
        %192 = arith.divsi %88, %c256_i64 : i64
        %193 = arith.addi %88, %c1023_i64 : i64
        %194 = arith.remsi %193, %c1024_i64 : i64
        %195 = arith.cmpi ne, %194, %c0_i64 : i64
        %196 = arith.cmpi slt, %194, %c0_i64 : i64
        %197 = arith.andi %195, %196 : i1
        %198 = arith.divsi %193, %c1024_i64 : i64
        %199 = arith.extui %197 : i1 to i64
        %200 = arith.subi %198, %199 : i64
        %201 = arith.trunci %1 : i64 to i32
        %202 = arith.extui %201 : i32 to i64
        %203 = arith.subi %1, %202 : i64
        %204 = arith.remsi %203, %c4294967296_i64 : i64
        %205 = arith.cmpi ne, %204, %c0_i64 : i64
        %206 = arith.cmpi slt, %204, %c0_i64 : i64
        %207 = arith.andi %205, %206 : i1
        %208 = arith.divsi %203, %c4294967296_i64 : i64
        %209 = arith.extui %207 : i1 to i64
        %210 = arith.subi %208, %209 : i64
        %211 = arith.trunci %210 : i64 to i32
        %212 = arith.divui %111, %c16_i64 : i64
        %213 = arith.trunci %212 : i64 to i32
        %214 = arith.extui %213 : i32 to i64
        %215 = arith.subi %212, %214 : i64
        %216 = arith.remsi %215, %c4294967296_i64 : i64
        %217 = arith.cmpi ne, %216, %c0_i64 : i64
        %218 = arith.cmpi slt, %216, %c0_i64 : i64
        %219 = arith.andi %217, %218 : i1
        %220 = arith.divsi %215, %c4294967296_i64 : i64
        %221 = arith.extui %219 : i1 to i64
        %222 = arith.subi %220, %221 : i64
        %223 = arith.trunci %222 : i64 to i32
        %224 = arith.trunci %3 : i64 to i32
        %225 = arith.extui %224 : i32 to i64
        %226 = arith.subi %3, %225 : i64
        %227 = arith.remsi %226, %c4294967296_i64 : i64
        %228 = arith.cmpi ne, %227, %c0_i64 : i64
        %229 = arith.cmpi slt, %227, %c0_i64 : i64
        %230 = arith.andi %228, %229 : i1
        %231 = arith.divsi %226, %c4294967296_i64 : i64
        %232 = arith.extui %230 : i1 to i64
        %233 = arith.subi %231, %232 : i64
        %234 = arith.trunci %233 : i64 to i32
        %235 = arith.divui %224, %c4_i32 : i32
        %236 = arith.remui %234, %c4_i32 : i32
        %237 = arith.muli %236, %c1073741824_i32 : i32
        %238 = arith.addi %235, %237 : i32
        %239 = arith.divui %234, %c4_i32 : i32
        %240 = arith.muli %200, %c64_i64 : i64
        %241 = arith.divui %212, %c16384_i64 : i64
        %242 = arith.addi %212, %240 : i64
        %243 = arith.subi %242, %c1_i64 : i64
        %244 = arith.divui %243, %c16384_i64 : i64
        %245 = arith.cmpi eq, %241, %244 : i64
        scf.if %245 {
          %248 = arith.trunci %241 : i64 to i32
          %249 = arith.extui %248 : i32 to i64
          %250 = arith.subi %241, %249 : i64
          %251 = arith.remsi %250, %c4294967296_i64 : i64
          %252 = arith.cmpi ne, %251, %c0_i64 : i64
          %253 = arith.cmpi slt, %251, %c0_i64 : i64
          %254 = arith.andi %252, %253 : i1
          %255 = arith.divsi %250, %c4294967296_i64 : i64
          %256 = arith.extui %254 : i1 to i64
          %257 = arith.subi %255, %256 : i64
          %258 = arith.trunci %257 : i64 to i32
          %259 = arith.addi %238, %248 : i32
          %260 = arith.addi %239, %258 : i32
          %261 = arith.cmpi ult, %259, %238 : i32
          %262 = arith.extui %261 : i1 to i32
          %263 = arith.addi %260, %262 : i32
          %264 = arith.extui %259 : i32 to i64
          %265 = arith.muli %264, %c3449720151_i64 : i64
          %266 = arith.trunci %265 : i64 to i32
          %267 = arith.extui %266 : i32 to i64
          %268 = arith.subi %265, %267 : i64
          %269 = arith.remsi %268, %c4294967296_i64 : i64
          %270 = arith.cmpi ne, %269, %c0_i64 : i64
          %271 = arith.cmpi slt, %269, %c0_i64 : i64
          %272 = arith.andi %270, %271 : i1
          %273 = arith.divsi %268, %c4294967296_i64 : i64
          %274 = arith.extui %272 : i1 to i64
          %275 = arith.subi %273, %274 : i64
          %276 = arith.trunci %275 : i64 to i32
          %277 = arith.xori %263, %211 : i32
          %278 = arith.extui %277 : i32 to i64
          %279 = arith.muli %278, %c3449720151_i64 : i64
          %280 = arith.trunci %279 : i64 to i32
          %281 = arith.extui %280 : i32 to i64
          %282 = arith.subi %279, %281 : i64
          %283 = arith.remsi %282, %c4294967296_i64 : i64
          %284 = arith.cmpi ne, %283, %c0_i64 : i64
          %285 = arith.cmpi slt, %283, %c0_i64 : i64
          %286 = arith.andi %284, %285 : i1
          %287 = arith.divsi %282, %c4294967296_i64 : i64
          %288 = arith.extui %286 : i1 to i64
          %289 = arith.subi %287, %288 : i64
          %290 = arith.trunci %289 : i64 to i32
          %291 = arith.addi %201, %c-1640531527_i32 : i32
          %292 = arith.addi %211, %c-1150833019_i32 : i32
          %293 = arith.xori %290, %266 : i32
          %294 = arith.xori %293, %291 : i32
          %295 = arith.extui %294 : i32 to i64
          %296 = arith.muli %295, %c3528531795_i64 : i64
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
          %308 = arith.xori %276, %201 : i32
          %309 = arith.addi %201, %c1013904242_i32 : i32
          %310 = arith.xori %280, %309 : i32
          %311 = arith.addi %211, %c1993301258_i32 : i32
          %312 = arith.xori %307, %311 : i32
          ascsimt.vf(%213, %308, %292, %310, %312, %201, %211, %297, %alloc_11, %240 : i32, i32, i32, i32, i32, i32, i32, i32, memref<4096xi32, 1>, i64) {
          ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: memref<4096xi32, 1>, %arg18: i64):
            %c256_i64_12 = arith.constant 256 : i64
            %c3_i64_13 = arith.constant 3 : i64
            %c2_i64_14 = arith.constant 2 : i64
            %c1_i64_15 = arith.constant 1 : i64
            %c4_i64_16 = arith.constant 4 : i64
            %c1_i32_17 = arith.constant 1 : i32
            %c10_i32 = arith.constant 10 : i32
            %c3_i32 = arith.constant 3 : i32
            %c-1150833019_i32_18 = arith.constant -1150833019 : i32
            %c-1640531527_i32_19 = arith.constant -1640531527 : i32
            %c-845247145_i32 = arith.constant -845247145 : i32
            %c-766435501_i32 = arith.constant -766435501 : i32
            %c16384_i32_20 = arith.constant 16384 : i32
            %313 = ascsimt.thread_idx {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
            %314 = arith.extsi %313 {cannir.vf_threads = 256 : i64} : i32 to i64
            scf.for %arg19 = %314 to %arg18 step %c256_i64_12  : i64 {
              %315 = arith.trunci %arg19 : i64 to i32
              %316 = arith.addi %arg9, %315 : i32
              %317 = arith.remui %316, %c16384_i32_20 : i32
              %318 = arith.xori %317, %arg10 : i32
              %319 = arith.muli %318, %c-766435501_i32 : i32
              %320 = "ascsimt.bitcast"(%318) : (i32) -> ui32
              %321 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %322 = ascsimt.math_binary %320, %321, "mulhi" : ui32, ui32 -> ui32
              %323 = "ascsimt.bitcast"(%322) : (ui32) -> i32
              %324 = arith.xori %323, %arg11 : i32
              %325 = arith.muli %324, %c-845247145_i32 : i32
              %326 = "ascsimt.bitcast"(%324) : (i32) -> ui32
              %327 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %328 = ascsimt.math_binary %326, %327, "mulhi" : ui32, ui32 -> ui32
              %329 = "ascsimt.bitcast"(%328) : (ui32) -> i32
              %330 = arith.xori %329, %arg12 : i32
              %331 = arith.xori %319, %arg13 : i32
              %332:4 = scf.for %arg20 = %c3_i32 to %c10_i32 step %c1_i32_17 iter_args(%arg21 = %330, %arg22 = %325, %arg23 = %331, %arg24 = %arg16) -> (i32, i32, i32, i32)  : i32 {
                %341 = arith.muli %arg21, %c-766435501_i32 : i32
                %342 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                %343 = ascsimt.math_binary %342, %321, "mulhi" : ui32, ui32 -> ui32
                %344 = "ascsimt.bitcast"(%343) : (ui32) -> i32
                %345 = arith.muli %arg23, %c-845247145_i32 : i32
                %346 = "ascsimt.bitcast"(%arg23) : (i32) -> ui32
                %347 = ascsimt.math_binary %346, %327, "mulhi" : ui32, ui32 -> ui32
                %348 = "ascsimt.bitcast"(%347) : (ui32) -> i32
                %349 = arith.muli %arg20, %c-1640531527_i32_19 : i32
                %350 = arith.addi %arg14, %349 : i32
                %351 = arith.muli %arg20, %c-1150833019_i32_18 : i32
                %352 = arith.addi %arg15, %351 : i32
                %353 = arith.xori %348, %arg22 : i32
                %354 = arith.xori %353, %350 : i32
                %355 = arith.xori %344, %arg24 : i32
                %356 = arith.xori %355, %352 : i32
                scf.yield %354, %345, %356, %341 : i32, i32, i32, i32
              }
              %333 = arith.muli %arg19, %c4_i64_16 : i64
              %334 = arith.index_cast %333 : i64 to index
              ascsimt.store %arg17[%334], %332#0 : memref<4096xi32, 1>, i32
              %335 = arith.addi %333, %c1_i64_15 : i64
              %336 = arith.index_cast %335 : i64 to index
              ascsimt.store %arg17[%336], %332#1 : memref<4096xi32, 1>, i32
              %337 = arith.addi %333, %c2_i64_14 : i64
              %338 = arith.index_cast %337 : i64 to index
              ascsimt.store %arg17[%338], %332#2 : memref<4096xi32, 1>, i32
              %339 = arith.addi %333, %c3_i64_13 : i64
              %340 = arith.index_cast %339 : i64 to index
              ascsimt.store %arg17[%340], %332#3 : memref<4096xi32, 1>, i32
            } {cannir.vf_threads = 256 : i64}
          } {threads = 256 : i64}
        } else {
          ascsimt.vf(%213, %223, %238, %239, %201, %211, %alloc_11, %240 : i32, i32, i32, i32, i32, i32, memref<4096xi32, 1>, i64) {
          ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<4096xi32, 1>, %arg16: i64):
            %c256_i64_12 = arith.constant 256 : i64
            %c3_i64_13 = arith.constant 3 : i64
            %c2_i64_14 = arith.constant 2 : i64
            %c1_i64_15 = arith.constant 1 : i64
            %c4_i64_16 = arith.constant 4 : i64
            %c1_i32_17 = arith.constant 1 : i32
            %c10_i32 = arith.constant 10 : i32
            %c0_i32_18 = arith.constant 0 : i32
            %c-1150833019_i32_19 = arith.constant -1150833019 : i32
            %c-1640531527_i32_20 = arith.constant -1640531527 : i32
            %c-845247145_i32 = arith.constant -845247145 : i32
            %c-766435501_i32 = arith.constant -766435501 : i32
            %c262144_i32 = arith.constant 262144 : i32
            %c16384_i32_21 = arith.constant 16384 : i32
            %248 = ascsimt.thread_idx {ascsimt.scope = 2 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
            %249 = arith.extsi %248 {cannir.vf_threads = 256 : i64} : i32 to i64
            scf.for %arg17 = %249 to %arg16 step %c256_i64_12  : i64 {
              %250 = arith.trunci %arg17 : i64 to i32
              %251 = arith.addi %arg9, %250 : i32
              %252 = arith.cmpi ult, %251, %arg9 : i32
              %253 = arith.extui %252 : i1 to i32
              %254 = arith.addi %arg10, %253 : i32
              %255 = arith.divui %251, %c16384_i32_21 : i32
              %256 = arith.remui %254, %c16384_i32_21 : i32
              %257 = arith.muli %256, %c262144_i32 : i32
              %258 = arith.addi %255, %257 : i32
              %259 = arith.divui %254, %c16384_i32_21 : i32
              %260 = arith.addi %arg11, %258 : i32
              %261 = arith.addi %arg12, %259 : i32
              %262 = arith.cmpi ult, %260, %arg11 : i32
              %263 = arith.extui %262 : i1 to i32
              %264 = arith.addi %261, %263 : i32
              %265 = arith.remui %251, %c16384_i32_21 : i32
              %266:4 = scf.for %arg18 = %c0_i32_18 to %c10_i32 step %c1_i32_17 iter_args(%arg19 = %c0_i32_18, %arg20 = %265, %arg21 = %260, %arg22 = %264) -> (i32, i32, i32, i32)  : i32 {
                %275 = arith.muli %arg19, %c-766435501_i32 : i32
                %276 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                %277 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                %278 = ascsimt.math_binary %276, %277, "mulhi" : ui32, ui32 -> ui32
                %279 = "ascsimt.bitcast"(%278) : (ui32) -> i32
                %280 = arith.muli %arg21, %c-845247145_i32 : i32
                %281 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                %282 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                %283 = ascsimt.math_binary %281, %282, "mulhi" : ui32, ui32 -> ui32
                %284 = "ascsimt.bitcast"(%283) : (ui32) -> i32
                %285 = arith.muli %arg18, %c-1640531527_i32_20 : i32
                %286 = arith.addi %arg13, %285 : i32
                %287 = arith.muli %arg18, %c-1150833019_i32_19 : i32
                %288 = arith.addi %arg14, %287 : i32
                %289 = arith.xori %284, %arg20 : i32
                %290 = arith.xori %289, %286 : i32
                %291 = arith.xori %279, %arg22 : i32
                %292 = arith.xori %291, %288 : i32
                scf.yield %290, %280, %292, %275 : i32, i32, i32, i32
              }
              %267 = arith.muli %arg17, %c4_i64_16 : i64
              %268 = arith.index_cast %267 : i64 to index
              ascsimt.store %arg15[%268], %266#0 : memref<4096xi32, 1>, i32
              %269 = arith.addi %267, %c1_i64_15 : i64
              %270 = arith.index_cast %269 : i64 to index
              ascsimt.store %arg15[%270], %266#1 : memref<4096xi32, 1>, i32
              %271 = arith.addi %267, %c2_i64_14 : i64
              %272 = arith.index_cast %271 : i64 to index
              ascsimt.store %arg15[%272], %266#2 : memref<4096xi32, 1>, i32
              %273 = arith.addi %267, %c3_i64_13 : i64
              %274 = arith.index_cast %273 : i64 to index
              ascsimt.store %arg15[%274], %266#3 : memref<4096xi32, 1>, i32
            } {cannir.vf_threads = 256 : i64}
          } {threads = 256 : i64}
        }
        %246 = arith.trunci %192 : i64 to i32
        ascvec.get_buf v, %109, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %97, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
        %247 = ascscalar.bitcast_memref %95 byte_offset = %c0_i64 : memref<16384xbf16, 1> -> memref<?xui8, 1>
        ascvec.vf(%108, %95, %247, %alloc_11, %103, %246 : memref<1024xui8, 1>, memref<16384xbf16, 1>, memref<?xui8, 1>, memref<4096xi32, 1>, memref<16384xui8, 1>, i32) {
        ^bb0(%arg9: memref<1024xui8, 1>, %arg10: memref<16384xbf16, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<4096xi32, 1>, %arg13: memref<16384xui8, 1>, %arg14: i32):
          %c512_i32 = arith.constant 512 : i32
          %c1_i32_12 = arith.constant 1 : i32
          %c2139062143_i32 = arith.constant 2139062143 : i32
          %c-8323200_i32 = arith.constant -8323200 : i32
          %c123_i8 = arith.constant 123 : i8
          %c-1_i8 = arith.constant -1 : i8
          %c60_i8 = arith.constant 60 : i8
          %cst = arith.constant 6.10351526E-5 : f32
          %c64_i32 = arith.constant 64 : i32
          %c0_i16 = arith.constant 0 : i16
          %c0_i32_13 = arith.constant 0 : i32
          %c254_i32 = arith.constant 254 : i32
          %c-15_i32 = arith.constant -15 : i32
          %c15_i32 = arith.constant 15 : i32
          %c255_i32 = arith.constant 255 : i32
          %c32767_i16 = arith.constant 32767 : i16
          %c256_i32 = arith.constant 256 : i32
          %c0_14 = arith.constant 0 : index
          %c1_i8 = arith.constant 1 : i8
          %c32_i8 = arith.constant 32 : i8
          %c65535_i32 = arith.constant 65535 : i32
          %c3_i8 = arith.constant 3 : i8
          %c0_i8 = arith.constant 0 : i8
          %c-128_i8 = arith.constant -128 : i8
          %248 = ascvec.create_mask : !ascvec.mask
          %249 = ascvec.create_mask elem_bits 16 : !ascvec.mask
          %250 = ascvec.create_mask elem_bits 8 : !ascvec.mask
          %251 = ascvec.duplicate_scalar %c-128_i8, %250 : i8 -> <ui8>
          %252 = ascvec.arange %c0_i8 : i8 -> <ui8>
          %253 = ascvec.duplicate_scalar %c3_i8, %250 : i8 -> <ui8>
          %254 = ascvec.and %252, %253, %250 : <ui8>, <ui8> -> <ui8>
          %255 = ascvec.shl %254, 6, %250 : <ui8> -> <ui8>
          %256 = ascvec.shr %252, 2, %250 : <ui8> -> <ui8>
          %257 = ascvec.or %255, %256, %250 : <ui8>, <ui8> -> <ui8>
          %258 = ascvec.duplicate_scalar %c65535_i32, %248 : i32 -> <ui32>
          %259 = ascvec.cmp_lt_scalar %252, %c32_i8, %250 : <ui8>, i8 -> !ascvec.mask
          %260 = ascvec.cmp_lt_scalar %254, %c1_i8, %250 : <ui8>, i8 -> !ascvec.mask
          %261 = ascvec.mask_and %259, %260, %250
          ascvec.store_unalign_begin %arg9 : memref<1024xui8, 1>
          scf.for %arg15 = %c0_i32_13 to %arg14 step %c1_i32_12  : i32 {
            %262 = arith.muli %arg15, %c256_i32 : i32
            %263 = arith.index_cast %262 : i32 to index
            %result0, %result1 = ascvec.loadalign_deintlv %arg10[%263] {dintlv_width = "b16"} : memref<16384xbf16, 1>, index -> <bf16>, <bf16>
            %264 = ascvec.duplicate_scalar %c32767_i16, %249 : i16 -> <ui16>
            %265 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
            %266 = ascvec.and %265, %264, %249 : <ui16>, <ui16> -> <ui16>
            %267 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
            %268 = ascvec.and %267, %264, %249 : <ui16>, <ui16> -> <ui16>
            %269 = ascvec.max %266, %268, %249 : <ui16>
            %270 = ascvec.reduce_max_datablock %269, %249 : <ui16>
            %271 = ascvec.unpack %270 {part = "lower"} : <ui16> -> <ui32>
            %272 = ascvec.shl %271, 16, %248 : <ui32> -> <ui32>
            %273 = arith.muli %arg15, %c512_i32 : i32
            %274 = arith.index_cast %273 : i32 to index
            %result0_17, %result1_18 = ascvec.loadalign_deintlv %arg11[%274] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
            %275 = ascvec.and %result1_18, %251, %250 : <ui8>, <ui8> -> <ui8>
            %276 = ascvec.reinterpret %275 : <ui8> -> <ui32>
            %277 = ascvec.shr %272, 23, %248 : <ui32> -> <ui32>
            %278 = ascvec.cmp_ge_scalar %277, %c255_i32, %248 : <ui32>, i32 -> !ascvec.mask
            %279 = ascvec.max_scalar %277, %c15_i32, %248 : <ui32>, i32
            %280 = ascvec.add_scalar %279, %c-15_i32, %248 : <ui32>, i32
            %281 = ascvec.duplicate_scalar %c255_i32, %278 merge %280 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %282 = ascvec.duplicate_scalar %c254_i32, %248 : i32 -> <ui32>
            %283 = ascvec.sub %282, %281, %248 : <ui32>
            %284 = ascvec.cmp_ne_scalar %281, %c0_i32_13, %248 : <ui32>, i32 -> !ascvec.mask
            %285 = ascvec.shl %283, 23, %284 : <ui32> -> <ui32>
            %286 = ascvec.reinterpret %285 : <ui32> -> <f32>
            %287 = ascvec.reinterpret %286 : <f32> -> <ui32>
            %288 = ascvec.shr %287, 16, %248 : <ui32> -> <ui32>
            %289 = ascvec.pack %288 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %290 = ascvec.arange %c0_i16 : i16 -> <ui16>
            %291 = ascvec.shr %290, 4, %249 : <ui16> -> <ui16>
            %292 = ascvec.gather_reg %289, %291 : <ui16>, <ui16> -> <ui16>
            %293 = ascvec.reinterpret %292 : <ui16> -> <bf16>
            %294 = ascvec.reinterpret %266 : <ui16> -> <bf16>
            %295 = ascvec.mul %294, %293, %249 : <bf16>
            %296 = ascvec.reinterpret %268 : <ui16> -> <bf16>
            %297 = ascvec.mul %296, %293, %249 : <bf16>
            %298 = ascvec.cast %295, %249 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
            %299 = ascvec.cast %295, %249 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
            %300 = ascvec.cast %297, %249 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
            %301 = ascvec.cast %297, %249 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
            %302 = ascvec.reinterpret %293 : <bf16> -> <ui32>
            %303 = arith.muli %arg15, %c64_i32 : i32
            %304 = arith.index_cast %303 : i32 to index
            %305 = ascvec.loadalign %arg12[%304] : memref<4096xi32, 1>, index -> <i32>
            %306 = ascvec.reinterpret %305 : <i32> -> <ui32>
            %307 = ascvec.shl %306, 8, %248 : <ui32> -> <ui32>
            %308 = ascvec.shr %306, 24, %248 : <ui32> -> <ui32>
            %309 = ascvec.or %307, %308, %248 : <ui32>, <ui32> -> <ui32>
            %310 = ascvec.shr %306, 16, %248 : <ui32> -> <ui32>
            %311 = ascvec.and %309, %258, %248 : <ui32>, <ui32> -> <ui32>
            %312 = ascvec.and %306, %258, %248 : <ui32>, <ui32> -> <ui32>
            %313 = ascvec.shr %309, 16, %248 : <ui32> -> <ui32>
            %314 = ascvec.reinterpret %298 : <f32> -> <ui32>
            %315 = ascvec.reinterpret %314 : <ui32> -> <f32>
            %316 = ascvec.max_scalar %315, %cst, %248 : <f32>, f32
            %317 = ascvec.add %315, %316, %248 : <f32>
            %318 = ascvec.reinterpret %317 : <f32> -> <ui32>
            %319 = ascvec.shr %318, 5, %248 : <ui32> -> <ui32>
            %320 = ascvec.add %319, %310, %248 : <ui32>
            %321 = ascvec.reinterpret %300 : <f32> -> <ui32>
            %322 = ascvec.reinterpret %321 : <ui32> -> <f32>
            %323 = ascvec.max_scalar %322, %cst, %248 : <f32>, f32
            %324 = ascvec.add %322, %323, %248 : <f32>
            %325 = ascvec.reinterpret %324 : <f32> -> <ui32>
            %326 = ascvec.shr %325, 5, %248 : <ui32> -> <ui32>
            %327 = ascvec.add %326, %311, %248 : <ui32>
            %328 = ascvec.reinterpret %299 : <f32> -> <ui32>
            %329 = ascvec.reinterpret %328 : <ui32> -> <f32>
            %330 = ascvec.max_scalar %329, %cst, %248 : <f32>, f32
            %331 = ascvec.add %329, %330, %248 : <f32>
            %332 = ascvec.reinterpret %331 : <f32> -> <ui32>
            %333 = ascvec.shr %332, 5, %248 : <ui32> -> <ui32>
            %334 = ascvec.add %333, %312, %248 : <ui32>
            %335 = ascvec.reinterpret %301 : <f32> -> <ui32>
            %336 = ascvec.reinterpret %335 : <ui32> -> <f32>
            %337 = ascvec.max_scalar %336, %cst, %248 : <f32>, f32
            %338 = ascvec.add %336, %337, %248 : <f32>
            %339 = ascvec.reinterpret %338 : <f32> -> <ui32>
            %340 = ascvec.shr %339, 5, %248 : <ui32> -> <ui32>
            %341 = ascvec.add %340, %313, %248 : <ui32>
            %342 = ascvec.reinterpret %320 : <ui32> -> <ui16>
            %343 = ascvec.reinterpret %327 : <ui32> -> <ui16>
            %result0_19, %result1_20 = ascvec.deintlv %342, %343 : <ui16>, <ui16> -> <ui16>, <ui16>
            %344 = ascvec.reinterpret %334 : <ui32> -> <ui16>
            %345 = ascvec.reinterpret %341 : <ui32> -> <ui16>
            %result0_21, %result1_22 = ascvec.deintlv %344, %345 : <ui16>, <ui16> -> <ui16>, <ui16>
            %346 = ascvec.reinterpret %result1_20 : <ui16> -> <ui8>
            %347 = ascvec.reinterpret %result1_22 : <ui16> -> <ui8>
            %result0_23, %result1_24 = ascvec.deintlv %346, %347 : <ui8>, <ui8> -> <ui8>, <ui8>
            %348 = ascvec.gather_reg %result0_23, %257 : <ui8>, <ui8> -> <ui8>
            %349 = ascvec.reinterpret %348 : <ui8> -> <ui32>
            %350 = ascvec.reinterpret %349 : <ui32> -> <ui8>
            %351 = ascvec.add_scalar %350, %c60_i8, %250 : <ui8>, i8
            %352 = ascvec.cmp_ne_scalar %351, %c-1_i8, %250 : <ui8>, i8 -> !ascvec.mask
            %353 = ascvec.min_scalar %351, %c123_i8, %352 : <ui8>, i8
            %354 = ascvec.reinterpret %353 : <ui8> -> <ui32>
            %355 = ascvec.or %354, %276, %248 : <ui32>, <ui32> -> <ui32>
            %356 = ascvec.reinterpret %302 : <ui32> -> <ui32>
            %357 = ascvec.cmp_eq_scalar %356, %c-8323200_i32, %248 : <ui32>, i32 -> !ascvec.mask
            %358 = ascvec.duplicate_scalar %c2139062143_i32, %357 merge %355 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %359 = ascvec.reinterpret %358 : <ui32> -> <ui8>
            ascvec.storealign %arg13[%263], %359, %250 : memref<16384xui8, 1>, index, <ui8>
            %360 = ascvec.reinterpret %281 : <ui32> -> <ui8>
            %361 = ascvec.squeeze_store_reg %360, %261 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg9[%c0_14], %361 : memref<1024xui8, 1>, <ui8>
          }
          %c0_15 = arith.constant 0 : index
          %c1_16 = arith.constant 1 : index
          scf.for %arg15 = %c0_15 to %c1_16 step %c1_16 {
            ascvec.store_unalign_post %arg9[%c0_14] : memref<1024xui8, 1>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %97, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %109, 0 {cannir.buffer.generated_sync}
      } else {
        %192 = arith.addi %88, %c63_i64 : i64
        %193 = arith.remsi %192, %c64_i64 : i64
        %194 = arith.cmpi ne, %193, %c0_i64 : i64
        %195 = arith.cmpi slt, %193, %c0_i64 : i64
        %196 = arith.andi %194, %195 : i1
        %197 = arith.divsi %192, %c64_i64 : i64
        %198 = arith.extui %196 : i1 to i64
        %199 = arith.subi %197, %198 : i64
        %200 = arith.addi %88, %c1023_i64 : i64
        %201 = arith.remsi %200, %c1024_i64 : i64
        %202 = arith.cmpi ne, %201, %c0_i64 : i64
        %203 = arith.cmpi slt, %201, %c0_i64 : i64
        %204 = arith.andi %202, %203 : i1
        %205 = arith.divsi %200, %c1024_i64 : i64
        %206 = arith.extui %204 : i1 to i64
        %207 = arith.subi %205, %206 : i64
        %208 = arith.trunci %1 : i64 to i32
        %209 = arith.extui %208 : i32 to i64
        %210 = arith.subi %1, %209 : i64
        %211 = arith.remsi %210, %c4294967296_i64 : i64
        %212 = arith.cmpi ne, %211, %c0_i64 : i64
        %213 = arith.cmpi slt, %211, %c0_i64 : i64
        %214 = arith.andi %212, %213 : i1
        %215 = arith.divsi %210, %c4294967296_i64 : i64
        %216 = arith.extui %214 : i1 to i64
        %217 = arith.subi %215, %216 : i64
        %218 = arith.trunci %217 : i64 to i32
        %219 = arith.remsi %111, %c16_i64 : i64
        %220 = arith.cmpi ne, %219, %c0_i64 : i64
        %221 = arith.cmpi slt, %219, %c0_i64 : i64
        %222 = arith.andi %220, %221 : i1
        %223 = arith.divsi %111, %c16_i64 : i64
        %224 = arith.extui %222 : i1 to i64
        %225 = arith.subi %223, %224 : i64
        %226 = arith.remsi %arg4, %c16_i64 : i64
        %227 = arith.cmpi ne, %226, %c0_i64 : i64
        %228 = arith.cmpi slt, %226, %c0_i64 : i64
        %229 = arith.andi %227, %228 : i1
        %230 = arith.divsi %arg4, %c16_i64 : i64
        %231 = arith.extui %229 : i1 to i64
        %232 = arith.subi %230, %231 : i64
        %233 = arith.remsi %225, %232 : i64
        %234 = arith.cmpi ne, %233, %c0_i64 : i64
        %235 = arith.cmpi slt, %233, %c0_i64 : i64
        %236 = arith.cmpi slt, %232, %c0_i64 : i64
        %237 = arith.xori %235, %236 : i1
        %238 = arith.andi %234, %237 : i1
        %239 = arith.divsi %225, %232 : i64
        %240 = arith.extui %238 : i1 to i64
        %241 = arith.subi %239, %240 : i64
        %242 = arith.select %238, %232, %c0_i64 : i64
        %243 = arith.addi %233, %242 : i64
        %244 = arith.trunci %241 : i64 to i32
        %245 = arith.extui %244 : i32 to i64
        %246 = arith.subi %241, %245 : i64
        %247 = arith.remsi %246, %c4294967296_i64 : i64
        %248 = arith.cmpi ne, %247, %c0_i64 : i64
        %249 = arith.cmpi slt, %247, %c0_i64 : i64
        %250 = arith.andi %248, %249 : i1
        %251 = arith.divsi %246, %c4294967296_i64 : i64
        %252 = arith.extui %250 : i1 to i64
        %253 = arith.subi %251, %252 : i64
        %254 = arith.trunci %253 : i64 to i32
        %255 = arith.trunci %243 : i64 to i32
        %256 = arith.extui %255 : i32 to i64
        %257 = arith.subi %243, %256 : i64
        %258 = arith.remsi %257, %c4294967296_i64 : i64
        %259 = arith.cmpi ne, %258, %c0_i64 : i64
        %260 = arith.cmpi slt, %258, %c0_i64 : i64
        %261 = arith.andi %259, %260 : i1
        %262 = arith.divsi %257, %c4294967296_i64 : i64
        %263 = arith.extui %261 : i1 to i64
        %264 = arith.subi %262, %263 : i64
        %265 = arith.trunci %264 : i64 to i32
        %266 = arith.remsi %c64_i64, %232 : i64
        %267 = arith.cmpi ne, %266, %c0_i64 : i64
        %268 = arith.cmpi slt, %266, %c0_i64 : i64
        %269 = arith.xori %268, %236 : i1
        %270 = arith.andi %267, %269 : i1
        %271 = arith.divsi %c64_i64, %232 : i64
        %272 = arith.extui %270 : i1 to i64
        %273 = arith.subi %271, %272 : i64
        %274 = arith.select %270, %232, %c0_i64 : i64
        %275 = arith.addi %266, %274 : i64
        %276 = arith.trunci %207 : i64 to i32
        ascvec.vf(%232, %3, %208, %218, %alloc_11, %275, %273, %276, %244, %254, %255, %265 : i64, i64, i32, i32, memref<4096xi32, 1>, i64, i64, i32, i32, i32, i32, i32) {
        ^bb0(%arg9: i64, %arg10: i64, %arg11: i32, %arg12: i32, %arg13: memref<4096xi32, 1>, %arg14: i64, %arg15: i64, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32):
          %c1_i32_12 = arith.constant 1 : i32
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
          %c1993301258_i32_13 = arith.constant 1993301258 : i32
          %c1013904242_i32_14 = arith.constant 1013904242 : i32
          %c-1150833019_i32_15 = arith.constant -1150833019 : i32
          %c-1640531527_i32_16 = arith.constant -1640531527 : i32
          %c-845247145_i32 = arith.constant -845247145 : i32
          %c-766435501_i32 = arith.constant -766435501 : i32
          %c16383_i32 = arith.constant 16383 : i32
          %c1073741824_i32_17 = arith.constant 1073741824 : i32
          %c4_i32_18 = arith.constant 4 : i32
          %c0_i64_19 = arith.constant 0 : i64
          %c0_i32_20 = arith.constant 0 : i32
          %c256_i32 = arith.constant 256 : i32
          %c4294967296_i64_21 = arith.constant 4294967296 : i64
          %279:4 = scf.for %arg21 = %c0_i32_20 to %arg16 step %c1_i32_12 iter_args(%arg22 = %arg17, %arg23 = %arg18, %arg24 = %arg19, %arg25 = %arg20) -> (i32, i32, i32, i32)  : i32 {
            %280 = arith.extui %arg22 : i32 to i64
            %281 = arith.extui %arg23 : i32 to i64
            %282 = arith.muli %281, %c4294967296_i64_21 : i64
            %283 = arith.addi %280, %282 : i64
            %284 = arith.extui %arg24 : i32 to i64
            %285 = arith.extui %arg25 : i32 to i64
            %286 = arith.muli %285, %c4294967296_i64_21 : i64
            %287 = arith.addi %284, %286 : i64
            %288 = arith.muli %arg21, %c256_i32 : i32
            %289 = ascvec.create_mask : !ascvec.mask
            %290 = ascvec.arange %c0_i32_20 : i32 -> <ui32>
            %291 = arith.muli %283, %arg9 : i64
            %292 = arith.addi %291, %287 : i64
            %293 = arith.trunci %292 : i64 to i32
            %294 = arith.extui %293 : i32 to i64
            %295 = arith.subi %292, %294 : i64
            %296 = arith.remsi %295, %c4294967296_i64_21 : i64
            %297 = arith.cmpi ne, %296, %c0_i64_19 : i64
            %298 = arith.cmpi slt, %296, %c0_i64_19 : i64
            %299 = arith.andi %297, %298 : i1
            %300 = arith.divsi %295, %c4294967296_i64_21 : i64
            %301 = arith.extui %299 : i1 to i64
            %302 = arith.subi %300, %301 : i64
            %303 = arith.trunci %302 : i64 to i32
            %304 = ascvec.duplicate_scalar %293, %289 : i32 -> <ui32>
            %305 = ascvec.duplicate_scalar %303, %289 : i32 -> <ui32>
            %306 = ascvec.duplicate_scalar %c0_i32_20, %289 : i32 -> <ui32>
            %carry, %dst = ascvec.addc %304, %290, %289 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %carry_22, %dst_23 = ascvec.addc %305, %306, %carry, %289 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %307 = arith.trunci %arg10 : i64 to i32
            %308 = arith.extui %307 : i32 to i64
            %309 = arith.subi %arg10, %308 : i64
            %310 = arith.remsi %309, %c4294967296_i64_21 : i64
            %311 = arith.cmpi ne, %310, %c0_i64_19 : i64
            %312 = arith.cmpi slt, %310, %c0_i64_19 : i64
            %313 = arith.andi %311, %312 : i1
            %314 = arith.divsi %309, %c4294967296_i64_21 : i64
            %315 = arith.extui %313 : i1 to i64
            %316 = arith.subi %314, %315 : i64
            %317 = arith.trunci %316 : i64 to i32
            %318 = arith.divui %307, %c4_i32_18 : i32
            %319 = arith.remui %317, %c4_i32_18 : i32
            %320 = arith.muli %319, %c1073741824_i32_17 : i32
            %321 = arith.addi %318, %320 : i32
            %322 = arith.divui %317, %c4_i32_18 : i32
            %323 = ascvec.shr %dst, 14, %289 : <ui32> -> <ui32>
            %324 = ascvec.shl %dst_23, 18, %289 : <ui32> -> <ui32>
            %325 = ascvec.or %323, %324, %289 : <ui32>, <ui32> -> <ui32>
            %326 = ascvec.shr %dst_23, 14, %289 : <ui32> -> <ui32>
            %327 = ascvec.duplicate_scalar %321, %289 : i32 -> <ui32>
            %328 = ascvec.duplicate_scalar %322, %289 : i32 -> <ui32>
            %carry_24, %dst_25 = ascvec.addc %327, %325, %289 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %carry_26, %dst_27 = ascvec.addc %328, %326, %carry_24, %289 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %329 = ascvec.duplicate_scalar %c16383_i32, %289 : i32 -> <ui32>
            %330 = ascvec.and %dst, %329, %289 : <ui32>, <ui32> -> <ui32>
            %331 = ascvec.duplicate_scalar %c-766435501_i32, %289 : i32 -> <ui32>
            %332 = ascvec.duplicate_scalar %c-845247145_i32, %289 : i32 -> <ui32>
            %dst0, %dst1 = ascvec.mull %306, %331, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_28, %dst1_29 = ascvec.mull %dst_25, %332, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %333 = ascvec.xor %dst1_29, %330, %289 : <ui32>, <ui32> -> <ui32>
            %334 = ascvec.xor %dst1, %dst_27, %289 : <ui32>, <ui32> -> <ui32>
            %335 = ascvec.duplicate_scalar %arg11, %289 : i32 -> <ui32>
            %336 = ascvec.duplicate_scalar %arg12, %289 : i32 -> <ui32>
            %337 = ascvec.xor %333, %335, %289 : <ui32>, <ui32> -> <ui32>
            %338 = ascvec.xor %334, %336, %289 : <ui32>, <ui32> -> <ui32>
            %dst0_30, %dst1_31 = ascvec.mull %337, %331, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_32, %dst1_33 = ascvec.mull %338, %332, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %339 = ascvec.xor %dst1_33, %dst0_28, %289 : <ui32>, <ui32> -> <ui32>
            %340 = ascvec.xor %dst1_31, %dst0, %289 : <ui32>, <ui32> -> <ui32>
            %341 = arith.addi %arg11, %c-1640531527_i32_16 : i32
            %342 = ascvec.duplicate_scalar %341, %289 : i32 -> <ui32>
            %343 = arith.addi %arg12, %c-1150833019_i32_15 : i32
            %344 = ascvec.duplicate_scalar %343, %289 : i32 -> <ui32>
            %345 = ascvec.xor %339, %342, %289 : <ui32>, <ui32> -> <ui32>
            %346 = ascvec.xor %340, %344, %289 : <ui32>, <ui32> -> <ui32>
            %dst0_34, %dst1_35 = ascvec.mull %345, %331, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_36, %dst1_37 = ascvec.mull %346, %332, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %347 = ascvec.xor %dst1_37, %dst0_32, %289 : <ui32>, <ui32> -> <ui32>
            %348 = ascvec.xor %dst1_35, %dst0_30, %289 : <ui32>, <ui32> -> <ui32>
            %349 = arith.addi %arg11, %c1013904242_i32_14 : i32
            %350 = ascvec.duplicate_scalar %349, %289 : i32 -> <ui32>
            %351 = arith.addi %arg12, %c1993301258_i32_13 : i32
            %352 = ascvec.duplicate_scalar %351, %289 : i32 -> <ui32>
            %353 = ascvec.xor %347, %350, %289 : <ui32>, <ui32> -> <ui32>
            %354 = ascvec.xor %348, %352, %289 : <ui32>, <ui32> -> <ui32>
            %dst0_38, %dst1_39 = ascvec.mull %353, %331, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_40, %dst1_41 = ascvec.mull %354, %332, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %355 = ascvec.xor %dst1_41, %dst0_36, %289 : <ui32>, <ui32> -> <ui32>
            %356 = ascvec.xor %dst1_39, %dst0_34, %289 : <ui32>, <ui32> -> <ui32>
            %357 = arith.addi %arg11, %c-626627285_i32 : i32
            %358 = ascvec.duplicate_scalar %357, %289 : i32 -> <ui32>
            %359 = arith.addi %arg12, %c842468239_i32 : i32
            %360 = ascvec.duplicate_scalar %359, %289 : i32 -> <ui32>
            %361 = ascvec.xor %355, %358, %289 : <ui32>, <ui32> -> <ui32>
            %362 = ascvec.xor %356, %360, %289 : <ui32>, <ui32> -> <ui32>
            %dst0_42, %dst1_43 = ascvec.mull %361, %331, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_44, %dst1_45 = ascvec.mull %362, %332, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %363 = ascvec.xor %dst1_45, %dst0_40, %289 : <ui32>, <ui32> -> <ui32>
            %364 = ascvec.xor %dst1_43, %dst0_38, %289 : <ui32>, <ui32> -> <ui32>
            %365 = arith.addi %arg11, %c2027808484_i32 : i32
            %366 = ascvec.duplicate_scalar %365, %289 : i32 -> <ui32>
            %367 = arith.addi %arg12, %c-308364780_i32 : i32
            %368 = ascvec.duplicate_scalar %367, %289 : i32 -> <ui32>
            %369 = ascvec.xor %363, %366, %289 : <ui32>, <ui32> -> <ui32>
            %370 = ascvec.xor %364, %368, %289 : <ui32>, <ui32> -> <ui32>
            %dst0_46, %dst1_47 = ascvec.mull %369, %331, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_48, %dst1_49 = ascvec.mull %370, %332, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %371 = ascvec.xor %dst1_49, %dst0_44, %289 : <ui32>, <ui32> -> <ui32>
            %372 = ascvec.xor %dst1_47, %dst0_42, %289 : <ui32>, <ui32> -> <ui32>
            %373 = arith.addi %arg11, %c387276957_i32 : i32
            %374 = ascvec.duplicate_scalar %373, %289 : i32 -> <ui32>
            %375 = arith.addi %arg12, %c-1459197799_i32 : i32
            %376 = ascvec.duplicate_scalar %375, %289 : i32 -> <ui32>
            %377 = ascvec.xor %371, %374, %289 : <ui32>, <ui32> -> <ui32>
            %378 = ascvec.xor %372, %376, %289 : <ui32>, <ui32> -> <ui32>
            %dst0_50, %dst1_51 = ascvec.mull %377, %331, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_52, %dst1_53 = ascvec.mull %378, %332, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %379 = ascvec.xor %dst1_53, %dst0_48, %289 : <ui32>, <ui32> -> <ui32>
            %380 = ascvec.xor %dst1_51, %dst0_46, %289 : <ui32>, <ui32> -> <ui32>
            %381 = arith.addi %arg11, %c-1253254570_i32 : i32
            %382 = ascvec.duplicate_scalar %381, %289 : i32 -> <ui32>
            %383 = arith.addi %arg12, %c1684936478_i32 : i32
            %384 = ascvec.duplicate_scalar %383, %289 : i32 -> <ui32>
            %385 = ascvec.xor %379, %382, %289 : <ui32>, <ui32> -> <ui32>
            %386 = ascvec.xor %380, %384, %289 : <ui32>, <ui32> -> <ui32>
            %dst0_54, %dst1_55 = ascvec.mull %385, %331, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_56, %dst1_57 = ascvec.mull %386, %332, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %387 = ascvec.xor %dst1_57, %dst0_52, %289 : <ui32>, <ui32> -> <ui32>
            %388 = ascvec.xor %dst1_55, %dst0_50, %289 : <ui32>, <ui32> -> <ui32>
            %389 = arith.addi %arg11, %c1401181199_i32 : i32
            %390 = ascvec.duplicate_scalar %389, %289 : i32 -> <ui32>
            %391 = arith.addi %arg12, %c534103459_i32 : i32
            %392 = ascvec.duplicate_scalar %391, %289 : i32 -> <ui32>
            %393 = ascvec.xor %387, %390, %289 : <ui32>, <ui32> -> <ui32>
            %394 = ascvec.xor %388, %392, %289 : <ui32>, <ui32> -> <ui32>
            %dst0_58, %dst1_59 = ascvec.mull %393, %331, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_60, %dst1_61 = ascvec.mull %394, %332, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %395 = ascvec.xor %dst1_61, %dst0_56, %289 : <ui32>, <ui32> -> <ui32>
            %396 = ascvec.xor %dst1_59, %dst0_54, %289 : <ui32>, <ui32> -> <ui32>
            %397 = arith.addi %arg11, %c-239350328_i32 : i32
            %398 = ascvec.duplicate_scalar %397, %289 : i32 -> <ui32>
            %399 = arith.addi %arg12, %c-616729560_i32 : i32
            %400 = ascvec.duplicate_scalar %399, %289 : i32 -> <ui32>
            %401 = ascvec.xor %395, %398, %289 : <ui32>, <ui32> -> <ui32>
            %402 = ascvec.xor %396, %400, %289 : <ui32>, <ui32> -> <ui32>
            %dst0_62, %dst1_63 = ascvec.mull %401, %331, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_64, %dst1_65 = ascvec.mull %402, %332, %289 : <ui32>, <ui32> -> <ui32>, <ui32>
            %403 = ascvec.xor %dst1_65, %dst0_60, %289 : <ui32>, <ui32> -> <ui32>
            %404 = ascvec.xor %dst1_63, %dst0_58, %289 : <ui32>, <ui32> -> <ui32>
            %405 = arith.addi %arg11, %c-1879881855_i32 : i32
            %406 = ascvec.duplicate_scalar %405, %289 : i32 -> <ui32>
            %407 = arith.addi %arg12, %c-1767562579_i32 : i32
            %408 = ascvec.duplicate_scalar %407, %289 : i32 -> <ui32>
            %409 = ascvec.xor %403, %406, %289 : <ui32>, <ui32> -> <ui32>
            %410 = ascvec.xor %404, %408, %289 : <ui32>, <ui32> -> <ui32>
            %411 = ascvec.reinterpret %409 : <ui32> -> <i32>
            %412 = arith.index_cast %288 : i32 to index
            ascvec.storealign %arg13[%412], %411, %289 : memref<4096xi32, 1>, index, <i32>
            %413 = arith.addi %288, %c64_i32 : i32
            %414 = ascvec.reinterpret %dst0_64 : <ui32> -> <i32>
            %415 = arith.index_cast %413 : i32 to index
            ascvec.storealign %arg13[%415], %414, %289 : memref<4096xi32, 1>, index, <i32>
            %416 = arith.addi %288, %c128_i32 : i32
            %417 = ascvec.reinterpret %410 : <ui32> -> <i32>
            %418 = arith.index_cast %416 : i32 to index
            ascvec.storealign %arg13[%418], %417, %289 : memref<4096xi32, 1>, index, <i32>
            %419 = arith.addi %288, %c192_i32 : i32
            %420 = ascvec.reinterpret %dst0_62 : <ui32> -> <i32>
            %421 = arith.index_cast %419 : i32 to index
            ascvec.storealign %arg13[%421], %420, %289 : memref<4096xi32, 1>, index, <i32>
            %422 = arith.addi %287, %arg14 : i64
            %423 = arith.cmpi sge, %422, %arg9 : i64
            %424 = arith.extui %423 : i1 to i64
            %425 = arith.addi %283, %arg15 : i64
            %426 = arith.addi %425, %424 : i64
            %427 = arith.muli %424, %arg9 : i64
            %428 = arith.subi %422, %427 : i64
            %429 = arith.trunci %426 : i64 to i32
            %430 = arith.extui %429 : i32 to i64
            %431 = arith.subi %426, %430 : i64
            %432 = arith.remsi %431, %c4294967296_i64_21 : i64
            %433 = arith.cmpi ne, %432, %c0_i64_19 : i64
            %434 = arith.cmpi slt, %432, %c0_i64_19 : i64
            %435 = arith.andi %433, %434 : i1
            %436 = arith.divsi %431, %c4294967296_i64_21 : i64
            %437 = arith.extui %435 : i1 to i64
            %438 = arith.subi %436, %437 : i64
            %439 = arith.trunci %438 : i64 to i32
            %440 = arith.trunci %428 : i64 to i32
            %441 = arith.extui %440 : i32 to i64
            %442 = arith.subi %428, %441 : i64
            %443 = arith.remsi %442, %c4294967296_i64_21 : i64
            %444 = arith.cmpi ne, %443, %c0_i64_19 : i64
            %445 = arith.cmpi slt, %443, %c0_i64_19 : i64
            %446 = arith.andi %444, %445 : i1
            %447 = arith.divsi %442, %c4294967296_i64_21 : i64
            %448 = arith.extui %446 : i1 to i64
            %449 = arith.subi %447, %448 : i64
            %450 = arith.trunci %449 : i64 to i32
            scf.yield %429, %439, %440, %450 : i32, i32, i32, i32
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %277 = arith.trunci %199 : i64 to i32
        %278 = arith.muli %277, %c2_i32 : i32
        ascvec.get_buf v, %97, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %104, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %109, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%88, %95, %alloc_11, %103, %alloc_10, %277, %278, %108 : i64, memref<16384xbf16, 1>, memref<4096xi32, 1>, memref<16384xui8, 1>, memref<512xui32, 1>, i32, i32, memref<1024xui8, 1>) {
        ^bb0(%arg9: i64, %arg10: memref<16384xbf16, 1>, %arg11: memref<4096xi32, 1>, %arg12: memref<16384xui8, 1>, %arg13: memref<512xui32, 1>, %arg14: i32, %arg15: i32, %arg16: memref<1024xui8, 1>):
          %c2_i32_12 = arith.constant 2 : i32
          %c2139095040_i32 = arith.constant 2139095040 : i32
          %c127_i32 = arith.constant 127 : i32
          %c123_i32 = arith.constant 123 : i32
          %c947912704_i32 = arith.constant 947912704 : i32
          %cst = arith.constant 4.2949673E+9 : f32
          %c29360128_i32 = arith.constant 29360128 : i32
          %c128_i32 = arith.constant 128 : i32
          %c65535_i32 = arith.constant 65535 : i32
          %c4_i32_13 = arith.constant 4 : i32
          %c256_i32 = arith.constant 256 : i32
          %c1_i32_14 = arith.constant 1 : i32
          %c16_i32 = arith.constant 16 : i32
          %c3_i32 = arith.constant 3 : i32
          %c2139160576_i32 = arith.constant 2139160576 : i32
          %c254_i32 = arith.constant 254 : i32
          %c-15_i32 = arith.constant -15 : i32
          %c15_i32 = arith.constant 15 : i32
          %c255_i32 = arith.constant 255 : i32
          %c2147483647_i32 = arith.constant 2147483647 : i32
          %cst_15 = arith.constant 0.000000e+00 : f32
          %c0_16 = arith.constant 0 : index
          %c64_i32 = arith.constant 64 : i32
          %c32_i32 = arith.constant 32 : i32
          %c0_i32_17 = arith.constant 0 : i32
          %279 = ascvec.create_mask : !ascvec.mask
          %280 = ascvec.arange %c0_i32_17 : i32 -> <ui32>
          %281 = ascvec.cmp_lt_scalar %280, %c32_i32, %279 : <ui32>, i32 -> !ascvec.mask
          %282 = ascvec.cmp_ge_scalar %280, %c32_i32, %279 : <ui32>, i32 -> !ascvec.mask
          scf.for %arg17 = %c0_i32_17 to %arg14 step %c1_i32_14  : i32 {
            %283 = arith.muli %arg17, %c64_i32 : i32
            %284 = arith.trunci %arg9 : i64 to i32
            %285 = arith.subi %284, %283 : i32
            %mask, %remaining_out = ascvec.update_mask %285 : !ascvec.mask
            %286 = arith.index_cast %283 : i32 to index
            %287 = ascvec.loadalign_unpack %arg10[%286] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %288 = ascvec.cast %287, %279 {rounding = 6 : i32} : <bf16> -> <f32>
            %289 = ascvec.duplicate_scalar %cst_15, %279 : f32 -> <f32>
            %290 = ascvec.select %288, %289, %mask : <f32>
            %291 = ascvec.reinterpret %290 : <f32> -> <ui32>
            %292 = ascvec.duplicate_scalar %c2147483647_i32, %279 : i32 -> <ui32>
            %293 = ascvec.and %291, %292, %279 : <ui32>, <ui32> -> <ui32>
            %294 = ascvec.reduce_max %293, %281 : <ui32>
            %295 = ascvec.duplicate %294, %279 : <ui32> -> <ui32>
            %296 = ascvec.reduce_max %293, %282 : <ui32>
            %297 = ascvec.duplicate %296, %279 : <ui32> -> <ui32>
            %298 = ascvec.select %295, %297, %281 : <ui32>
            %299 = ascvec.shr %298, 23, %279 : <ui32> -> <ui32>
            %300 = ascvec.cmp_ge_scalar %299, %c255_i32, %279 : <ui32>, i32 -> !ascvec.mask
            %301 = ascvec.max_scalar %299, %c15_i32, %279 : <ui32>, i32
            %302 = ascvec.add_scalar %301, %c-15_i32, %279 : <ui32>, i32
            %303 = ascvec.duplicate_scalar %c255_i32, %300 merge %302 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %304 = ascvec.duplicate_scalar %c254_i32, %279 : i32 -> <ui32>
            %305 = ascvec.sub %304, %303, %279 : <ui32>
            %306 = ascvec.cmp_ne_scalar %303, %c0_i32_17, %279 : <ui32>, i32 -> !ascvec.mask
            %307 = ascvec.shl %305, 23, %306 : <ui32> -> <ui32>
            %308 = ascvec.duplicate_scalar %c2139160576_i32, %300 merge %307 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %309 = ascvec.reinterpret %308 : <ui32> -> <f32>
            %310 = ascvec.mul %290, %309, %279 : <f32>
            %311 = ascvec.shr %280, 2, %279 : <ui32> -> <ui32>
            %312 = ascvec.duplicate_scalar %c3_i32, %279 : i32 -> <ui32>
            %313 = ascvec.and %311, %312, %279 : <ui32>, <ui32> -> <ui32>
            %314 = ascvec.shl %313, 6, %279 : <ui32> -> <ui32>
            %315 = ascvec.shr %280, 4, %279 : <ui32> -> <ui32>
            %316 = ascvec.add %314, %315, %279 : <ui32>
            %317 = arith.remsi %arg17, %c16_i32 : i32
            %318 = arith.cmpi ne, %317, %c0_i32_17 : i32
            %319 = arith.cmpi slt, %317, %c0_i32_17 : i32
            %320 = arith.andi %318, %319 : i1
            %321 = arith.divsi %arg17, %c16_i32 : i32
            %322 = arith.extui %320 : i1 to i32
            %323 = arith.subi %321, %322 : i32
            %324 = arith.muli %323, %c256_i32 : i32
            %325 = arith.select %320, %c16_i32, %c0_i32_17 : i32
            %326 = arith.addi %317, %325 overflow<nsw> : i32
            %327 = arith.muli %326, %c4_i32_13 overflow<nsw> : i32
            %328 = arith.addi %324, %327 : i32
            %329 = ascvec.add_scalar %316, %328, %279 : <ui32>, i32
            %330 = ascvec.gather %arg11[%c0_16], %329, %279 : memref<4096xi32, 1>, <ui32>, !ascvec.mask -> <i32>
            %331 = ascvec.reinterpret %330 : <i32> -> <ui32>
            %332 = ascvec.shl %331, 8, %279 : <ui32> -> <ui32>
            %333 = ascvec.shr %331, 24, %279 : <ui32> -> <ui32>
            %334 = ascvec.or %332, %333, %279 : <ui32>, <ui32> -> <ui32>
            %335 = ascvec.and %280, %312, %279 : <ui32>, <ui32> -> <ui32>
            %336 = ascvec.duplicate_scalar %c1_i32_14, %279 : i32 -> <ui32>
            %337 = ascvec.and %335, %336, %279 : <ui32>, <ui32> -> <ui32>
            %338 = ascvec.cmp_eq_scalar %337, %c0_i32_17, %279 : <ui32>, i32 -> !ascvec.mask
            %339 = ascvec.select %331, %334, %338 : <ui32>
            %340 = ascvec.cmp_eq_scalar %335, %c0_i32_17, %279 : <ui32>, i32 -> !ascvec.mask
            %341 = ascvec.cmp_eq_scalar %335, %c3_i32, %279 : <ui32>, i32 -> !ascvec.mask
            %342 = ascvec.mask_or %340, %341, %279
            %343 = ascvec.shr %339, 16, %279 : <ui32> -> <ui32>
            %344 = ascvec.duplicate_scalar %c65535_i32, %279 : i32 -> <ui32>
            %345 = ascvec.and %339, %344, %279 : <ui32>, <ui32> -> <ui32>
            %346 = ascvec.select %343, %345, %342 : <ui32>
            %347 = ascvec.reinterpret %310 : <f32> -> <ui32>
            %348 = ascvec.shr %347, 24, %279 : <ui32> -> <ui32>
            %349 = ascvec.duplicate_scalar %c128_i32, %279 : i32 -> <ui32>
            %350 = ascvec.and %348, %349, %279 : <ui32>, <ui32> -> <ui32>
            %351 = ascvec.and %347, %292, %279 : <ui32>, <ui32> -> <ui32>
            %352 = ascvec.shr %351, 5, %279 : <ui32> -> <ui32>
            %353 = ascvec.duplicate_scalar %c29360128_i32, %279 : i32 -> <ui32>
            %354 = ascvec.sub %352, %353, %279 : <ui32>
            %355 = ascvec.reinterpret %351 : <ui32> -> <f32>
            %356 = ascvec.mul_scalar %355, %cst, %279 : <f32>, f32
            %357 = ascvec.cast %356, %279 {rounding = 4 : i32} : <f32> -> <i32>
            %358 = ascvec.reinterpret %357 : <i32> -> <ui32>
            %359 = ascvec.cmp_ge_scalar %351, %c947912704_i32, %279 : <ui32>, i32 -> !ascvec.mask
            %360 = ascvec.select %354, %358, %359 : <ui32>
            %361 = ascvec.and %346, %344, %279 : <ui32>, <ui32> -> <ui32>
            %362 = ascvec.add %360, %361, %279 : <ui32>
            %363 = ascvec.shr %362, 16, %279 : <ui32> -> <ui32>
            %364 = ascvec.min_scalar %363, %c123_i32, %279 : <ui32>, i32
            %365 = ascvec.or %364, %350, %279 : <ui32>, <ui32> -> <ui32>
            %366 = ascvec.duplicate_scalar %c127_i32, %279 : i32 -> <ui32>
            %367 = ascvec.cmp_gt_scalar %351, %c2139095040_i32, %279 : <ui32>, i32 -> !ascvec.mask
            %368 = ascvec.select %366, %365, %367 : <ui32>
            ascvec.storealign_pack %arg12[%286], %368, %mask {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
            %369 = arith.muli %arg17, %c2_i32_12 : i32
            %370 = arith.index_cast %369 : i32 to index
            ascvec.storealign_1st %arg13[%370], %303 : memref<512xui32, 1>, index, <ui32>
            %371 = ascvec.duplicate_scalar %c32_i32, %279 : i32 -> <ui32>
            %372 = ascvec.gather_reg %303, %371 : <ui32>, <ui32> -> <ui32>
            %373 = arith.addi %369, %c1_i32_14 : i32
            %374 = arith.index_cast %373 : i32 to index
            ascvec.storealign_1st %arg13[%374], %372 : memref<512xui32, 1>, index, <ui32>
          }
          ascvec.mem_bar vst_vld
          scf.for %arg17 = %c0_i32_17 to %arg15 step %c64_i32  : i32 {
            %283 = arith.subi %arg15, %arg17 : i32
            %mask, %remaining_out = ascvec.update_mask %283 : !ascvec.mask
            %284 = ascvec.add_scalar %280, %arg17, %279 : <ui32>, i32
            %285 = ascvec.gather %arg13[%c0_16], %284, %mask : memref<512xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
            %286 = arith.index_cast %arg17 : i32 to index
            ascvec.storealign_pack %arg16[%286], %285, %mask {mode = "b32_to_b8"} : memref<1024xui8, 1>, index, <ui32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %109, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %104, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %97, 0 {cannir.buffer.generated_sync}
      }
      %116 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %117 = cannir.extract_shape %116 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %118 = cannir.extract_dim %117[0] : !cannir.shape<(-1, -1)>
      %119 = arith.minsi %73, %118 : i64
      %120 = arith.subi %118, %119 : i64
      %121 = arith.minsi %120, %c1_i64 : i64
      %122 = arith.cmpi sle, %73, %118 : i64
      %123 = arith.select %122, %121, %c0_i64 : i64
      %124 = arith.cmpi sgt, %123, %c1_i64 : i64
      %125 = arith.select %124, %c1_i64, %123 : i64
      %126 = cannir.extract_dim %117[1] : !cannir.shape<(-1, -1)>
      %127 = arith.maxsi %50, %c1_i64 : i64
      %128 = arith.divsi %126, %127 : i64
      %129 = arith.minsi %84, %128 : i64
      %130 = arith.muli %129, %50 : i64
      %131 = arith.subi %126, %130 : i64
      %132 = arith.minsi %50, %131 : i64
      %133 = arith.cmpi sle, %84, %128 : i64
      %134 = arith.select %133, %132, %c0_i64 : i64
      %135 = arith.cmpi sgt, %134, %c16384_i64 : i64
      %136 = arith.select %135, %c16384_i64, %134 : i64
      %137 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %138 = cannir.extract_stride %137 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %139 = cannir.extract_dim %138[0] : !cannir.stride<(-1, 1)>
      %140 = arith.muli %73, %139 : i64
      %141 = arith.addi %140, %85 : i64
      %142 = arith.muli %141, %c8_i64 : i64
      %143 = ascscalar.memref_address %alloc_1 : memref<1xui8>
      %144 = arith.divui %142, %c8_i64 : i64
      %145 = arith.addi %143, %144 : i64
      %146 = ascscalar.bind_gm_ptr %145 : memref<1xui8>
      %147 = ascscalar.bind_local_ptr %102 : memref<1xui8, 1>
      %148 = cannir.extract_stride %116 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %149 = cannir.extract_dim %148[0] : !cannir.stride<(-1, 1)>
      %150 = arith.cmpi ugt, %125, %c0_i64 : i64
      %151 = arith.cmpi ugt, %136, %c0_i64 : i64
      %152 = arith.andi %150, %151 : i1
      scf.if %152 {
        ascvec.get_buf mte3, %104, 0 {cannir.buffer.generated_sync}
        %192 = arith.trunci %125 : i64 to i32
        %193 = arith.trunci %136 : i64 to i32
        ascvec.copy_ub2gm %146[%c0], %147[%c0], %192, %193, %149, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %104, 0 {cannir.buffer.generated_sync}
      }
      %153 = arith.muli %49, %c2_i64 : i64
      %154 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %155 = cannir.extract_shape %154 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %156 = cannir.extract_dim %155[0] : !cannir.shape<(-1, -1)>
      %157 = arith.minsi %73, %156 : i64
      %158 = arith.subi %156, %157 : i64
      %159 = arith.minsi %158, %c1_i64 : i64
      %160 = arith.cmpi sle, %73, %156 : i64
      %161 = arith.select %160, %159, %c0_i64 : i64
      %162 = arith.cmpi sgt, %161, %c1_i64 : i64
      %163 = arith.select %162, %c1_i64, %161 : i64
      %164 = cannir.extract_dim %155[1] : !cannir.shape<(-1, -1)>
      %165 = arith.maxsi %153, %c1_i64 : i64
      %166 = arith.divsi %164, %165 : i64
      %167 = arith.minsi %84, %166 : i64
      %168 = arith.muli %167, %153 : i64
      %169 = arith.subi %164, %168 : i64
      %170 = arith.minsi %153, %169 : i64
      %171 = arith.cmpi sle, %84, %166 : i64
      %172 = arith.select %171, %170, %c0_i64 : i64
      %173 = arith.cmpi sgt, %172, %c1024_i64 : i64
      %174 = arith.select %173, %c1024_i64, %172 : i64
      %175 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %176 = cannir.extract_stride %175 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %177 = cannir.extract_dim %176[0] : !cannir.stride<(-1, 1)>
      %178 = arith.muli %73, %177 : i64
      %179 = arith.muli %84, %153 : i64
      %180 = arith.addi %178, %179 : i64
      %181 = arith.muli %180, %c8_i64 : i64
      %182 = ascscalar.memref_address %alloc_2 : memref<1xui8>
      %183 = arith.divui %181, %c8_i64 : i64
      %184 = arith.addi %182, %183 : i64
      %185 = ascscalar.bind_gm_ptr %184 : memref<1xui8>
      %186 = ascscalar.bind_local_ptr %107 : memref<1xui8, 1>
      %187 = cannir.extract_stride %154 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %188 = cannir.extract_dim %187[0] : !cannir.stride<(-1, 1)>
      %189 = arith.cmpi ugt, %163, %c0_i64 : i64
      %190 = arith.cmpi ugt, %174, %c0_i64 : i64
      %191 = arith.andi %189, %190 : i1
      scf.if %191 {
        ascvec.get_buf mte3, %109, 0 {cannir.buffer.generated_sync}
        %192 = arith.trunci %163 : i64 to i32
        %193 = arith.trunci %174 : i64 to i32
        ascvec.copy_ub2gm %185[%c0], %186[%c0], %192, %193, %188, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %109, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %90, %99 : i64, i64
    }
    return
  }
}
