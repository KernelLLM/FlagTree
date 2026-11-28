module {
  func.func @run_non_tail_graph_rng_00e3e424b68387816b123bdc8b641460(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: i64, %arg5: i64, %arg6: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg7: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg8: i64) attributes {cannbotdsl.contract_digest = "00e3e424b68387816b123bdc8b6414605c36bec40759578143a1838f3d756f98", cannbotdsl.public_slot_count = 9 : i64, cannbotdsl.slot_dtype_bits = [16, 16, 8, 8, 64, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 4, 1, 1, 0, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 2, 2, 1, 1, 2], cannbotdsl.slot_type_digests = ["d3381cb3edb8f40514b57c6c942d36e84a87723300273978fe7c3f381dbd4f9d", "d3381cb3edb8f40514b57c6c942d36e84a87723300273978fe7c3f381dbd4f9d", "e3a7bc6c4b8cb98d68cf0f892959846f4e8f11c27f2c9e430aec968ae281159e", "9332f9464639b7e579b015309e81477f71c4c6705898c0b7ab38269d65f6d71a", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "a9d876affafdf57be3f0e1a45cb0b293a48851dbda8b74101f5184f5130b1b63", "a9d876affafdf57be3f0e1a45cb0b293a48851dbda8b74101f5184f5130b1b63", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel_run_non_tail_0(%arg5) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg6, %arg7, %arg8] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 5, 6>, name = "_DynamicMxQuantSrKernel_run_non_tail", op_type = "_DynamicMxQuantSrKernel_run_non_tail", roles = array<i64: 0, 0, 1, 1, 0, 0>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, i64
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel_run_non_tail_0(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg4: i64, %arg5: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg7: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel_run_non_tail_0", cannir.resource.ids_prepared} {
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
    %alloc_3 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xi64>
    %alloc_4 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1xi64>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xbf16, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xbf16, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<16384xui8, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 81920 : i64} : memref<16384xui8, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 98304 : i64} : memref<1024xui8, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 99328 : i64} : memref<1024xui8, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 100352 : i64} : memref<512xui32, 1>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 102400 : i64} : memref<4096xi32, 1>
    %0 = ascscalar.scalar_ptr %alloc_3[%c0] : memref<1xi64>, index -> <gm, i64>
    %1 = ascscalar.load_dev %0 : !ascscalar.scalar_ptr<gm, i64> -> i64
    %2 = ascscalar.scalar_ptr %alloc_4[%c0] : memref<1xi64>, index -> <gm, i64>
    %3 = ascscalar.load_dev %2 : !ascscalar.scalar_ptr<gm, i64> -> i64
    %4 = arith.addi %3, %arg7 : i64
    %5 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
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
    %16 = arith.muli %15, %c2_i64 : i64
    %17 = arith.muli %16, %14 : i64
    %18 = ascscalar.get_block_num
    %19 = arith.addi %17, %18 : i64
    %20 = arith.subi %19, %c1_i64 : i64
    %21 = arith.remsi %20, %18 : i64
    %22 = arith.cmpi ne, %21, %c0_i64 : i64
    %23 = arith.cmpi slt, %21, %c0_i64 : i64
    %24 = arith.cmpi slt, %18, %c0_i64 : i64
    %25 = arith.xori %23, %24 : i1
    %26 = arith.andi %22, %25 : i1
    %27 = arith.divsi %20, %18 : i64
    %28 = arith.extui %26 : i1 to i64
    %29 = arith.subi %27, %28 : i64
    %30 = arith.addi %29, %c255_i64 : i64
    %31 = arith.remsi %30, %c256_i64 : i64
    %32 = arith.cmpi ne, %31, %c0_i64 : i64
    %33 = arith.cmpi slt, %31, %c0_i64 : i64
    %34 = arith.andi %32, %33 : i1
    %35 = arith.divsi %30, %c256_i64 : i64
    %36 = arith.extui %34 : i1 to i64
    %37 = arith.subi %35, %36 : i64
    %38 = arith.addi %29, %37 : i64
    %39 = arith.subi %38, %c1_i64 : i64
    %40 = arith.remsi %39, %37 : i64
    %41 = arith.cmpi ne, %40, %c0_i64 : i64
    %42 = arith.cmpi slt, %40, %c0_i64 : i64
    %43 = arith.cmpi slt, %37, %c0_i64 : i64
    %44 = arith.xori %42, %43 : i1
    %45 = arith.andi %41, %44 : i1
    %46 = arith.divsi %39, %37 : i64
    %47 = arith.extui %45 : i1 to i64
    %48 = arith.subi %46, %47 : i64
    %49 = arith.cmpi sge, %48, %c4_i64 : i64
    %50 = scf.if %49 -> (i64) {
      %66 = arith.addi %48, %c3_i64 : i64
      %67 = arith.remsi %66, %c4_i64 : i64
      %68 = arith.cmpi ne, %67, %c0_i64 : i64
      %69 = arith.cmpi slt, %67, %c0_i64 : i64
      %70 = arith.andi %68, %69 : i1
      %71 = arith.divsi %66, %c4_i64 : i64
      %72 = arith.extui %70 : i1 to i64
      %73 = arith.subi %71, %72 : i64
      %74 = arith.muli %73, %c4_i64 : i64
      scf.yield %74 : i64
    } else {
      scf.yield %48 : i64
    }
    %51 = arith.muli %50, %c64_i64 : i64
    %52 = arith.addi %14, %50 : i64
    %53 = arith.subi %52, %c1_i64 : i64
    %54 = arith.remsi %53, %50 : i64
    %55 = arith.cmpi ne, %54, %c0_i64 : i64
    %56 = arith.cmpi slt, %54, %c0_i64 : i64
    %57 = arith.cmpi slt, %50, %c0_i64 : i64
    %58 = arith.xori %56, %57 : i1
    %59 = arith.andi %55, %58 : i1
    %60 = arith.divsi %53, %50 : i64
    %61 = arith.extui %59 : i1 to i64
    %62 = arith.subi %60, %61 : i64
    %63 = arith.muli %16, %62 : i64
    %64 = ascscalar.get_block_idx
    %65:2 = scf.for %arg8 = %64 to %63 step %18 iter_args(%arg9 = %c0_i64, %arg10 = %c0_i64) -> (i64, i64)  : i64 {
      %66 = arith.remsi %arg8, %62 : i64
      %67 = arith.cmpi ne, %66, %c0_i64 : i64
      %68 = arith.cmpi slt, %66, %c0_i64 : i64
      %69 = arith.cmpi slt, %62, %c0_i64 : i64
      %70 = arith.xori %68, %69 : i1
      %71 = arith.andi %67, %70 : i1
      %72 = arith.divsi %arg8, %62 : i64
      %73 = arith.extui %71 : i1 to i64
      %74 = arith.subi %72, %73 : i64
      %75 = arith.remsi %74, %c2_i64 : i64
      %76 = arith.cmpi ne, %75, %c0_i64 : i64
      %77 = arith.cmpi slt, %75, %c0_i64 : i64
      %78 = arith.andi %76, %77 : i1
      %79 = arith.divsi %74, %c2_i64 : i64
      %80 = arith.extui %78 : i1 to i64
      %81 = arith.subi %79, %80 : i64
      %82 = arith.select %78, %c2_i64, %c0_i64 : i64
      %83 = arith.addi %75, %82 overflow<nsw> : i64
      %84 = arith.select %71, %62, %c0_i64 : i64
      %85 = arith.addi %66, %84 : i64
      %86 = arith.muli %85, %51 : i64
      %87 = arith.subi %7, %86 : i64
      %88 = arith.cmpi sgt, %87, %51 : i64
      %89 = arith.select %88, %51, %87 : i64
      %90 = arith.cmpi eq, %83, %c0_i64 : i64
      %91 = scf.if %90 -> (i64) {
        %193 = arith.cmpi eq, %arg9, %c0_i64 : i64
        %194 = ascscalar.memref_address %alloc_5 : memref<16384xbf16, 1>
        %195 = ascscalar.memref_address %alloc_6 : memref<16384xbf16, 1>
        %196 = arith.select %193, %194, %195 : i64
        %197 = arith.cmpi ne, %arg9, %c0_i64 : i64
        %198 = arith.extui %197 : i1 to i64
        %199 = arith.addi %arg9, %c1_i64 overflow<nsw> : i64
        %200 = arith.remui %199, %c2_i64 : i64
        %201 = arith.minsi %81, %15 : i64
        %202 = arith.subi %15, %201 : i64
        %203 = arith.minsi %202, %c1_i64 : i64
        %204 = arith.cmpi sle, %81, %15 : i64
        %205 = arith.select %204, %203, %c0_i64 : i64
        %206 = arith.maxsi %51, %c1_i64 : i64
        %207 = arith.divsi %7, %206 : i64
        %208 = arith.minsi %85, %207 : i64
        %209 = arith.muli %208, %51 : i64
        %210 = arith.subi %7, %209 : i64
        %211 = arith.minsi %51, %210 : i64
        %212 = arith.cmpi sle, %85, %207 : i64
        %213 = arith.select %212, %211, %c0_i64 : i64
        %214 = cannir.extract_stride %5 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
        %215 = cannir.extract_dim %214[0] : !cannir.stride<(-1, 2)>
        %216 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
        %217 = cannir.extract_stride %216 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
        %218 = cannir.extract_dim %217[0] : !cannir.stride<(-1, 2)>
        %219 = arith.muli %81, %218 : i64
        %220 = arith.muli %86, %c2_i64 : i64
        %221 = arith.addi %219, %220 : i64
        %222 = arith.muli %221, %c16_i64 : i64
        %223 = ascscalar.memref_address %alloc : memref<1xbf16>
        %224 = arith.divui %222, %c8_i64 : i64
        %225 = arith.addi %223, %224 : i64
        %226 = ascscalar.bind_gm_ptr %225 : memref<1xbf16>
        %227 = ascscalar.bind_local_ptr %196 : memref<1xbf16, 1>
        %228 = arith.cmpi ugt, %205, %c0_i64 : i64
        %229 = arith.cmpi ugt, %213, %c0_i64 : i64
        %230 = arith.andi %228, %229 : i1
        scf.if %230 {
          ascvec.get_buf mte2, %198, 0 {cannir.buffer.generated_sync}
          %231 = arith.trunci %213 : i64 to i32
          %232 = arith.trunci %205 : i64 to i32
          ascvec.copy_gm2ub_nddma %227[%c0], %226[%c0], %231, %232, %c1_i32, %c1_i32, %c1_i32, %c2_i64, %215, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c16384_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 : memref<1xbf16, 1>, memref<1xbf16>
          ascvec.release_buf mte2, %198, 0 {cannir.buffer.generated_sync}
        }
        scf.yield %200 : i64
      } else {
        %193 = arith.cmpi eq, %arg9, %c0_i64 : i64
        %194 = ascscalar.memref_address %alloc_5 : memref<16384xbf16, 1>
        %195 = ascscalar.memref_address %alloc_6 : memref<16384xbf16, 1>
        %196 = arith.select %193, %194, %195 : i64
        %197 = arith.cmpi ne, %arg9, %c0_i64 : i64
        %198 = arith.extui %197 : i1 to i64
        %199 = arith.addi %arg9, %c1_i64 overflow<nsw> : i64
        %200 = arith.remui %199, %c2_i64 : i64
        %201 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
        %202 = cannir.extract_shape %201 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, -1)>
        %203 = cannir.extract_dim %202[0] : !cannir.shape<(-1, -1)>
        %204 = arith.minsi %81, %203 : i64
        %205 = arith.subi %203, %204 : i64
        %206 = arith.minsi %205, %c1_i64 : i64
        %207 = arith.cmpi sle, %81, %203 : i64
        %208 = arith.select %207, %206, %c0_i64 : i64
        %209 = cannir.extract_dim %202[1] : !cannir.shape<(-1, -1)>
        %210 = arith.maxsi %51, %c1_i64 : i64
        %211 = arith.divsi %209, %210 : i64
        %212 = arith.minsi %85, %211 : i64
        %213 = arith.muli %212, %51 : i64
        %214 = arith.subi %209, %213 : i64
        %215 = arith.minsi %51, %214 : i64
        %216 = arith.cmpi sle, %85, %211 : i64
        %217 = arith.select %216, %215, %c0_i64 : i64
        %218 = cannir.extract_stride %201 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
        %219 = cannir.extract_dim %218[0] : !cannir.stride<(-1, 2)>
        %220 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 2)>>, <<(-1, -1)>, <(-1, 2)>>, identity>
        %221 = cannir.extract_stride %220 : <<(-1, -1)>, <(-1, 2)>> -> <(-1, 2)>
        %222 = cannir.extract_dim %221[0] : !cannir.stride<(-1, 2)>
        %223 = arith.muli %81, %222 : i64
        %224 = arith.muli %86, %c2_i64 : i64
        %225 = arith.addi %223, %224 : i64
        %226 = arith.muli %225, %c16_i64 : i64
        %227 = ascscalar.memref_address %alloc_0 : memref<1xbf16>
        %228 = arith.divui %226, %c8_i64 : i64
        %229 = arith.addi %227, %228 : i64
        %230 = ascscalar.bind_gm_ptr %229 : memref<1xbf16>
        %231 = ascscalar.bind_local_ptr %196 : memref<1xbf16, 1>
        %232 = arith.cmpi ugt, %208, %c0_i64 : i64
        %233 = arith.cmpi ugt, %217, %c0_i64 : i64
        %234 = arith.andi %232, %233 : i1
        scf.if %234 {
          ascvec.get_buf mte2, %198, 0 {cannir.buffer.generated_sync}
          %235 = arith.trunci %217 : i64 to i32
          %236 = arith.trunci %208 : i64 to i32
          ascvec.copy_gm2ub_nddma %231[%c0], %230[%c0], %235, %236, %c1_i32, %c1_i32, %c1_i32, %c2_i64, %219, %c0_i64, %c0_i64, %c0_i64, %c1_i64, %c16384_i64, %c0_i64, %c0_i64, %c0_i64 l2_cache_ctl %c0_i32 left_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) right_padding(%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32) pad_value %c0_i32 : i32 : memref<1xbf16, 1>, memref<1xbf16>
          ascvec.release_buf mte2, %198, 0 {cannir.buffer.generated_sync}
        }
        scf.yield %200 : i64
      }
      %92 = arith.cmpi eq, %arg10, %c0_i64 : i64
      %93 = ascscalar.memref_address %alloc_5 : memref<16384xbf16, 1>
      %94 = ascscalar.memref_address %alloc_6 : memref<16384xbf16, 1>
      %95 = arith.select %92, %93, %94 : i64
      %96 = ascscalar.bind_local_ptr %95 : memref<16384xbf16, 1>
      %97 = arith.cmpi ne, %arg10, %c0_i64 : i64
      %98 = arith.extui %97 : i1 to i64
      %99 = arith.addi %arg10, %c1_i64 overflow<nsw> : i64
      %100 = arith.remui %99, %c2_i64 : i64
      %101 = ascscalar.memref_address %alloc_7 : memref<16384xui8, 1>
      %102 = ascscalar.memref_address %alloc_8 : memref<16384xui8, 1>
      %103 = arith.select %92, %101, %102 : i64
      %104 = ascscalar.bind_local_ptr %103 : memref<16384xui8, 1>
      %105 = arith.select %92, %c2_i64, %c3_i64 : i64
      %106 = ascscalar.memref_address %alloc_9 : memref<1024xui8, 1>
      %107 = ascscalar.memref_address %alloc_10 : memref<1024xui8, 1>
      %108 = arith.select %92, %106, %107 : i64
      %109 = ascscalar.bind_local_ptr %108 : memref<1024xui8, 1>
      %110 = arith.select %92, %c4_i64, %c5_i64 : i64
      %111 = arith.muli %74, %7 : i64
      %112 = arith.addi %111, %86 : i64
      %113 = arith.cmpi sge, %89, %c256_i64 : i64
      %114 = arith.remsi %89, %c256_i64 : i64
      %115 = arith.cmpi eq, %114, %c0_i64 : i64
      %116 = arith.select %113, %115, %false : i1
      scf.if %116 {
        %193 = arith.divsi %89, %c256_i64 : i64
        %194 = arith.addi %89, %c1023_i64 : i64
        %195 = arith.remsi %194, %c1024_i64 : i64
        %196 = arith.cmpi ne, %195, %c0_i64 : i64
        %197 = arith.cmpi slt, %195, %c0_i64 : i64
        %198 = arith.andi %196, %197 : i1
        %199 = arith.divsi %194, %c1024_i64 : i64
        %200 = arith.extui %198 : i1 to i64
        %201 = arith.subi %199, %200 : i64
        %202 = arith.trunci %1 : i64 to i32
        %203 = arith.extui %202 : i32 to i64
        %204 = arith.subi %1, %203 : i64
        %205 = arith.remsi %204, %c4294967296_i64 : i64
        %206 = arith.cmpi ne, %205, %c0_i64 : i64
        %207 = arith.cmpi slt, %205, %c0_i64 : i64
        %208 = arith.andi %206, %207 : i1
        %209 = arith.divsi %204, %c4294967296_i64 : i64
        %210 = arith.extui %208 : i1 to i64
        %211 = arith.subi %209, %210 : i64
        %212 = arith.trunci %211 : i64 to i32
        %213 = arith.divui %112, %c16_i64 : i64
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
        %225 = arith.trunci %4 : i64 to i32
        %226 = arith.extui %225 : i32 to i64
        %227 = arith.subi %4, %226 : i64
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
          %249 = arith.trunci %242 : i64 to i32
          %250 = arith.extui %249 : i32 to i64
          %251 = arith.subi %242, %250 : i64
          %252 = arith.remsi %251, %c4294967296_i64 : i64
          %253 = arith.cmpi ne, %252, %c0_i64 : i64
          %254 = arith.cmpi slt, %252, %c0_i64 : i64
          %255 = arith.andi %253, %254 : i1
          %256 = arith.divsi %251, %c4294967296_i64 : i64
          %257 = arith.extui %255 : i1 to i64
          %258 = arith.subi %256, %257 : i64
          %259 = arith.trunci %258 : i64 to i32
          %260 = arith.addi %239, %249 : i32
          %261 = arith.addi %240, %259 : i32
          %262 = arith.cmpi ult, %260, %239 : i32
          %263 = arith.extui %262 : i1 to i32
          %264 = arith.addi %261, %263 : i32
          %265 = arith.extui %260 : i32 to i64
          %266 = arith.muli %265, %c3449720151_i64 : i64
          %267 = arith.trunci %266 : i64 to i32
          %268 = arith.extui %267 : i32 to i64
          %269 = arith.subi %266, %268 : i64
          %270 = arith.remsi %269, %c4294967296_i64 : i64
          %271 = arith.cmpi ne, %270, %c0_i64 : i64
          %272 = arith.cmpi slt, %270, %c0_i64 : i64
          %273 = arith.andi %271, %272 : i1
          %274 = arith.divsi %269, %c4294967296_i64 : i64
          %275 = arith.extui %273 : i1 to i64
          %276 = arith.subi %274, %275 : i64
          %277 = arith.trunci %276 : i64 to i32
          %278 = arith.xori %264, %212 : i32
          %279 = arith.extui %278 : i32 to i64
          %280 = arith.muli %279, %c3449720151_i64 : i64
          %281 = arith.trunci %280 : i64 to i32
          %282 = arith.extui %281 : i32 to i64
          %283 = arith.subi %280, %282 : i64
          %284 = arith.remsi %283, %c4294967296_i64 : i64
          %285 = arith.cmpi ne, %284, %c0_i64 : i64
          %286 = arith.cmpi slt, %284, %c0_i64 : i64
          %287 = arith.andi %285, %286 : i1
          %288 = arith.divsi %283, %c4294967296_i64 : i64
          %289 = arith.extui %287 : i1 to i64
          %290 = arith.subi %288, %289 : i64
          %291 = arith.trunci %290 : i64 to i32
          %292 = arith.addi %202, %c-1640531527_i32 : i32
          %293 = arith.addi %212, %c-1150833019_i32 : i32
          %294 = arith.xori %291, %267 : i32
          %295 = arith.xori %294, %292 : i32
          %296 = arith.extui %295 : i32 to i64
          %297 = arith.muli %296, %c3528531795_i64 : i64
          %298 = arith.trunci %297 : i64 to i32
          %299 = arith.extui %298 : i32 to i64
          %300 = arith.subi %297, %299 : i64
          %301 = arith.remsi %300, %c4294967296_i64 : i64
          %302 = arith.cmpi ne, %301, %c0_i64 : i64
          %303 = arith.cmpi slt, %301, %c0_i64 : i64
          %304 = arith.andi %302, %303 : i1
          %305 = arith.divsi %300, %c4294967296_i64 : i64
          %306 = arith.extui %304 : i1 to i64
          %307 = arith.subi %305, %306 : i64
          %308 = arith.trunci %307 : i64 to i32
          %309 = arith.xori %277, %202 : i32
          %310 = arith.addi %202, %c1013904242_i32 : i32
          %311 = arith.xori %281, %310 : i32
          %312 = arith.addi %212, %c1993301258_i32 : i32
          %313 = arith.xori %308, %312 : i32
          ascsimt.vf(%214, %309, %293, %311, %313, %202, %212, %298, %alloc_12, %241 : i32, i32, i32, i32, i32, i32, i32, i32, memref<4096xi32, 1>, i64) {
          ^bb0(%arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: memref<4096xi32, 1>, %arg20: i64):
            %c256_i64_13 = arith.constant 256 : i64
            %c3_i64_14 = arith.constant 3 : i64
            %c2_i64_15 = arith.constant 2 : i64
            %c1_i64_16 = arith.constant 1 : i64
            %c4_i64_17 = arith.constant 4 : i64
            %c1_i32_18 = arith.constant 1 : i32
            %c10_i32 = arith.constant 10 : i32
            %c3_i32 = arith.constant 3 : i32
            %c-1150833019_i32_19 = arith.constant -1150833019 : i32
            %c-1640531527_i32_20 = arith.constant -1640531527 : i32
            %c-845247145_i32 = arith.constant -845247145 : i32
            %c-766435501_i32 = arith.constant -766435501 : i32
            %c16384_i32_21 = arith.constant 16384 : i32
            %314 = ascsimt.thread_idx {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
            %315 = arith.extsi %314 {cannir.vf_threads = 256 : i64} : i32 to i64
            scf.for %arg21 = %315 to %arg20 step %c256_i64_13  : i64 {
              %316 = arith.trunci %arg21 : i64 to i32
              %317 = arith.addi %arg11, %316 : i32
              %318 = arith.remui %317, %c16384_i32_21 : i32
              %319 = arith.xori %318, %arg12 : i32
              %320 = arith.muli %319, %c-766435501_i32 : i32
              %321 = "ascsimt.bitcast"(%319) : (i32) -> ui32
              %322 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %323 = ascsimt.math_binary %321, %322, "mulhi" : ui32, ui32 -> ui32
              %324 = "ascsimt.bitcast"(%323) : (ui32) -> i32
              %325 = arith.xori %324, %arg13 : i32
              %326 = arith.muli %325, %c-845247145_i32 : i32
              %327 = "ascsimt.bitcast"(%325) : (i32) -> ui32
              %328 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
              %329 = ascsimt.math_binary %327, %328, "mulhi" : ui32, ui32 -> ui32
              %330 = "ascsimt.bitcast"(%329) : (ui32) -> i32
              %331 = arith.xori %330, %arg14 : i32
              %332 = arith.xori %320, %arg15 : i32
              %333:4 = scf.for %arg22 = %c3_i32 to %c10_i32 step %c1_i32_18 iter_args(%arg23 = %331, %arg24 = %326, %arg25 = %332, %arg26 = %arg18) -> (i32, i32, i32, i32)  : i32 {
                %342 = arith.muli %arg23, %c-766435501_i32 : i32
                %343 = "ascsimt.bitcast"(%arg23) : (i32) -> ui32
                %344 = ascsimt.math_binary %343, %322, "mulhi" : ui32, ui32 -> ui32
                %345 = "ascsimt.bitcast"(%344) : (ui32) -> i32
                %346 = arith.muli %arg25, %c-845247145_i32 : i32
                %347 = "ascsimt.bitcast"(%arg25) : (i32) -> ui32
                %348 = ascsimt.math_binary %347, %328, "mulhi" : ui32, ui32 -> ui32
                %349 = "ascsimt.bitcast"(%348) : (ui32) -> i32
                %350 = arith.muli %arg22, %c-1640531527_i32_20 : i32
                %351 = arith.addi %arg16, %350 : i32
                %352 = arith.muli %arg22, %c-1150833019_i32_19 : i32
                %353 = arith.addi %arg17, %352 : i32
                %354 = arith.xori %349, %arg24 : i32
                %355 = arith.xori %354, %351 : i32
                %356 = arith.xori %345, %arg26 : i32
                %357 = arith.xori %356, %353 : i32
                scf.yield %355, %346, %357, %342 : i32, i32, i32, i32
              }
              %334 = arith.muli %arg21, %c4_i64_17 : i64
              %335 = arith.index_cast %334 : i64 to index
              ascsimt.store %arg19[%335], %333#0 : memref<4096xi32, 1>, i32
              %336 = arith.addi %334, %c1_i64_16 : i64
              %337 = arith.index_cast %336 : i64 to index
              ascsimt.store %arg19[%337], %333#1 : memref<4096xi32, 1>, i32
              %338 = arith.addi %334, %c2_i64_15 : i64
              %339 = arith.index_cast %338 : i64 to index
              ascsimt.store %arg19[%339], %333#2 : memref<4096xi32, 1>, i32
              %340 = arith.addi %334, %c3_i64_14 : i64
              %341 = arith.index_cast %340 : i64 to index
              ascsimt.store %arg19[%341], %333#3 : memref<4096xi32, 1>, i32
            } {cannir.vf_threads = 256 : i64}
          } {threads = 256 : i64}
        } else {
          ascsimt.vf(%214, %224, %239, %240, %202, %212, %alloc_12, %241 : i32, i32, i32, i32, i32, i32, memref<4096xi32, 1>, i64) {
          ^bb0(%arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: memref<4096xi32, 1>, %arg18: i64):
            %c256_i64_13 = arith.constant 256 : i64
            %c3_i64_14 = arith.constant 3 : i64
            %c2_i64_15 = arith.constant 2 : i64
            %c1_i64_16 = arith.constant 1 : i64
            %c4_i64_17 = arith.constant 4 : i64
            %c1_i32_18 = arith.constant 1 : i32
            %c10_i32 = arith.constant 10 : i32
            %c0_i32_19 = arith.constant 0 : i32
            %c-1150833019_i32_20 = arith.constant -1150833019 : i32
            %c-1640531527_i32_21 = arith.constant -1640531527 : i32
            %c-845247145_i32 = arith.constant -845247145 : i32
            %c-766435501_i32 = arith.constant -766435501 : i32
            %c262144_i32 = arith.constant 262144 : i32
            %c16384_i32_22 = arith.constant 16384 : i32
            %249 = ascsimt.thread_idx {ascsimt.scope = 2 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
            %250 = arith.extsi %249 {cannir.vf_threads = 256 : i64} : i32 to i64
            scf.for %arg19 = %250 to %arg18 step %c256_i64_13  : i64 {
              %251 = arith.trunci %arg19 : i64 to i32
              %252 = arith.addi %arg11, %251 : i32
              %253 = arith.cmpi ult, %252, %arg11 : i32
              %254 = arith.extui %253 : i1 to i32
              %255 = arith.addi %arg12, %254 : i32
              %256 = arith.divui %252, %c16384_i32_22 : i32
              %257 = arith.remui %255, %c16384_i32_22 : i32
              %258 = arith.muli %257, %c262144_i32 : i32
              %259 = arith.addi %256, %258 : i32
              %260 = arith.divui %255, %c16384_i32_22 : i32
              %261 = arith.addi %arg13, %259 : i32
              %262 = arith.addi %arg14, %260 : i32
              %263 = arith.cmpi ult, %261, %arg13 : i32
              %264 = arith.extui %263 : i1 to i32
              %265 = arith.addi %262, %264 : i32
              %266 = arith.remui %252, %c16384_i32_22 : i32
              %267:4 = scf.for %arg20 = %c0_i32_19 to %c10_i32 step %c1_i32_18 iter_args(%arg21 = %c0_i32_19, %arg22 = %266, %arg23 = %261, %arg24 = %265) -> (i32, i32, i32, i32)  : i32 {
                %276 = arith.muli %arg21, %c-766435501_i32 : i32
                %277 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                %278 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                %279 = ascsimt.math_binary %277, %278, "mulhi" : ui32, ui32 -> ui32
                %280 = "ascsimt.bitcast"(%279) : (ui32) -> i32
                %281 = arith.muli %arg23, %c-845247145_i32 : i32
                %282 = "ascsimt.bitcast"(%arg23) : (i32) -> ui32
                %283 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                %284 = ascsimt.math_binary %282, %283, "mulhi" : ui32, ui32 -> ui32
                %285 = "ascsimt.bitcast"(%284) : (ui32) -> i32
                %286 = arith.muli %arg20, %c-1640531527_i32_21 : i32
                %287 = arith.addi %arg15, %286 : i32
                %288 = arith.muli %arg20, %c-1150833019_i32_20 : i32
                %289 = arith.addi %arg16, %288 : i32
                %290 = arith.xori %285, %arg22 : i32
                %291 = arith.xori %290, %287 : i32
                %292 = arith.xori %280, %arg24 : i32
                %293 = arith.xori %292, %289 : i32
                scf.yield %291, %281, %293, %276 : i32, i32, i32, i32
              }
              %268 = arith.muli %arg19, %c4_i64_17 : i64
              %269 = arith.index_cast %268 : i64 to index
              ascsimt.store %arg17[%269], %267#0 : memref<4096xi32, 1>, i32
              %270 = arith.addi %268, %c1_i64_16 : i64
              %271 = arith.index_cast %270 : i64 to index
              ascsimt.store %arg17[%271], %267#1 : memref<4096xi32, 1>, i32
              %272 = arith.addi %268, %c2_i64_15 : i64
              %273 = arith.index_cast %272 : i64 to index
              ascsimt.store %arg17[%273], %267#2 : memref<4096xi32, 1>, i32
              %274 = arith.addi %268, %c3_i64_14 : i64
              %275 = arith.index_cast %274 : i64 to index
              ascsimt.store %arg17[%275], %267#3 : memref<4096xi32, 1>, i32
            } {cannir.vf_threads = 256 : i64}
          } {threads = 256 : i64}
        }
        %247 = arith.trunci %193 : i64 to i32
        ascvec.get_buf v, %110, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %98, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
        %248 = ascscalar.bitcast_memref %96 byte_offset = %c0_i64 : memref<16384xbf16, 1> -> memref<?xui8, 1>
        ascvec.vf(%109, %96, %248, %alloc_12, %104, %247 : memref<1024xui8, 1>, memref<16384xbf16, 1>, memref<?xui8, 1>, memref<4096xi32, 1>, memref<16384xui8, 1>, i32) {
        ^bb0(%arg11: memref<1024xui8, 1>, %arg12: memref<16384xbf16, 1>, %arg13: memref<?xui8, 1>, %arg14: memref<4096xi32, 1>, %arg15: memref<16384xui8, 1>, %arg16: i32):
          %c512_i32 = arith.constant 512 : i32
          %c1_i32_13 = arith.constant 1 : i32
          %c2139062143_i32 = arith.constant 2139062143 : i32
          %c-8323200_i32 = arith.constant -8323200 : i32
          %c123_i8 = arith.constant 123 : i8
          %c-1_i8 = arith.constant -1 : i8
          %c60_i8 = arith.constant 60 : i8
          %cst = arith.constant 6.10351526E-5 : f32
          %c64_i32 = arith.constant 64 : i32
          %c0_i16 = arith.constant 0 : i16
          %c0_i32_14 = arith.constant 0 : i32
          %c254_i32 = arith.constant 254 : i32
          %c-15_i32 = arith.constant -15 : i32
          %c15_i32 = arith.constant 15 : i32
          %c255_i32 = arith.constant 255 : i32
          %c32767_i16 = arith.constant 32767 : i16
          %c256_i32 = arith.constant 256 : i32
          %c0_15 = arith.constant 0 : index
          %c1_i8 = arith.constant 1 : i8
          %c32_i8 = arith.constant 32 : i8
          %c65535_i32 = arith.constant 65535 : i32
          %c3_i8 = arith.constant 3 : i8
          %c0_i8 = arith.constant 0 : i8
          %c-128_i8 = arith.constant -128 : i8
          %249 = ascvec.create_mask : !ascvec.mask
          %250 = ascvec.create_mask elem_bits 16 : !ascvec.mask
          %251 = ascvec.create_mask elem_bits 8 : !ascvec.mask
          %252 = ascvec.duplicate_scalar %c-128_i8, %251 : i8 -> <ui8>
          %253 = ascvec.arange %c0_i8 : i8 -> <ui8>
          %254 = ascvec.duplicate_scalar %c3_i8, %251 : i8 -> <ui8>
          %255 = ascvec.and %253, %254, %251 : <ui8>, <ui8> -> <ui8>
          %256 = ascvec.shl %255, 6, %251 : <ui8> -> <ui8>
          %257 = ascvec.shr %253, 2, %251 : <ui8> -> <ui8>
          %258 = ascvec.or %256, %257, %251 : <ui8>, <ui8> -> <ui8>
          %259 = ascvec.duplicate_scalar %c65535_i32, %249 : i32 -> <ui32>
          %260 = ascvec.cmp_lt_scalar %253, %c32_i8, %251 : <ui8>, i8 -> !ascvec.mask
          %261 = ascvec.cmp_lt_scalar %255, %c1_i8, %251 : <ui8>, i8 -> !ascvec.mask
          %262 = ascvec.mask_and %260, %261, %251
          ascvec.store_unalign_begin %arg11 : memref<1024xui8, 1>
          scf.for %arg17 = %c0_i32_14 to %arg16 step %c1_i32_13  : i32 {
            %263 = arith.muli %arg17, %c256_i32 : i32
            %264 = arith.index_cast %263 : i32 to index
            %result0, %result1 = ascvec.loadalign_deintlv %arg12[%264] {dintlv_width = "b16"} : memref<16384xbf16, 1>, index -> <bf16>, <bf16>
            %265 = ascvec.duplicate_scalar %c32767_i16, %250 : i16 -> <ui16>
            %266 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
            %267 = ascvec.and %266, %265, %250 : <ui16>, <ui16> -> <ui16>
            %268 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
            %269 = ascvec.and %268, %265, %250 : <ui16>, <ui16> -> <ui16>
            %270 = ascvec.max %267, %269, %250 : <ui16>
            %271 = ascvec.reduce_max_datablock %270, %250 : <ui16>
            %272 = ascvec.unpack %271 {part = "lower"} : <ui16> -> <ui32>
            %273 = ascvec.shl %272, 16, %249 : <ui32> -> <ui32>
            %274 = arith.muli %arg17, %c512_i32 : i32
            %275 = arith.index_cast %274 : i32 to index
            %result0_17, %result1_18 = ascvec.loadalign_deintlv %arg13[%275] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
            %276 = ascvec.and %result1_18, %252, %251 : <ui8>, <ui8> -> <ui8>
            %277 = ascvec.reinterpret %276 : <ui8> -> <ui32>
            %278 = ascvec.shr %273, 23, %249 : <ui32> -> <ui32>
            %279 = ascvec.cmp_ge_scalar %278, %c255_i32, %249 : <ui32>, i32 -> !ascvec.mask
            %280 = ascvec.max_scalar %278, %c15_i32, %249 : <ui32>, i32
            %281 = ascvec.add_scalar %280, %c-15_i32, %249 : <ui32>, i32
            %282 = ascvec.duplicate_scalar %c255_i32, %279 merge %281 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %283 = ascvec.duplicate_scalar %c254_i32, %249 : i32 -> <ui32>
            %284 = ascvec.sub %283, %282, %249 : <ui32>
            %285 = ascvec.cmp_ne_scalar %282, %c0_i32_14, %249 : <ui32>, i32 -> !ascvec.mask
            %286 = ascvec.shl %284, 23, %285 : <ui32> -> <ui32>
            %287 = ascvec.reinterpret %286 : <ui32> -> <f32>
            %288 = ascvec.reinterpret %287 : <f32> -> <ui32>
            %289 = ascvec.shr %288, 16, %249 : <ui32> -> <ui32>
            %290 = ascvec.pack %289 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
            %291 = ascvec.arange %c0_i16 : i16 -> <ui16>
            %292 = ascvec.shr %291, 4, %250 : <ui16> -> <ui16>
            %293 = ascvec.gather_reg %290, %292 : <ui16>, <ui16> -> <ui16>
            %294 = ascvec.reinterpret %293 : <ui16> -> <bf16>
            %295 = ascvec.reinterpret %267 : <ui16> -> <bf16>
            %296 = ascvec.mul %295, %294, %250 : <bf16>
            %297 = ascvec.reinterpret %269 : <ui16> -> <bf16>
            %298 = ascvec.mul %297, %294, %250 : <bf16>
            %299 = ascvec.cast %296, %250 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
            %300 = ascvec.cast %296, %250 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
            %301 = ascvec.cast %298, %250 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
            %302 = ascvec.cast %298, %250 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
            %303 = ascvec.reinterpret %294 : <bf16> -> <ui32>
            %304 = arith.muli %arg17, %c64_i32 : i32
            %305 = arith.index_cast %304 : i32 to index
            %306 = ascvec.loadalign %arg14[%305] : memref<4096xi32, 1>, index -> <i32>
            %307 = ascvec.reinterpret %306 : <i32> -> <ui32>
            %308 = ascvec.shl %307, 8, %249 : <ui32> -> <ui32>
            %309 = ascvec.shr %307, 24, %249 : <ui32> -> <ui32>
            %310 = ascvec.or %308, %309, %249 : <ui32>, <ui32> -> <ui32>
            %311 = ascvec.shr %307, 16, %249 : <ui32> -> <ui32>
            %312 = ascvec.and %310, %259, %249 : <ui32>, <ui32> -> <ui32>
            %313 = ascvec.and %307, %259, %249 : <ui32>, <ui32> -> <ui32>
            %314 = ascvec.shr %310, 16, %249 : <ui32> -> <ui32>
            %315 = ascvec.reinterpret %299 : <f32> -> <ui32>
            %316 = ascvec.reinterpret %315 : <ui32> -> <f32>
            %317 = ascvec.max_scalar %316, %cst, %249 : <f32>, f32
            %318 = ascvec.add %316, %317, %249 : <f32>
            %319 = ascvec.reinterpret %318 : <f32> -> <ui32>
            %320 = ascvec.shr %319, 5, %249 : <ui32> -> <ui32>
            %321 = ascvec.add %320, %311, %249 : <ui32>
            %322 = ascvec.reinterpret %301 : <f32> -> <ui32>
            %323 = ascvec.reinterpret %322 : <ui32> -> <f32>
            %324 = ascvec.max_scalar %323, %cst, %249 : <f32>, f32
            %325 = ascvec.add %323, %324, %249 : <f32>
            %326 = ascvec.reinterpret %325 : <f32> -> <ui32>
            %327 = ascvec.shr %326, 5, %249 : <ui32> -> <ui32>
            %328 = ascvec.add %327, %312, %249 : <ui32>
            %329 = ascvec.reinterpret %300 : <f32> -> <ui32>
            %330 = ascvec.reinterpret %329 : <ui32> -> <f32>
            %331 = ascvec.max_scalar %330, %cst, %249 : <f32>, f32
            %332 = ascvec.add %330, %331, %249 : <f32>
            %333 = ascvec.reinterpret %332 : <f32> -> <ui32>
            %334 = ascvec.shr %333, 5, %249 : <ui32> -> <ui32>
            %335 = ascvec.add %334, %313, %249 : <ui32>
            %336 = ascvec.reinterpret %302 : <f32> -> <ui32>
            %337 = ascvec.reinterpret %336 : <ui32> -> <f32>
            %338 = ascvec.max_scalar %337, %cst, %249 : <f32>, f32
            %339 = ascvec.add %337, %338, %249 : <f32>
            %340 = ascvec.reinterpret %339 : <f32> -> <ui32>
            %341 = ascvec.shr %340, 5, %249 : <ui32> -> <ui32>
            %342 = ascvec.add %341, %314, %249 : <ui32>
            %343 = ascvec.reinterpret %321 : <ui32> -> <ui16>
            %344 = ascvec.reinterpret %328 : <ui32> -> <ui16>
            %result0_19, %result1_20 = ascvec.deintlv %343, %344 : <ui16>, <ui16> -> <ui16>, <ui16>
            %345 = ascvec.reinterpret %335 : <ui32> -> <ui16>
            %346 = ascvec.reinterpret %342 : <ui32> -> <ui16>
            %result0_21, %result1_22 = ascvec.deintlv %345, %346 : <ui16>, <ui16> -> <ui16>, <ui16>
            %347 = ascvec.reinterpret %result1_20 : <ui16> -> <ui8>
            %348 = ascvec.reinterpret %result1_22 : <ui16> -> <ui8>
            %result0_23, %result1_24 = ascvec.deintlv %347, %348 : <ui8>, <ui8> -> <ui8>, <ui8>
            %349 = ascvec.gather_reg %result0_23, %258 : <ui8>, <ui8> -> <ui8>
            %350 = ascvec.reinterpret %349 : <ui8> -> <ui32>
            %351 = ascvec.reinterpret %350 : <ui32> -> <ui8>
            %352 = ascvec.add_scalar %351, %c60_i8, %251 : <ui8>, i8
            %353 = ascvec.cmp_ne_scalar %352, %c-1_i8, %251 : <ui8>, i8 -> !ascvec.mask
            %354 = ascvec.min_scalar %352, %c123_i8, %353 : <ui8>, i8
            %355 = ascvec.reinterpret %354 : <ui8> -> <ui32>
            %356 = ascvec.or %355, %277, %249 : <ui32>, <ui32> -> <ui32>
            %357 = ascvec.reinterpret %303 : <ui32> -> <ui32>
            %358 = ascvec.cmp_eq_scalar %357, %c-8323200_i32, %249 : <ui32>, i32 -> !ascvec.mask
            %359 = ascvec.duplicate_scalar %c2139062143_i32, %358 merge %356 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %360 = ascvec.reinterpret %359 : <ui32> -> <ui8>
            ascvec.storealign %arg15[%264], %360, %251 : memref<16384xui8, 1>, index, <ui8>
            %361 = ascvec.reinterpret %282 : <ui32> -> <ui8>
            %362 = ascvec.squeeze_store_reg %361, %262 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg11[%c0_15], %362 : memref<1024xui8, 1>, <ui8>
          }
          %c0_16 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg17 = %c0_16 to %c1 step %c1 {
            ascvec.store_unalign_post %arg11[%c0_15] : memref<1024xui8, 1>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %98, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %110, 0 {cannir.buffer.generated_sync}
      } else {
        %193 = arith.addi %89, %c63_i64 : i64
        %194 = arith.remsi %193, %c64_i64 : i64
        %195 = arith.cmpi ne, %194, %c0_i64 : i64
        %196 = arith.cmpi slt, %194, %c0_i64 : i64
        %197 = arith.andi %195, %196 : i1
        %198 = arith.divsi %193, %c64_i64 : i64
        %199 = arith.extui %197 : i1 to i64
        %200 = arith.subi %198, %199 : i64
        %201 = arith.addi %89, %c1023_i64 : i64
        %202 = arith.remsi %201, %c1024_i64 : i64
        %203 = arith.cmpi ne, %202, %c0_i64 : i64
        %204 = arith.cmpi slt, %202, %c0_i64 : i64
        %205 = arith.andi %203, %204 : i1
        %206 = arith.divsi %201, %c1024_i64 : i64
        %207 = arith.extui %205 : i1 to i64
        %208 = arith.subi %206, %207 : i64
        %209 = arith.trunci %1 : i64 to i32
        %210 = arith.extui %209 : i32 to i64
        %211 = arith.subi %1, %210 : i64
        %212 = arith.remsi %211, %c4294967296_i64 : i64
        %213 = arith.cmpi ne, %212, %c0_i64 : i64
        %214 = arith.cmpi slt, %212, %c0_i64 : i64
        %215 = arith.andi %213, %214 : i1
        %216 = arith.divsi %211, %c4294967296_i64 : i64
        %217 = arith.extui %215 : i1 to i64
        %218 = arith.subi %216, %217 : i64
        %219 = arith.trunci %218 : i64 to i32
        %220 = arith.remsi %112, %c16_i64 : i64
        %221 = arith.cmpi ne, %220, %c0_i64 : i64
        %222 = arith.cmpi slt, %220, %c0_i64 : i64
        %223 = arith.andi %221, %222 : i1
        %224 = arith.divsi %112, %c16_i64 : i64
        %225 = arith.extui %223 : i1 to i64
        %226 = arith.subi %224, %225 : i64
        %227 = arith.remsi %arg4, %c16_i64 : i64
        %228 = arith.cmpi ne, %227, %c0_i64 : i64
        %229 = arith.cmpi slt, %227, %c0_i64 : i64
        %230 = arith.andi %228, %229 : i1
        %231 = arith.divsi %arg4, %c16_i64 : i64
        %232 = arith.extui %230 : i1 to i64
        %233 = arith.subi %231, %232 : i64
        %234 = arith.remsi %226, %233 : i64
        %235 = arith.cmpi ne, %234, %c0_i64 : i64
        %236 = arith.cmpi slt, %234, %c0_i64 : i64
        %237 = arith.cmpi slt, %233, %c0_i64 : i64
        %238 = arith.xori %236, %237 : i1
        %239 = arith.andi %235, %238 : i1
        %240 = arith.divsi %226, %233 : i64
        %241 = arith.extui %239 : i1 to i64
        %242 = arith.subi %240, %241 : i64
        %243 = arith.select %239, %233, %c0_i64 : i64
        %244 = arith.addi %234, %243 : i64
        %245 = arith.trunci %242 : i64 to i32
        %246 = arith.extui %245 : i32 to i64
        %247 = arith.subi %242, %246 : i64
        %248 = arith.remsi %247, %c4294967296_i64 : i64
        %249 = arith.cmpi ne, %248, %c0_i64 : i64
        %250 = arith.cmpi slt, %248, %c0_i64 : i64
        %251 = arith.andi %249, %250 : i1
        %252 = arith.divsi %247, %c4294967296_i64 : i64
        %253 = arith.extui %251 : i1 to i64
        %254 = arith.subi %252, %253 : i64
        %255 = arith.trunci %254 : i64 to i32
        %256 = arith.trunci %244 : i64 to i32
        %257 = arith.extui %256 : i32 to i64
        %258 = arith.subi %244, %257 : i64
        %259 = arith.remsi %258, %c4294967296_i64 : i64
        %260 = arith.cmpi ne, %259, %c0_i64 : i64
        %261 = arith.cmpi slt, %259, %c0_i64 : i64
        %262 = arith.andi %260, %261 : i1
        %263 = arith.divsi %258, %c4294967296_i64 : i64
        %264 = arith.extui %262 : i1 to i64
        %265 = arith.subi %263, %264 : i64
        %266 = arith.trunci %265 : i64 to i32
        %267 = arith.remsi %c64_i64, %233 : i64
        %268 = arith.cmpi ne, %267, %c0_i64 : i64
        %269 = arith.cmpi slt, %267, %c0_i64 : i64
        %270 = arith.xori %269, %237 : i1
        %271 = arith.andi %268, %270 : i1
        %272 = arith.divsi %c64_i64, %233 : i64
        %273 = arith.extui %271 : i1 to i64
        %274 = arith.subi %272, %273 : i64
        %275 = arith.select %271, %233, %c0_i64 : i64
        %276 = arith.addi %267, %275 : i64
        %277 = arith.trunci %208 : i64 to i32
        ascvec.vf(%233, %4, %209, %219, %alloc_12, %276, %274, %277, %245, %255, %256, %266 : i64, i64, i32, i32, memref<4096xi32, 1>, i64, i64, i32, i32, i32, i32, i32) {
        ^bb0(%arg11: i64, %arg12: i64, %arg13: i32, %arg14: i32, %arg15: memref<4096xi32, 1>, %arg16: i64, %arg17: i64, %arg18: i32, %arg19: i32, %arg20: i32, %arg21: i32, %arg22: i32):
          %c1_i32_13 = arith.constant 1 : i32
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
          %c1993301258_i32_14 = arith.constant 1993301258 : i32
          %c1013904242_i32_15 = arith.constant 1013904242 : i32
          %c-1150833019_i32_16 = arith.constant -1150833019 : i32
          %c-1640531527_i32_17 = arith.constant -1640531527 : i32
          %c-845247145_i32 = arith.constant -845247145 : i32
          %c-766435501_i32 = arith.constant -766435501 : i32
          %c16383_i32 = arith.constant 16383 : i32
          %c1073741824_i32_18 = arith.constant 1073741824 : i32
          %c4_i32_19 = arith.constant 4 : i32
          %c0_i64_20 = arith.constant 0 : i64
          %c0_i32_21 = arith.constant 0 : i32
          %c256_i32 = arith.constant 256 : i32
          %c4294967296_i64_22 = arith.constant 4294967296 : i64
          %280:4 = scf.for %arg23 = %c0_i32_21 to %arg18 step %c1_i32_13 iter_args(%arg24 = %arg19, %arg25 = %arg20, %arg26 = %arg21, %arg27 = %arg22) -> (i32, i32, i32, i32)  : i32 {
            %281 = arith.extui %arg24 : i32 to i64
            %282 = arith.extui %arg25 : i32 to i64
            %283 = arith.muli %282, %c4294967296_i64_22 : i64
            %284 = arith.addi %281, %283 : i64
            %285 = arith.extui %arg26 : i32 to i64
            %286 = arith.extui %arg27 : i32 to i64
            %287 = arith.muli %286, %c4294967296_i64_22 : i64
            %288 = arith.addi %285, %287 : i64
            %289 = arith.muli %arg23, %c256_i32 : i32
            %290 = ascvec.create_mask : !ascvec.mask
            %291 = ascvec.arange %c0_i32_21 : i32 -> <ui32>
            %292 = arith.muli %284, %arg11 : i64
            %293 = arith.addi %292, %288 : i64
            %294 = arith.trunci %293 : i64 to i32
            %295 = arith.extui %294 : i32 to i64
            %296 = arith.subi %293, %295 : i64
            %297 = arith.remsi %296, %c4294967296_i64_22 : i64
            %298 = arith.cmpi ne, %297, %c0_i64_20 : i64
            %299 = arith.cmpi slt, %297, %c0_i64_20 : i64
            %300 = arith.andi %298, %299 : i1
            %301 = arith.divsi %296, %c4294967296_i64_22 : i64
            %302 = arith.extui %300 : i1 to i64
            %303 = arith.subi %301, %302 : i64
            %304 = arith.trunci %303 : i64 to i32
            %305 = ascvec.duplicate_scalar %294, %290 : i32 -> <ui32>
            %306 = ascvec.duplicate_scalar %304, %290 : i32 -> <ui32>
            %307 = ascvec.duplicate_scalar %c0_i32_21, %290 : i32 -> <ui32>
            %carry, %dst = ascvec.addc %305, %291, %290 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %carry_23, %dst_24 = ascvec.addc %306, %307, %carry, %290 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %308 = arith.trunci %arg12 : i64 to i32
            %309 = arith.extui %308 : i32 to i64
            %310 = arith.subi %arg12, %309 : i64
            %311 = arith.remsi %310, %c4294967296_i64_22 : i64
            %312 = arith.cmpi ne, %311, %c0_i64_20 : i64
            %313 = arith.cmpi slt, %311, %c0_i64_20 : i64
            %314 = arith.andi %312, %313 : i1
            %315 = arith.divsi %310, %c4294967296_i64_22 : i64
            %316 = arith.extui %314 : i1 to i64
            %317 = arith.subi %315, %316 : i64
            %318 = arith.trunci %317 : i64 to i32
            %319 = arith.divui %308, %c4_i32_19 : i32
            %320 = arith.remui %318, %c4_i32_19 : i32
            %321 = arith.muli %320, %c1073741824_i32_18 : i32
            %322 = arith.addi %319, %321 : i32
            %323 = arith.divui %318, %c4_i32_19 : i32
            %324 = ascvec.shr %dst, 14, %290 : <ui32> -> <ui32>
            %325 = ascvec.shl %dst_24, 18, %290 : <ui32> -> <ui32>
            %326 = ascvec.or %324, %325, %290 : <ui32>, <ui32> -> <ui32>
            %327 = ascvec.shr %dst_24, 14, %290 : <ui32> -> <ui32>
            %328 = ascvec.duplicate_scalar %322, %290 : i32 -> <ui32>
            %329 = ascvec.duplicate_scalar %323, %290 : i32 -> <ui32>
            %carry_25, %dst_26 = ascvec.addc %328, %326, %290 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %carry_27, %dst_28 = ascvec.addc %329, %327, %carry_25, %290 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
            %330 = ascvec.duplicate_scalar %c16383_i32, %290 : i32 -> <ui32>
            %331 = ascvec.and %dst, %330, %290 : <ui32>, <ui32> -> <ui32>
            %332 = ascvec.duplicate_scalar %c-766435501_i32, %290 : i32 -> <ui32>
            %333 = ascvec.duplicate_scalar %c-845247145_i32, %290 : i32 -> <ui32>
            %dst0, %dst1 = ascvec.mull %307, %332, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_29, %dst1_30 = ascvec.mull %dst_26, %333, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %334 = ascvec.xor %dst1_30, %331, %290 : <ui32>, <ui32> -> <ui32>
            %335 = ascvec.xor %dst1, %dst_28, %290 : <ui32>, <ui32> -> <ui32>
            %336 = ascvec.duplicate_scalar %arg13, %290 : i32 -> <ui32>
            %337 = ascvec.duplicate_scalar %arg14, %290 : i32 -> <ui32>
            %338 = ascvec.xor %334, %336, %290 : <ui32>, <ui32> -> <ui32>
            %339 = ascvec.xor %335, %337, %290 : <ui32>, <ui32> -> <ui32>
            %dst0_31, %dst1_32 = ascvec.mull %338, %332, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_33, %dst1_34 = ascvec.mull %339, %333, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %340 = ascvec.xor %dst1_34, %dst0_29, %290 : <ui32>, <ui32> -> <ui32>
            %341 = ascvec.xor %dst1_32, %dst0, %290 : <ui32>, <ui32> -> <ui32>
            %342 = arith.addi %arg13, %c-1640531527_i32_17 : i32
            %343 = ascvec.duplicate_scalar %342, %290 : i32 -> <ui32>
            %344 = arith.addi %arg14, %c-1150833019_i32_16 : i32
            %345 = ascvec.duplicate_scalar %344, %290 : i32 -> <ui32>
            %346 = ascvec.xor %340, %343, %290 : <ui32>, <ui32> -> <ui32>
            %347 = ascvec.xor %341, %345, %290 : <ui32>, <ui32> -> <ui32>
            %dst0_35, %dst1_36 = ascvec.mull %346, %332, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_37, %dst1_38 = ascvec.mull %347, %333, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %348 = ascvec.xor %dst1_38, %dst0_33, %290 : <ui32>, <ui32> -> <ui32>
            %349 = ascvec.xor %dst1_36, %dst0_31, %290 : <ui32>, <ui32> -> <ui32>
            %350 = arith.addi %arg13, %c1013904242_i32_15 : i32
            %351 = ascvec.duplicate_scalar %350, %290 : i32 -> <ui32>
            %352 = arith.addi %arg14, %c1993301258_i32_14 : i32
            %353 = ascvec.duplicate_scalar %352, %290 : i32 -> <ui32>
            %354 = ascvec.xor %348, %351, %290 : <ui32>, <ui32> -> <ui32>
            %355 = ascvec.xor %349, %353, %290 : <ui32>, <ui32> -> <ui32>
            %dst0_39, %dst1_40 = ascvec.mull %354, %332, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_41, %dst1_42 = ascvec.mull %355, %333, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %356 = ascvec.xor %dst1_42, %dst0_37, %290 : <ui32>, <ui32> -> <ui32>
            %357 = ascvec.xor %dst1_40, %dst0_35, %290 : <ui32>, <ui32> -> <ui32>
            %358 = arith.addi %arg13, %c-626627285_i32 : i32
            %359 = ascvec.duplicate_scalar %358, %290 : i32 -> <ui32>
            %360 = arith.addi %arg14, %c842468239_i32 : i32
            %361 = ascvec.duplicate_scalar %360, %290 : i32 -> <ui32>
            %362 = ascvec.xor %356, %359, %290 : <ui32>, <ui32> -> <ui32>
            %363 = ascvec.xor %357, %361, %290 : <ui32>, <ui32> -> <ui32>
            %dst0_43, %dst1_44 = ascvec.mull %362, %332, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_45, %dst1_46 = ascvec.mull %363, %333, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %364 = ascvec.xor %dst1_46, %dst0_41, %290 : <ui32>, <ui32> -> <ui32>
            %365 = ascvec.xor %dst1_44, %dst0_39, %290 : <ui32>, <ui32> -> <ui32>
            %366 = arith.addi %arg13, %c2027808484_i32 : i32
            %367 = ascvec.duplicate_scalar %366, %290 : i32 -> <ui32>
            %368 = arith.addi %arg14, %c-308364780_i32 : i32
            %369 = ascvec.duplicate_scalar %368, %290 : i32 -> <ui32>
            %370 = ascvec.xor %364, %367, %290 : <ui32>, <ui32> -> <ui32>
            %371 = ascvec.xor %365, %369, %290 : <ui32>, <ui32> -> <ui32>
            %dst0_47, %dst1_48 = ascvec.mull %370, %332, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_49, %dst1_50 = ascvec.mull %371, %333, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %372 = ascvec.xor %dst1_50, %dst0_45, %290 : <ui32>, <ui32> -> <ui32>
            %373 = ascvec.xor %dst1_48, %dst0_43, %290 : <ui32>, <ui32> -> <ui32>
            %374 = arith.addi %arg13, %c387276957_i32 : i32
            %375 = ascvec.duplicate_scalar %374, %290 : i32 -> <ui32>
            %376 = arith.addi %arg14, %c-1459197799_i32 : i32
            %377 = ascvec.duplicate_scalar %376, %290 : i32 -> <ui32>
            %378 = ascvec.xor %372, %375, %290 : <ui32>, <ui32> -> <ui32>
            %379 = ascvec.xor %373, %377, %290 : <ui32>, <ui32> -> <ui32>
            %dst0_51, %dst1_52 = ascvec.mull %378, %332, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_53, %dst1_54 = ascvec.mull %379, %333, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %380 = ascvec.xor %dst1_54, %dst0_49, %290 : <ui32>, <ui32> -> <ui32>
            %381 = ascvec.xor %dst1_52, %dst0_47, %290 : <ui32>, <ui32> -> <ui32>
            %382 = arith.addi %arg13, %c-1253254570_i32 : i32
            %383 = ascvec.duplicate_scalar %382, %290 : i32 -> <ui32>
            %384 = arith.addi %arg14, %c1684936478_i32 : i32
            %385 = ascvec.duplicate_scalar %384, %290 : i32 -> <ui32>
            %386 = ascvec.xor %380, %383, %290 : <ui32>, <ui32> -> <ui32>
            %387 = ascvec.xor %381, %385, %290 : <ui32>, <ui32> -> <ui32>
            %dst0_55, %dst1_56 = ascvec.mull %386, %332, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_57, %dst1_58 = ascvec.mull %387, %333, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %388 = ascvec.xor %dst1_58, %dst0_53, %290 : <ui32>, <ui32> -> <ui32>
            %389 = ascvec.xor %dst1_56, %dst0_51, %290 : <ui32>, <ui32> -> <ui32>
            %390 = arith.addi %arg13, %c1401181199_i32 : i32
            %391 = ascvec.duplicate_scalar %390, %290 : i32 -> <ui32>
            %392 = arith.addi %arg14, %c534103459_i32 : i32
            %393 = ascvec.duplicate_scalar %392, %290 : i32 -> <ui32>
            %394 = ascvec.xor %388, %391, %290 : <ui32>, <ui32> -> <ui32>
            %395 = ascvec.xor %389, %393, %290 : <ui32>, <ui32> -> <ui32>
            %dst0_59, %dst1_60 = ascvec.mull %394, %332, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_61, %dst1_62 = ascvec.mull %395, %333, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %396 = ascvec.xor %dst1_62, %dst0_57, %290 : <ui32>, <ui32> -> <ui32>
            %397 = ascvec.xor %dst1_60, %dst0_55, %290 : <ui32>, <ui32> -> <ui32>
            %398 = arith.addi %arg13, %c-239350328_i32 : i32
            %399 = ascvec.duplicate_scalar %398, %290 : i32 -> <ui32>
            %400 = arith.addi %arg14, %c-616729560_i32 : i32
            %401 = ascvec.duplicate_scalar %400, %290 : i32 -> <ui32>
            %402 = ascvec.xor %396, %399, %290 : <ui32>, <ui32> -> <ui32>
            %403 = ascvec.xor %397, %401, %290 : <ui32>, <ui32> -> <ui32>
            %dst0_63, %dst1_64 = ascvec.mull %402, %332, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %dst0_65, %dst1_66 = ascvec.mull %403, %333, %290 : <ui32>, <ui32> -> <ui32>, <ui32>
            %404 = ascvec.xor %dst1_66, %dst0_61, %290 : <ui32>, <ui32> -> <ui32>
            %405 = ascvec.xor %dst1_64, %dst0_59, %290 : <ui32>, <ui32> -> <ui32>
            %406 = arith.addi %arg13, %c-1879881855_i32 : i32
            %407 = ascvec.duplicate_scalar %406, %290 : i32 -> <ui32>
            %408 = arith.addi %arg14, %c-1767562579_i32 : i32
            %409 = ascvec.duplicate_scalar %408, %290 : i32 -> <ui32>
            %410 = ascvec.xor %404, %407, %290 : <ui32>, <ui32> -> <ui32>
            %411 = ascvec.xor %405, %409, %290 : <ui32>, <ui32> -> <ui32>
            %412 = ascvec.reinterpret %410 : <ui32> -> <i32>
            %413 = arith.index_cast %289 : i32 to index
            ascvec.storealign %arg15[%413], %412, %290 : memref<4096xi32, 1>, index, <i32>
            %414 = arith.addi %289, %c64_i32 : i32
            %415 = ascvec.reinterpret %dst0_65 : <ui32> -> <i32>
            %416 = arith.index_cast %414 : i32 to index
            ascvec.storealign %arg15[%416], %415, %290 : memref<4096xi32, 1>, index, <i32>
            %417 = arith.addi %289, %c128_i32 : i32
            %418 = ascvec.reinterpret %411 : <ui32> -> <i32>
            %419 = arith.index_cast %417 : i32 to index
            ascvec.storealign %arg15[%419], %418, %290 : memref<4096xi32, 1>, index, <i32>
            %420 = arith.addi %289, %c192_i32 : i32
            %421 = ascvec.reinterpret %dst0_63 : <ui32> -> <i32>
            %422 = arith.index_cast %420 : i32 to index
            ascvec.storealign %arg15[%422], %421, %290 : memref<4096xi32, 1>, index, <i32>
            %423 = arith.addi %288, %arg16 : i64
            %424 = arith.cmpi sge, %423, %arg11 : i64
            %425 = arith.extui %424 : i1 to i64
            %426 = arith.addi %284, %arg17 : i64
            %427 = arith.addi %426, %425 : i64
            %428 = arith.muli %425, %arg11 : i64
            %429 = arith.subi %423, %428 : i64
            %430 = arith.trunci %427 : i64 to i32
            %431 = arith.extui %430 : i32 to i64
            %432 = arith.subi %427, %431 : i64
            %433 = arith.remsi %432, %c4294967296_i64_22 : i64
            %434 = arith.cmpi ne, %433, %c0_i64_20 : i64
            %435 = arith.cmpi slt, %433, %c0_i64_20 : i64
            %436 = arith.andi %434, %435 : i1
            %437 = arith.divsi %432, %c4294967296_i64_22 : i64
            %438 = arith.extui %436 : i1 to i64
            %439 = arith.subi %437, %438 : i64
            %440 = arith.trunci %439 : i64 to i32
            %441 = arith.trunci %429 : i64 to i32
            %442 = arith.extui %441 : i32 to i64
            %443 = arith.subi %429, %442 : i64
            %444 = arith.remsi %443, %c4294967296_i64_22 : i64
            %445 = arith.cmpi ne, %444, %c0_i64_20 : i64
            %446 = arith.cmpi slt, %444, %c0_i64_20 : i64
            %447 = arith.andi %445, %446 : i1
            %448 = arith.divsi %443, %c4294967296_i64_22 : i64
            %449 = arith.extui %447 : i1 to i64
            %450 = arith.subi %448, %449 : i64
            %451 = arith.trunci %450 : i64 to i32
            scf.yield %430, %440, %441, %451 : i32, i32, i32, i32
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        %278 = arith.trunci %200 : i64 to i32
        %279 = arith.muli %278, %c2_i32 : i32
        ascvec.get_buf v, %98, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %110, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%89, %96, %alloc_12, %104, %alloc_11, %278, %279, %109 : i64, memref<16384xbf16, 1>, memref<4096xi32, 1>, memref<16384xui8, 1>, memref<512xui32, 1>, i32, i32, memref<1024xui8, 1>) {
        ^bb0(%arg11: i64, %arg12: memref<16384xbf16, 1>, %arg13: memref<4096xi32, 1>, %arg14: memref<16384xui8, 1>, %arg15: memref<512xui32, 1>, %arg16: i32, %arg17: i32, %arg18: memref<1024xui8, 1>):
          %c2_i32_13 = arith.constant 2 : i32
          %c2139095040_i32 = arith.constant 2139095040 : i32
          %c127_i32 = arith.constant 127 : i32
          %c123_i32 = arith.constant 123 : i32
          %c947912704_i32 = arith.constant 947912704 : i32
          %cst = arith.constant 4.2949673E+9 : f32
          %c29360128_i32 = arith.constant 29360128 : i32
          %c128_i32 = arith.constant 128 : i32
          %c65535_i32 = arith.constant 65535 : i32
          %c4_i32_14 = arith.constant 4 : i32
          %c256_i32 = arith.constant 256 : i32
          %c1_i32_15 = arith.constant 1 : i32
          %c16_i32 = arith.constant 16 : i32
          %c3_i32 = arith.constant 3 : i32
          %c2139160576_i32 = arith.constant 2139160576 : i32
          %c254_i32 = arith.constant 254 : i32
          %c-15_i32 = arith.constant -15 : i32
          %c15_i32 = arith.constant 15 : i32
          %c255_i32 = arith.constant 255 : i32
          %c2147483647_i32 = arith.constant 2147483647 : i32
          %cst_16 = arith.constant 0.000000e+00 : f32
          %c0_17 = arith.constant 0 : index
          %c64_i32 = arith.constant 64 : i32
          %c32_i32 = arith.constant 32 : i32
          %c0_i32_18 = arith.constant 0 : i32
          %280 = ascvec.create_mask : !ascvec.mask
          %281 = ascvec.arange %c0_i32_18 : i32 -> <ui32>
          %282 = ascvec.cmp_lt_scalar %281, %c32_i32, %280 : <ui32>, i32 -> !ascvec.mask
          %283 = ascvec.cmp_ge_scalar %281, %c32_i32, %280 : <ui32>, i32 -> !ascvec.mask
          scf.for %arg19 = %c0_i32_18 to %arg16 step %c1_i32_15  : i32 {
            %284 = arith.muli %arg19, %c64_i32 : i32
            %285 = arith.trunci %arg11 : i64 to i32
            %286 = arith.subi %285, %284 : i32
            %mask, %remaining_out = ascvec.update_mask %286 : !ascvec.mask
            %287 = arith.index_cast %284 : i32 to index
            %288 = ascvec.loadalign_unpack %arg12[%287] {mode = "b16_to_b32"} : memref<16384xbf16, 1>, index -> <bf16>
            %289 = ascvec.cast %288, %280 {rounding = 6 : i32} : <bf16> -> <f32>
            %290 = ascvec.duplicate_scalar %cst_16, %280 : f32 -> <f32>
            %291 = ascvec.select %289, %290, %mask : <f32>
            %292 = ascvec.reinterpret %291 : <f32> -> <ui32>
            %293 = ascvec.duplicate_scalar %c2147483647_i32, %280 : i32 -> <ui32>
            %294 = ascvec.and %292, %293, %280 : <ui32>, <ui32> -> <ui32>
            %295 = ascvec.reduce_max %294, %282 : <ui32>
            %296 = ascvec.duplicate %295, %280 : <ui32> -> <ui32>
            %297 = ascvec.reduce_max %294, %283 : <ui32>
            %298 = ascvec.duplicate %297, %280 : <ui32> -> <ui32>
            %299 = ascvec.select %296, %298, %282 : <ui32>
            %300 = ascvec.shr %299, 23, %280 : <ui32> -> <ui32>
            %301 = ascvec.cmp_ge_scalar %300, %c255_i32, %280 : <ui32>, i32 -> !ascvec.mask
            %302 = ascvec.max_scalar %300, %c15_i32, %280 : <ui32>, i32
            %303 = ascvec.add_scalar %302, %c-15_i32, %280 : <ui32>, i32
            %304 = ascvec.duplicate_scalar %c255_i32, %301 merge %303 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %305 = ascvec.duplicate_scalar %c254_i32, %280 : i32 -> <ui32>
            %306 = ascvec.sub %305, %304, %280 : <ui32>
            %307 = ascvec.cmp_ne_scalar %304, %c0_i32_18, %280 : <ui32>, i32 -> !ascvec.mask
            %308 = ascvec.shl %306, 23, %307 : <ui32> -> <ui32>
            %309 = ascvec.duplicate_scalar %c2139160576_i32, %301 merge %308 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
            %310 = ascvec.reinterpret %309 : <ui32> -> <f32>
            %311 = ascvec.mul %291, %310, %280 : <f32>
            %312 = ascvec.shr %281, 2, %280 : <ui32> -> <ui32>
            %313 = ascvec.duplicate_scalar %c3_i32, %280 : i32 -> <ui32>
            %314 = ascvec.and %312, %313, %280 : <ui32>, <ui32> -> <ui32>
            %315 = ascvec.shl %314, 6, %280 : <ui32> -> <ui32>
            %316 = ascvec.shr %281, 4, %280 : <ui32> -> <ui32>
            %317 = ascvec.add %315, %316, %280 : <ui32>
            %318 = arith.remsi %arg19, %c16_i32 : i32
            %319 = arith.cmpi ne, %318, %c0_i32_18 : i32
            %320 = arith.cmpi slt, %318, %c0_i32_18 : i32
            %321 = arith.andi %319, %320 : i1
            %322 = arith.divsi %arg19, %c16_i32 : i32
            %323 = arith.extui %321 : i1 to i32
            %324 = arith.subi %322, %323 : i32
            %325 = arith.muli %324, %c256_i32 : i32
            %326 = arith.select %321, %c16_i32, %c0_i32_18 : i32
            %327 = arith.addi %318, %326 overflow<nsw> : i32
            %328 = arith.muli %327, %c4_i32_14 overflow<nsw> : i32
            %329 = arith.addi %325, %328 : i32
            %330 = ascvec.add_scalar %317, %329, %280 : <ui32>, i32
            %331 = ascvec.gather %arg13[%c0_17], %330, %280 : memref<4096xi32, 1>, <ui32>, !ascvec.mask -> <i32>
            %332 = ascvec.reinterpret %331 : <i32> -> <ui32>
            %333 = ascvec.shl %332, 8, %280 : <ui32> -> <ui32>
            %334 = ascvec.shr %332, 24, %280 : <ui32> -> <ui32>
            %335 = ascvec.or %333, %334, %280 : <ui32>, <ui32> -> <ui32>
            %336 = ascvec.and %281, %313, %280 : <ui32>, <ui32> -> <ui32>
            %337 = ascvec.duplicate_scalar %c1_i32_15, %280 : i32 -> <ui32>
            %338 = ascvec.and %336, %337, %280 : <ui32>, <ui32> -> <ui32>
            %339 = ascvec.cmp_eq_scalar %338, %c0_i32_18, %280 : <ui32>, i32 -> !ascvec.mask
            %340 = ascvec.select %332, %335, %339 : <ui32>
            %341 = ascvec.cmp_eq_scalar %336, %c0_i32_18, %280 : <ui32>, i32 -> !ascvec.mask
            %342 = ascvec.cmp_eq_scalar %336, %c3_i32, %280 : <ui32>, i32 -> !ascvec.mask
            %343 = ascvec.mask_or %341, %342, %280
            %344 = ascvec.shr %340, 16, %280 : <ui32> -> <ui32>
            %345 = ascvec.duplicate_scalar %c65535_i32, %280 : i32 -> <ui32>
            %346 = ascvec.and %340, %345, %280 : <ui32>, <ui32> -> <ui32>
            %347 = ascvec.select %344, %346, %343 : <ui32>
            %348 = ascvec.reinterpret %311 : <f32> -> <ui32>
            %349 = ascvec.shr %348, 24, %280 : <ui32> -> <ui32>
            %350 = ascvec.duplicate_scalar %c128_i32, %280 : i32 -> <ui32>
            %351 = ascvec.and %349, %350, %280 : <ui32>, <ui32> -> <ui32>
            %352 = ascvec.and %348, %293, %280 : <ui32>, <ui32> -> <ui32>
            %353 = ascvec.shr %352, 5, %280 : <ui32> -> <ui32>
            %354 = ascvec.duplicate_scalar %c29360128_i32, %280 : i32 -> <ui32>
            %355 = ascvec.sub %353, %354, %280 : <ui32>
            %356 = ascvec.reinterpret %352 : <ui32> -> <f32>
            %357 = ascvec.mul_scalar %356, %cst, %280 : <f32>, f32
            %358 = ascvec.cast %357, %280 {rounding = 4 : i32} : <f32> -> <i32>
            %359 = ascvec.reinterpret %358 : <i32> -> <ui32>
            %360 = ascvec.cmp_ge_scalar %352, %c947912704_i32, %280 : <ui32>, i32 -> !ascvec.mask
            %361 = ascvec.select %355, %359, %360 : <ui32>
            %362 = ascvec.and %347, %345, %280 : <ui32>, <ui32> -> <ui32>
            %363 = ascvec.add %361, %362, %280 : <ui32>
            %364 = ascvec.shr %363, 16, %280 : <ui32> -> <ui32>
            %365 = ascvec.min_scalar %364, %c123_i32, %280 : <ui32>, i32
            %366 = ascvec.or %365, %351, %280 : <ui32>, <ui32> -> <ui32>
            %367 = ascvec.duplicate_scalar %c127_i32, %280 : i32 -> <ui32>
            %368 = ascvec.cmp_gt_scalar %352, %c2139095040_i32, %280 : <ui32>, i32 -> !ascvec.mask
            %369 = ascvec.select %367, %366, %368 : <ui32>
            ascvec.storealign_pack %arg14[%287], %369, %mask {mode = "b32_to_b8"} : memref<16384xui8, 1>, index, <ui32>
            %370 = arith.muli %arg19, %c2_i32_13 : i32
            %371 = arith.index_cast %370 : i32 to index
            ascvec.storealign_1st %arg15[%371], %304 : memref<512xui32, 1>, index, <ui32>
            %372 = ascvec.duplicate_scalar %c32_i32, %280 : i32 -> <ui32>
            %373 = ascvec.gather_reg %304, %372 : <ui32>, <ui32> -> <ui32>
            %374 = arith.addi %370, %c1_i32_15 : i32
            %375 = arith.index_cast %374 : i32 to index
            ascvec.storealign_1st %arg15[%375], %373 : memref<512xui32, 1>, index, <ui32>
          }
          ascvec.mem_bar vst_vld
          scf.for %arg19 = %c0_i32_18 to %arg17 step %c64_i32  : i32 {
            %284 = arith.subi %arg17, %arg19 : i32
            %mask, %remaining_out = ascvec.update_mask %284 : !ascvec.mask
            %285 = ascvec.add_scalar %281, %arg19, %280 : <ui32>, i32
            %286 = ascvec.gather %arg15[%c0_17], %285, %mask : memref<512xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
            %287 = arith.index_cast %arg19 : i32 to index
            ascvec.storealign_pack %arg18[%287], %286, %mask {mode = "b32_to_b8"} : memref<1024xui8, 1>, index, <ui32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %110, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %98, 0 {cannir.buffer.generated_sync}
      }
      %117 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %118 = cannir.extract_shape %117 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %119 = cannir.extract_dim %118[0] : !cannir.shape<(-1, -1)>
      %120 = arith.minsi %74, %119 : i64
      %121 = arith.subi %119, %120 : i64
      %122 = arith.minsi %121, %c1_i64 : i64
      %123 = arith.cmpi sle, %74, %119 : i64
      %124 = arith.select %123, %122, %c0_i64 : i64
      %125 = arith.cmpi sgt, %124, %c1_i64 : i64
      %126 = arith.select %125, %c1_i64, %124 : i64
      %127 = cannir.extract_dim %118[1] : !cannir.shape<(-1, -1)>
      %128 = arith.maxsi %51, %c1_i64 : i64
      %129 = arith.divsi %127, %128 : i64
      %130 = arith.minsi %85, %129 : i64
      %131 = arith.muli %130, %51 : i64
      %132 = arith.subi %127, %131 : i64
      %133 = arith.minsi %51, %132 : i64
      %134 = arith.cmpi sle, %85, %129 : i64
      %135 = arith.select %134, %133, %c0_i64 : i64
      %136 = arith.cmpi sgt, %135, %c16384_i64 : i64
      %137 = arith.select %136, %c16384_i64, %135 : i64
      %138 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %139 = cannir.extract_stride %138 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %140 = cannir.extract_dim %139[0] : !cannir.stride<(-1, 1)>
      %141 = arith.muli %74, %140 : i64
      %142 = arith.addi %141, %86 : i64
      %143 = arith.muli %142, %c8_i64 : i64
      %144 = ascscalar.memref_address %alloc_1 : memref<1xui8>
      %145 = arith.divui %143, %c8_i64 : i64
      %146 = arith.addi %144, %145 : i64
      %147 = ascscalar.bind_gm_ptr %146 : memref<1xui8>
      %148 = ascscalar.bind_local_ptr %103 : memref<1xui8, 1>
      %149 = cannir.extract_stride %117 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %150 = cannir.extract_dim %149[0] : !cannir.stride<(-1, 1)>
      %151 = arith.cmpi ugt, %126, %c0_i64 : i64
      %152 = arith.cmpi ugt, %137, %c0_i64 : i64
      %153 = arith.andi %151, %152 : i1
      scf.if %153 {
        ascvec.get_buf mte3, %105, 0 {cannir.buffer.generated_sync}
        %193 = arith.trunci %126 : i64 to i32
        %194 = arith.trunci %137 : i64 to i32
        ascvec.copy_ub2gm %147[%c0], %148[%c0], %193, %194, %150, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %105, 0 {cannir.buffer.generated_sync}
      }
      %154 = arith.muli %50, %c2_i64 : i64
      %155 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %156 = cannir.extract_shape %155 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %157 = cannir.extract_dim %156[0] : !cannir.shape<(-1, -1)>
      %158 = arith.minsi %74, %157 : i64
      %159 = arith.subi %157, %158 : i64
      %160 = arith.minsi %159, %c1_i64 : i64
      %161 = arith.cmpi sle, %74, %157 : i64
      %162 = arith.select %161, %160, %c0_i64 : i64
      %163 = arith.cmpi sgt, %162, %c1_i64 : i64
      %164 = arith.select %163, %c1_i64, %162 : i64
      %165 = cannir.extract_dim %156[1] : !cannir.shape<(-1, -1)>
      %166 = arith.maxsi %154, %c1_i64 : i64
      %167 = arith.divsi %165, %166 : i64
      %168 = arith.minsi %85, %167 : i64
      %169 = arith.muli %168, %154 : i64
      %170 = arith.subi %165, %169 : i64
      %171 = arith.minsi %154, %170 : i64
      %172 = arith.cmpi sle, %85, %167 : i64
      %173 = arith.select %172, %171, %c0_i64 : i64
      %174 = arith.cmpi sgt, %173, %c1024_i64 : i64
      %175 = arith.select %174, %c1024_i64, %173 : i64
      %176 = cannir.extract_physical_layout %arg3 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %177 = cannir.extract_stride %176 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %178 = cannir.extract_dim %177[0] : !cannir.stride<(-1, 1)>
      %179 = arith.muli %74, %178 : i64
      %180 = arith.muli %85, %154 : i64
      %181 = arith.addi %179, %180 : i64
      %182 = arith.muli %181, %c8_i64 : i64
      %183 = ascscalar.memref_address %alloc_2 : memref<1xui8>
      %184 = arith.divui %182, %c8_i64 : i64
      %185 = arith.addi %183, %184 : i64
      %186 = ascscalar.bind_gm_ptr %185 : memref<1xui8>
      %187 = ascscalar.bind_local_ptr %108 : memref<1xui8, 1>
      %188 = cannir.extract_stride %155 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
      %189 = cannir.extract_dim %188[0] : !cannir.stride<(-1, 1)>
      %190 = arith.cmpi ugt, %164, %c0_i64 : i64
      %191 = arith.cmpi ugt, %175, %c0_i64 : i64
      %192 = arith.andi %190, %191 : i1
      scf.if %192 {
        ascvec.get_buf mte3, %110, 0 {cannir.buffer.generated_sync}
        %193 = arith.trunci %164 : i64 to i32
        %194 = arith.trunci %175 : i64 to i32
        ascvec.copy_ub2gm %186[%c0], %187[%c0], %193, %194, %189, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        ascvec.release_buf mte3, %110, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %91, %100 : i64, i64
    }
    return
  }
}
