module {
  func.func @run_graph_rng_899f7531bba7b15c8a082c5bb0e40041(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: i64, %arg5: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg7: i64) attributes {cannbotdsl.contract_digest = "899f7531bba7b15c8a082c5bb0e400410bf11cbb6c1160e2e67cb65296688086", cannbotdsl.public_slot_count = 8 : i64, cannbotdsl.slot_dtype_bits = [16, 8, 8, 64, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 1, 1, 0, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 2, 2, 1, 1, 2], cannbotdsl.slot_type_digests = ["b61381ca951a0f603976832b4955166996f3e8a7d1460da4be1b89bae4e92f47", "481dbcd23d79e136c72cda6b88f83676c3549f81b3cd3be5caf1bc2d99f3440f", "f3b37a0bd876490cb7c40dd697b264fbf7c668be5608a770181d9307f4b9c55d", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "a9d876affafdf57be3f0e1a45cb0b293a48851dbda8b74101f5184f5130b1b63", "a9d876affafdf57be3f0e1a45cb0b293a48851dbda8b74101f5184f5130b1b63", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__DynamicMxQuantSrKernel___call___0(%arg4) [%arg0, %arg1, %arg2, %arg3, %arg5, %arg6, %arg7] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 4, 5>, name = "_DynamicMxQuantSrKernel___call__", op_type = "_DynamicMxQuantSrKernel___call__", roles = array<i64: 0, 1, 1, 0, 0>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, i64, !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, i64
    return
  }
  func.func @kernel__DynamicMxQuantSrKernel___call___0(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>, %arg3: i64, %arg4: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg5: !cannir.tensor<<gm, i64>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg6: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__DynamicMxQuantSrKernel___call___0", cannir.resource.ids_prepared} {
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
    %alloc_2 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<1xi64>
    %alloc_3 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xi64>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<32768xbf16, 1>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<32768xbf16, 1>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 131072 : i64} : memref<32768xui8, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 163840 : i64} : memref<2048xui8, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 165888 : i64} : memref<1024xui32, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 169984 : i64} : memref<8192xi32, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 202752 : i64} : memref<2048xui16, 1>
    %0 = ascscalar.scalar_ptr %alloc_2[%c0] : memref<1xi64>, index -> <gm, i64>
    %1 = ascscalar.load_dev %0 : !ascscalar.scalar_ptr<gm, i64> -> i64
    %2 = ascscalar.scalar_ptr %alloc_3[%c0] : memref<1xi64>, index -> <gm, i64>
    %3 = ascscalar.load_dev %2 : !ascscalar.scalar_ptr<gm, i64> -> i64
    %4 = arith.addi %3, %arg6 : i64
    %5 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
    %6 = cannir.extract_shape %5 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
    %7 = cannir.extract_dim %6[0] : !cannir.shape<(-1, -1)>
    %8 = arith.cmpi eq, %7, %c1_i64 : i64
    %9 = scf.if %8 -> (i1) {
      %10 = cannir.extract_dim %6[1] : !cannir.shape<(-1, -1)>
      %11 = arith.remsi %10, %c64_i64 : i64
      %12 = arith.cmpi ne, %11, %c0_i64 : i64
      %13 = arith.cmpi slt, %11, %c0_i64 : i64
      %14 = arith.andi %12, %13 : i1
      %15 = arith.select %14, %c64_i64, %c0_i64 : i64
      %16 = arith.addi %11, %15 overflow<nsw> : i64
      %17 = arith.cmpi eq, %16, %c0_i64 : i64
      %18 = scf.if %17 -> (i1) {
        %19 = ascscalar.get_block_num
        %20 = arith.muli %19, %c32768_i64 : i64
        %21 = arith.cmpi sgt, %10, %20 : i64
        %22 = scf.if %21 -> (i1) {
          %23 = arith.remsi %10, %20 : i64
          %24 = arith.cmpi ne, %23, %c0_i64 : i64
          %25 = arith.cmpi slt, %23, %c0_i64 : i64
          %26 = arith.cmpi slt, %20, %c0_i64 : i64
          %27 = arith.xori %25, %26 : i1
          %28 = arith.andi %24, %27 : i1
          %29 = arith.select %28, %20, %c0_i64 : i64
          %30 = arith.addi %23, %29 : i64
          %31 = arith.cmpi ne, %30, %c0_i64 : i64
          scf.yield %31 : i1
        } else {
          scf.yield %false : i1
        }
        scf.yield %22 : i1
      } else {
        scf.yield %false : i1
      }
      scf.yield %18 : i1
    } else {
      scf.yield %false : i1
    }
    scf.if %9 {
      %10 = cannir.extract_dim %6[1] : !cannir.shape<(-1, -1)>
      %11 = arith.remsi %10, %c64_i64 : i64
      %12 = arith.cmpi ne, %11, %c0_i64 : i64
      %13 = arith.cmpi slt, %11, %c0_i64 : i64
      %14 = arith.andi %12, %13 : i1
      %15 = arith.divsi %10, %c64_i64 : i64
      %16 = arith.extui %14 : i1 to i64
      %17 = arith.subi %15, %16 : i64
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
      %30 = arith.cmpi sge, %29, %c4_i64 : i64
      %31 = scf.if %30 -> (i64) {
        %51 = arith.addi %29, %c3_i64 : i64
        %52 = arith.remsi %51, %c4_i64 : i64
        %53 = arith.cmpi ne, %52, %c0_i64 : i64
        %54 = arith.cmpi slt, %52, %c0_i64 : i64
        %55 = arith.andi %53, %54 : i1
        %56 = arith.divsi %51, %c4_i64 : i64
        %57 = arith.extui %55 : i1 to i64
        %58 = arith.subi %56, %57 : i64
        %59 = arith.muli %58, %c4_i64 : i64
        scf.yield %59 : i64
      } else {
        scf.yield %29 : i64
      }
      %32 = arith.muli %18, %c64_i64 : i64
      %33 = arith.cmpi sge, %17, %32 : i64
      %34 = scf.if %33 -> (i64) {
        %51 = arith.addi %31, %c15_i64 : i64
        %52 = arith.remsi %51, %c16_i64 : i64
        %53 = arith.cmpi ne, %52, %c0_i64 : i64
        %54 = arith.cmpi slt, %52, %c0_i64 : i64
        %55 = arith.andi %53, %54 : i1
        %56 = arith.divsi %51, %c16_i64 : i64
        %57 = arith.extui %55 : i1 to i64
        %58 = arith.subi %56, %57 : i64
        %59 = arith.muli %58, %c16_i64 : i64
        scf.yield %59 : i64
      } else {
        scf.yield %31 : i64
      }
      %35 = arith.muli %34, %c64_i64 : i64
      %36 = ascscalar.get_block_idx
      %37 = arith.muli %36, %35 : i64
      %38 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
      %39 = cannir.extract_shape %38 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
      %40 = cannir.extract_dim %39[1] : !cannir.shape<(-1, -1)>
      %41 = arith.remsi %10, %c16_i64 : i64
      %42 = arith.cmpi ne, %41, %c0_i64 : i64
      %43 = arith.cmpi slt, %41, %c0_i64 : i64
      %44 = arith.andi %42, %43 : i1
      %45 = arith.divsi %10, %c16_i64 : i64
      %46 = arith.extui %44 : i1 to i64
      %47 = arith.subi %45, %46 : i64
      %48 = arith.cmpi eq, %40, %47 : i64
      %49 = arith.select %48, %c4_i64, %c2_i64 : i64
      %50 = arith.cmpi slt, %37, %10 : i64
      scf.if %50 {
        %51 = arith.muli %34, %49 : i64
        %52 = arith.subi %10, %37 : i64
        %53 = arith.cmpi sgt, %52, %35 : i64
        %54 = arith.select %53, %35, %52 : i64
        %55 = arith.addi %54, %c32767_i64 : i64
        %56 = arith.remsi %55, %c32768_i64 : i64
        %57 = arith.cmpi ne, %56, %c0_i64 : i64
        %58 = arith.cmpi slt, %56, %c0_i64 : i64
        %59 = arith.andi %57, %58 : i1
        %60 = arith.divsi %55, %c32768_i64 : i64
        %61 = arith.extui %59 : i1 to i64
        %62 = arith.subi %60, %61 : i64
        %63 = scf.for %arg7 = %c0_i64 to %62 step %c1_i64 iter_args(%arg8 = %c0_i64) -> (i64)  : i64 {
          %64 = arith.muli %arg7, %c32768_i64 : i64
          %65 = arith.subi %54, %64 : i64
          %66 = arith.cmpi sgt, %65, %c32768_i64 : i64
          %67 = arith.select %66, %c32768_i64, %65 : i64
          %68 = arith.cmpi eq, %arg8, %c0_i64 : i64
          %69 = ascscalar.memref_address %alloc_4 : memref<32768xbf16, 1>
          %70 = ascscalar.memref_address %alloc_5 : memref<32768xbf16, 1>
          %71 = arith.select %68, %69, %70 : i64
          %72 = arith.cmpi ne, %arg8, %c0_i64 : i64
          %73 = arith.extui %72 : i1 to i64
          %74 = arith.addi %arg8, %c1_i64 overflow<nsw> : i64
          %75 = arith.remui %74, %c2_i64 : i64
          %76 = arith.minsi %7, %c0_i64 : i64
          %77 = arith.subi %7, %76 : i64
          %78 = arith.minsi %77, %c1_i64 : i64
          %79 = arith.cmpi sge, %7, %c0_i64 : i64
          %80 = arith.select %79, %78, %c0_i64 : i64
          %81 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %82 = cannir.extract_stride %81 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %83 = cannir.extract_dim %82[0] : !cannir.stride<(-1, 1)>
          %84 = arith.muli %83, %c2_i64 : i64
          %85 = arith.maxsi %35, %c1_i64 : i64
          %86 = arith.divsi %10, %85 : i64
          %87 = arith.minsi %36, %86 : i64
          %88 = arith.muli %87, %35 : i64
          %89 = arith.subi %10, %88 : i64
          %90 = arith.minsi %35, %89 : i64
          %91 = arith.cmpi sle, %36, %86 : i64
          %92 = arith.select %91, %90, %c0_i64 : i64
          %93 = arith.divsi %92, %c32768_i64 : i64
          %94 = arith.minsi %arg7, %93 : i64
          %95 = arith.muli %94, %c32768_i64 : i64
          %96 = arith.subi %92, %95 : i64
          %97 = arith.minsi %96, %c32768_i64 : i64
          %98 = arith.cmpi sle, %arg7, %93 : i64
          %99 = arith.select %98, %97, %c0_i64 : i64
          %100 = arith.muli %arg7, %c65536_i64 : i64
          %101 = arith.muli %37, %c2_i64 : i64
          %102 = arith.addi %100, %101 : i64
          %103 = arith.muli %99, %c2_i64 : i64
          %104 = arith.cmpi eq, %103, %c0_i64 : i64
          %105 = arith.select %104, %c0_i64, %80 : i64
          %106 = arith.select %104, %c2_i64, %103 : i64
          %107 = ascscalar.bind_local_ptr %71 : memref<1xbf16, 1>
          ascvec.get_buf mte2, %73, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%107, %alloc, %c0_i32, %c0_i32, %c0_i64, %102, %105, %106, %84, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %73, 0 {cannir.buffer.generated_sync}
          %108 = ascscalar.bind_local_ptr %71 : memref<32768xbf16, 1>
          %109 = arith.addi %37, %64 : i64
          %110 = arith.cmpi sge, %67, %c256_i64 : i64
          %111 = arith.remsi %67, %c256_i64 : i64
          %112 = arith.cmpi eq, %111, %c0_i64 : i64
          %113 = arith.select %110, %112, %false : i1
          scf.if %113 {
            %193 = arith.remsi %arg3, %c256_i64 : i64
            %194 = arith.cmpi ne, %193, %c0_i64 : i64
            %195 = arith.cmpi slt, %193, %c0_i64 : i64
            %196 = arith.andi %194, %195 : i1
            %197 = arith.select %196, %c256_i64, %c0_i64 : i64
            %198 = arith.addi %193, %197 overflow<nsw> : i64
            %199 = arith.cmpi eq, %198, %c0_i64 : i64
            scf.if %199 {
              %200 = arith.cmpi sge, %67, %c1024_i64 : i64
              %201 = arith.remsi %67, %c1024_i64 : i64
              %202 = arith.cmpi eq, %201, %c0_i64 : i64
              %203 = arith.select %200, %202, %false : i1
              scf.if %203 {
                %204 = arith.divsi %67, %c256_i64 : i64
                %205 = arith.addi %67, %c1023_i64 : i64
                %206 = arith.divsi %205, %c1024_i64 : i64
                %207 = arith.trunci %1 : i64 to i32
                %208 = arith.extui %207 : i32 to i64
                %209 = arith.subi %1, %208 : i64
                %210 = arith.remsi %209, %c4294967296_i64 : i64
                %211 = arith.cmpi ne, %210, %c0_i64 : i64
                %212 = arith.cmpi slt, %210, %c0_i64 : i64
                %213 = arith.andi %211, %212 : i1
                %214 = arith.divsi %209, %c4294967296_i64 : i64
                %215 = arith.extui %213 : i1 to i64
                %216 = arith.subi %214, %215 : i64
                %217 = arith.trunci %216 : i64 to i32
                %218 = arith.divui %109, %c32_i64 : i64
                %219 = arith.trunci %218 : i64 to i32
                %220 = arith.extui %219 : i32 to i64
                %221 = arith.subi %218, %220 : i64
                %222 = arith.remsi %221, %c4294967296_i64 : i64
                %223 = arith.cmpi ne, %222, %c0_i64 : i64
                %224 = arith.cmpi slt, %222, %c0_i64 : i64
                %225 = arith.andi %223, %224 : i1
                %226 = arith.divsi %221, %c4294967296_i64 : i64
                %227 = arith.extui %225 : i1 to i64
                %228 = arith.subi %226, %227 : i64
                %229 = arith.trunci %228 : i64 to i32
                %230 = arith.trunci %4 : i64 to i32
                %231 = arith.extui %230 : i32 to i64
                %232 = arith.subi %4, %231 : i64
                %233 = arith.remsi %232, %c4294967296_i64 : i64
                %234 = arith.cmpi ne, %233, %c0_i64 : i64
                %235 = arith.cmpi slt, %233, %c0_i64 : i64
                %236 = arith.andi %234, %235 : i1
                %237 = arith.divsi %232, %c4294967296_i64 : i64
                %238 = arith.extui %236 : i1 to i64
                %239 = arith.subi %237, %238 : i64
                %240 = arith.trunci %239 : i64 to i32
                %241 = arith.divui %230, %c4_i32 : i32
                %242 = arith.remui %240, %c4_i32 : i32
                %243 = arith.muli %242, %c1073741824_i32 : i32
                %244 = arith.addi %241, %243 : i32
                %245 = arith.divui %240, %c4_i32 : i32
                %246 = arith.muli %206, %c32_i64 : i64
                %247 = arith.divui %218, %c16384_i64 : i64
                %248 = arith.addi %218, %246 : i64
                %249 = arith.subi %248, %c1_i64 : i64
                %250 = arith.divui %249, %c16384_i64 : i64
                %251 = arith.cmpi eq, %247, %250 : i64
                scf.if %251 {
                  %261 = arith.trunci %247 : i64 to i32
                  %262 = arith.extui %261 : i32 to i64
                  %263 = arith.subi %247, %262 : i64
                  %264 = arith.remsi %263, %c4294967296_i64 : i64
                  %265 = arith.cmpi ne, %264, %c0_i64 : i64
                  %266 = arith.cmpi slt, %264, %c0_i64 : i64
                  %267 = arith.andi %265, %266 : i1
                  %268 = arith.divsi %263, %c4294967296_i64 : i64
                  %269 = arith.extui %267 : i1 to i64
                  %270 = arith.subi %268, %269 : i64
                  %271 = arith.trunci %270 : i64 to i32
                  %272 = arith.addi %244, %261 : i32
                  %273 = arith.addi %245, %271 : i32
                  %274 = arith.cmpi ult, %272, %244 : i32
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
                  %290 = arith.xori %276, %217 : i32
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
                  %304 = arith.addi %207, %c-1640531527_i32 : i32
                  %305 = arith.addi %217, %c-1150833019_i32 : i32
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
                  %321 = arith.xori %289, %207 : i32
                  %322 = arith.addi %207, %c1013904242_i32 : i32
                  %323 = arith.xori %293, %322 : i32
                  %324 = arith.addi %217, %c1993301258_i32 : i32
                  %325 = arith.xori %320, %324 : i32
                  ascsimt.vf(%219, %321, %305, %323, %325, %207, %217, %310, %alloc_9, %246 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: memref<8192xi32, 1>, %arg18: i64):
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
                    %c16384_i32 = arith.constant 16384 : i32
                    %326 = ascsimt.thread_idx {ascsimt.scope = 1 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %327 = arith.extsi %326 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg19 = %327 to %arg18 step %c256_i64_11  : i64 {
                      %328 = arith.trunci %arg19 : i64 to i32
                      %329 = arith.addi %arg9, %328 : i32
                      %330 = arith.remui %329, %c16384_i32 : i32
                      %331 = arith.xori %330, %arg10 : i32
                      %332 = arith.muli %331, %c-766435501_i32 : i32
                      %333 = "ascsimt.bitcast"(%331) : (i32) -> ui32
                      %334 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %335 = ascsimt.math_binary %333, %334, "mulhi" : ui32, ui32 -> ui32
                      %336 = "ascsimt.bitcast"(%335) : (ui32) -> i32
                      %337 = arith.xori %336, %arg11 : i32
                      %338 = arith.muli %337, %c-845247145_i32 : i32
                      %339 = "ascsimt.bitcast"(%337) : (i32) -> ui32
                      %340 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %341 = ascsimt.math_binary %339, %340, "mulhi" : ui32, ui32 -> ui32
                      %342 = "ascsimt.bitcast"(%341) : (ui32) -> i32
                      %343 = arith.xori %342, %arg12 : i32
                      %344 = arith.xori %332, %arg13 : i32
                      %345:4 = scf.for %arg20 = %c3_i32 to %c10_i32 step %c1_i32_16 iter_args(%arg21 = %343, %arg22 = %338, %arg23 = %344, %arg24 = %arg16) -> (i32, i32, i32, i32)  : i32 {
                        %354 = arith.muli %arg21, %c-766435501_i32 : i32
                        %355 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %356 = ascsimt.math_binary %355, %334, "mulhi" : ui32, ui32 -> ui32
                        %357 = "ascsimt.bitcast"(%356) : (ui32) -> i32
                        %358 = arith.muli %arg23, %c-845247145_i32 : i32
                        %359 = "ascsimt.bitcast"(%arg23) : (i32) -> ui32
                        %360 = ascsimt.math_binary %359, %340, "mulhi" : ui32, ui32 -> ui32
                        %361 = "ascsimt.bitcast"(%360) : (ui32) -> i32
                        %362 = arith.muli %arg20, %c-1640531527_i32_18 : i32
                        %363 = arith.addi %arg14, %362 : i32
                        %364 = arith.muli %arg20, %c-1150833019_i32_17 : i32
                        %365 = arith.addi %arg15, %364 : i32
                        %366 = arith.xori %361, %arg22 : i32
                        %367 = arith.xori %366, %363 : i32
                        %368 = arith.xori %357, %arg24 : i32
                        %369 = arith.xori %368, %365 : i32
                        scf.yield %367, %358, %369, %354 : i32, i32, i32, i32
                      }
                      %346 = arith.muli %arg19, %c4_i64_15 : i64
                      %347 = arith.index_cast %346 : i64 to index
                      ascsimt.store %arg17[%347], %345#0 : memref<8192xi32, 1>, i32
                      %348 = arith.addi %346, %c1_i64_14 : i64
                      %349 = arith.index_cast %348 : i64 to index
                      ascsimt.store %arg17[%349], %345#1 : memref<8192xi32, 1>, i32
                      %350 = arith.addi %346, %c2_i64_13 : i64
                      %351 = arith.index_cast %350 : i64 to index
                      ascsimt.store %arg17[%351], %345#2 : memref<8192xi32, 1>, i32
                      %352 = arith.addi %346, %c3_i64_12 : i64
                      %353 = arith.index_cast %352 : i64 to index
                      ascsimt.store %arg17[%353], %345#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%219, %229, %244, %245, %207, %217, %alloc_9, %246 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
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
                    %c16384_i32 = arith.constant 16384 : i32
                    %261 = ascsimt.thread_idx {ascsimt.scope = 2 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %262 = arith.extsi %261 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg17 = %262 to %arg16 step %c256_i64_11  : i64 {
                      %263 = arith.trunci %arg17 : i64 to i32
                      %264 = arith.addi %arg9, %263 : i32
                      %265 = arith.cmpi ult, %264, %arg9 : i32
                      %266 = arith.extui %265 : i1 to i32
                      %267 = arith.addi %arg10, %266 : i32
                      %268 = arith.divui %264, %c16384_i32 : i32
                      %269 = arith.remui %267, %c16384_i32 : i32
                      %270 = arith.muli %269, %c262144_i32 : i32
                      %271 = arith.addi %268, %270 : i32
                      %272 = arith.divui %267, %c16384_i32 : i32
                      %273 = arith.addi %arg11, %271 : i32
                      %274 = arith.addi %arg12, %272 : i32
                      %275 = arith.cmpi ult, %273, %arg11 : i32
                      %276 = arith.extui %275 : i1 to i32
                      %277 = arith.addi %274, %276 : i32
                      %278 = arith.remui %264, %c16384_i32 : i32
                      %279:4 = scf.for %arg18 = %c0_i32_17 to %c10_i32 step %c1_i32_16 iter_args(%arg19 = %c0_i32_17, %arg20 = %278, %arg21 = %273, %arg22 = %277) -> (i32, i32, i32, i32)  : i32 {
                        %288 = arith.muli %arg19, %c-766435501_i32 : i32
                        %289 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %290 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %291 = ascsimt.math_binary %289, %290, "mulhi" : ui32, ui32 -> ui32
                        %292 = "ascsimt.bitcast"(%291) : (ui32) -> i32
                        %293 = arith.muli %arg21, %c-845247145_i32 : i32
                        %294 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %295 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %296 = ascsimt.math_binary %294, %295, "mulhi" : ui32, ui32 -> ui32
                        %297 = "ascsimt.bitcast"(%296) : (ui32) -> i32
                        %298 = arith.muli %arg18, %c-1640531527_i32_19 : i32
                        %299 = arith.addi %arg13, %298 : i32
                        %300 = arith.muli %arg18, %c-1150833019_i32_18 : i32
                        %301 = arith.addi %arg14, %300 : i32
                        %302 = arith.xori %297, %arg20 : i32
                        %303 = arith.xori %302, %299 : i32
                        %304 = arith.xori %292, %arg22 : i32
                        %305 = arith.xori %304, %301 : i32
                        scf.yield %303, %293, %305, %288 : i32, i32, i32, i32
                      }
                      %280 = arith.muli %arg17, %c4_i64_15 : i64
                      %281 = arith.index_cast %280 : i64 to index
                      ascsimt.store %arg15[%281], %279#0 : memref<8192xi32, 1>, i32
                      %282 = arith.addi %280, %c1_i64_14 : i64
                      %283 = arith.index_cast %282 : i64 to index
                      ascsimt.store %arg15[%283], %279#1 : memref<8192xi32, 1>, i32
                      %284 = arith.addi %280, %c2_i64_13 : i64
                      %285 = arith.index_cast %284 : i64 to index
                      ascsimt.store %arg15[%285], %279#2 : memref<8192xi32, 1>, i32
                      %286 = arith.addi %280, %c3_i64_12 : i64
                      %287 = arith.index_cast %286 : i64 to index
                      ascsimt.store %arg15[%287], %279#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %252 = arith.trunci %49 : i64 to i32
                %253 = arith.divsi %252, %c2_i32 : i32
                %254 = arith.trunci %253 : i32 to i8
                %255 = arith.trunci %204 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %256 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%254, %alloc_7, %108, %256, %alloc_10, %255 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, memref<2048xui16, 1>, i32) {
                ^bb0(%arg9: i8, %arg10: memref<2048xui8, 1>, %arg11: memref<32768xbf16, 1>, %arg12: memref<?xui16, 1>, %arg13: memref<2048xui16, 1>, %arg14: i32):
                  %c16_11 = arith.constant 16 : index
                  %c4_i32_12 = arith.constant 4 : i32
                  %c0_i32_13 = arith.constant 0 : i32
                  %c3_i32 = arith.constant 3 : i32
                  %c2_i32_14 = arith.constant 2 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c1_i32_15 = arith.constant 1 : i32
                  %c0_16 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %261 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %262 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %263 = ascvec.duplicate_scalar %c32767_i16, %261 : i16 -> <ui16>
                  %264 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %265 = ascvec.shr %264, 4, %261 : <ui16> -> <ui16>
                  %266 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %267 = ascvec.duplicate_scalar %c1_i8, %262 : i8 -> <ui8>
                  %268 = ascvec.and %266, %267, %262 : <ui8>, <ui8> -> <ui8>
                  %269 = ascvec.cmp_lt_scalar %266, %c16_i8, %262 : <ui8>, i8 -> !ascvec.mask
                  %270 = ascvec.cmp_lt_scalar %268, %arg9, %262 : <ui8>, i8 -> !ascvec.mask
                  %271 = ascvec.mask_and %269, %270, %262
                  ascvec.store_unalign_begin %arg10 : memref<2048xui8, 1>
                  %mask_17, %remaining_out_18 = ascvec.update_mask %c1_i32_15 elem_bits 16 : !ascvec.mask
                  %272 = ascvec.duplicate_scalar %c-1_i16, %mask_17 : i16 -> <ui16>
                  %273 = scf.for %arg15 = %c0_i32_13 to %arg14 step %c4_i32_12 iter_args(%arg16 = %272) -> (!ascvec.vreg<ui16>)  : i32 {
                    %274 = ascvec.duplicate_scalar %c-1_i16, %261 : i16 -> <ui16>
                    %275 = ascvec.duplicate_scalar %c0_i16, %261 : i16 -> <ui16>
                    %276 = arith.muli %arg15, %c256_i32 : i32
                    %277 = arith.index_cast %276 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg11[%277] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %278 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %279 = ascvec.and %278, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %280 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %281 = ascvec.and %280, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %282 = ascvec.max %279, %281, %261 : <ui16>
                    %283 = ascvec.reduce_max_datablock %282, %261 : <ui16>
                    %284 = ascvec.shr %283, 7, %261 : <ui16> -> <ui16>
                    %285 = ascvec.cmp_ge_scalar %284, %c255_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %286 = ascvec.max_scalar %284, %c15_i16, %261 : <ui16>, i16
                    %287 = ascvec.add_scalar %286, %c-15_i16, %261 : <ui16>, i16
                    %288 = ascvec.duplicate_scalar %c255_i16, %285 merge %287 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %289 = ascvec.duplicate_scalar %c254_i16, %261 : i16 -> <ui16>
                    %290 = ascvec.sub %289, %288, %261 : <ui16>
                    %291 = ascvec.cmp_ne_scalar %288, %c0_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %292 = ascvec.shl %290, 7, %291 : <ui16> -> <ui16>
                    %293 = ascvec.max %275, %288, %mask : <ui16>
                    %294 = ascvec.gather_reg %292, %265 : <ui16>, <ui16> -> <ui16>
                    %295 = ascvec.reinterpret %294 : <ui16> -> <bf16>
                    %296 = ascvec.mul %result0, %295, %261 : <bf16>
                    %297 = ascvec.mul %result1, %295, %261 : <bf16>
                    ascvec.storealign_intlv %arg11[%277], %296, %297 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %298 = ascvec.reinterpret %296 : <bf16> -> <ui16>
                    %299 = ascvec.and %298, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %300 = ascvec.reinterpret %297 : <bf16> -> <ui16>
                    %301 = ascvec.and %300, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %302 = ascvec.min %299, %301, %261 : <ui16>
                    %303 = ascvec.min %274, %302, %261 : <ui16>
                    %304 = arith.muli %arg15, %c16_i32 : i32
                    %305 = arith.index_cast %304 : i32 to index
                    ascvec.storealign %arg12[%305], %288, %mask : memref<?xui16, 1>, index, <ui16>
                    %306 = ascvec.reinterpret %288 : <ui16> -> <ui8>
                    %307 = ascvec.squeeze_store_reg %306, %271 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_16], %307 : memref<2048xui8, 1>, <ui8>
                    %308 = arith.addi %arg15, %c1_i32_15 : i32
                    %309 = arith.muli %308, %c256_i32 : i32
                    %310 = arith.index_cast %309 : i32 to index
                    %result0_20, %result1_21 = ascvec.loadalign_deintlv %arg11[%310] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %311 = ascvec.reinterpret %result0_20 : <bf16> -> <ui16>
                    %312 = ascvec.and %311, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %313 = ascvec.reinterpret %result1_21 : <bf16> -> <ui16>
                    %314 = ascvec.and %313, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %315 = ascvec.max %312, %314, %261 : <ui16>
                    %316 = ascvec.reduce_max_datablock %315, %261 : <ui16>
                    %317 = ascvec.shr %316, 7, %261 : <ui16> -> <ui16>
                    %318 = ascvec.cmp_ge_scalar %317, %c255_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %319 = ascvec.max_scalar %317, %c15_i16, %261 : <ui16>, i16
                    %320 = ascvec.add_scalar %319, %c-15_i16, %261 : <ui16>, i16
                    %321 = ascvec.duplicate_scalar %c255_i16, %318 merge %320 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %322 = ascvec.sub %289, %321, %261 : <ui16>
                    %323 = ascvec.cmp_ne_scalar %321, %c0_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %324 = ascvec.shl %322, 7, %323 : <ui16> -> <ui16>
                    %325 = ascvec.max %293, %321, %mask : <ui16>
                    %326 = ascvec.gather_reg %324, %265 : <ui16>, <ui16> -> <ui16>
                    %327 = ascvec.reinterpret %326 : <ui16> -> <bf16>
                    %328 = ascvec.mul %result0_20, %327, %261 : <bf16>
                    %329 = ascvec.mul %result1_21, %327, %261 : <bf16>
                    ascvec.storealign_intlv %arg11[%310], %328, %329 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %330 = ascvec.reinterpret %328 : <bf16> -> <ui16>
                    %331 = ascvec.and %330, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %332 = ascvec.reinterpret %329 : <bf16> -> <ui16>
                    %333 = ascvec.and %332, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %334 = ascvec.min %331, %333, %261 : <ui16>
                    %335 = ascvec.min %303, %334, %261 : <ui16>
                    %336 = arith.muli %308, %c16_i32 : i32
                    %337 = arith.index_cast %336 : i32 to index
                    ascvec.storealign %arg12[%337], %321, %mask : memref<?xui16, 1>, index, <ui16>
                    %338 = ascvec.reinterpret %321 : <ui16> -> <ui8>
                    %339 = ascvec.squeeze_store_reg %338, %271 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_16], %339 : memref<2048xui8, 1>, <ui8>
                    %340 = arith.addi %arg15, %c2_i32_14 : i32
                    %341 = arith.muli %340, %c256_i32 : i32
                    %342 = arith.index_cast %341 : i32 to index
                    %result0_22, %result1_23 = ascvec.loadalign_deintlv %arg11[%342] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %343 = ascvec.reinterpret %result0_22 : <bf16> -> <ui16>
                    %344 = ascvec.and %343, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %345 = ascvec.reinterpret %result1_23 : <bf16> -> <ui16>
                    %346 = ascvec.and %345, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %347 = ascvec.max %344, %346, %261 : <ui16>
                    %348 = ascvec.reduce_max_datablock %347, %261 : <ui16>
                    %349 = ascvec.shr %348, 7, %261 : <ui16> -> <ui16>
                    %350 = ascvec.cmp_ge_scalar %349, %c255_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %351 = ascvec.max_scalar %349, %c15_i16, %261 : <ui16>, i16
                    %352 = ascvec.add_scalar %351, %c-15_i16, %261 : <ui16>, i16
                    %353 = ascvec.duplicate_scalar %c255_i16, %350 merge %352 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %354 = ascvec.sub %289, %353, %261 : <ui16>
                    %355 = ascvec.cmp_ne_scalar %353, %c0_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %356 = ascvec.shl %354, 7, %355 : <ui16> -> <ui16>
                    %357 = ascvec.max %325, %353, %mask : <ui16>
                    %358 = ascvec.gather_reg %356, %265 : <ui16>, <ui16> -> <ui16>
                    %359 = ascvec.reinterpret %358 : <ui16> -> <bf16>
                    %360 = ascvec.mul %result0_22, %359, %261 : <bf16>
                    %361 = ascvec.mul %result1_23, %359, %261 : <bf16>
                    ascvec.storealign_intlv %arg11[%342], %360, %361 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %362 = ascvec.reinterpret %360 : <bf16> -> <ui16>
                    %363 = ascvec.and %362, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %364 = ascvec.reinterpret %361 : <bf16> -> <ui16>
                    %365 = ascvec.and %364, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %366 = ascvec.min %363, %365, %261 : <ui16>
                    %367 = ascvec.min %335, %366, %261 : <ui16>
                    %368 = arith.muli %340, %c16_i32 : i32
                    %369 = arith.index_cast %368 : i32 to index
                    ascvec.storealign %arg12[%369], %353, %mask : memref<?xui16, 1>, index, <ui16>
                    %370 = ascvec.reinterpret %353 : <ui16> -> <ui8>
                    %371 = ascvec.squeeze_store_reg %370, %271 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_16], %371 : memref<2048xui8, 1>, <ui8>
                    %372 = arith.addi %arg15, %c3_i32 : i32
                    %373 = arith.muli %372, %c256_i32 : i32
                    %374 = arith.index_cast %373 : i32 to index
                    %result0_24, %result1_25 = ascvec.loadalign_deintlv %arg11[%374] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %375 = ascvec.reinterpret %result0_24 : <bf16> -> <ui16>
                    %376 = ascvec.and %375, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %377 = ascvec.reinterpret %result1_25 : <bf16> -> <ui16>
                    %378 = ascvec.and %377, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %379 = ascvec.max %376, %378, %261 : <ui16>
                    %380 = ascvec.reduce_max_datablock %379, %261 : <ui16>
                    %381 = ascvec.shr %380, 7, %261 : <ui16> -> <ui16>
                    %382 = ascvec.cmp_ge_scalar %381, %c255_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %383 = ascvec.max_scalar %381, %c15_i16, %261 : <ui16>, i16
                    %384 = ascvec.add_scalar %383, %c-15_i16, %261 : <ui16>, i16
                    %385 = ascvec.duplicate_scalar %c255_i16, %382 merge %384 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %386 = ascvec.sub %289, %385, %261 : <ui16>
                    %387 = ascvec.cmp_ne_scalar %385, %c0_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %388 = ascvec.shl %386, 7, %387 : <ui16> -> <ui16>
                    %389 = ascvec.max %357, %385, %mask : <ui16>
                    %390 = ascvec.gather_reg %388, %265 : <ui16>, <ui16> -> <ui16>
                    %391 = ascvec.reinterpret %390 : <ui16> -> <bf16>
                    %392 = ascvec.mul %result0_24, %391, %261 : <bf16>
                    %393 = ascvec.mul %result1_25, %391, %261 : <bf16>
                    ascvec.storealign_intlv %arg11[%374], %392, %393 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %394 = ascvec.reinterpret %392 : <bf16> -> <ui16>
                    %395 = ascvec.and %394, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %396 = ascvec.reinterpret %393 : <bf16> -> <ui16>
                    %397 = ascvec.and %396, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %398 = ascvec.min %395, %397, %261 : <ui16>
                    %399 = ascvec.min %367, %398, %261 : <ui16>
                    %400 = arith.muli %372, %c16_i32 : i32
                    %401 = arith.index_cast %400 : i32 to index
                    ascvec.storealign %arg12[%401], %385, %mask : memref<?xui16, 1>, index, <ui16>
                    %402 = ascvec.reinterpret %385 : <ui16> -> <ui8>
                    %403 = ascvec.squeeze_store_reg %402, %271 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_16], %403 : memref<2048xui8, 1>, <ui8>
                    %404 = ascvec.reduce_min %399, %261 : <ui16>
                    %405 = ascvec.reduce_max %389, %mask : <ui16>
                    %406 = ascvec.cmp_eq_scalar %405, %c255_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %407 = ascvec.duplicate_scalar %c0_i16, %406 merge %404 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.storealign_1st %arg13[%305], %407 : memref<2048xui16, 1>, index, <ui16>
                    %408 = ascvec.min %arg16, %407, %mask_17 : <ui16>
                    scf.yield %408 : !ascvec.vreg<ui16>
                  }
                  %c0_19 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg15 = %c0_19 to %c1 step %c1 {
                    ascvec.store_unalign_post %arg10[%c0_16] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg13[%c16_11], %273 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %257 = memref.load %alloc_10[%c16] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %258 = cannir.scalar_storage_cast %257 : ui16 -> i16
                %259 = arith.extui %258 : i16 to i64
                %260 = arith.cmpi sge, %259, %c14464_i64 : i64
                scf.if %260 {
                  ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %261 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                  %262 = ascscalar.bitcast_memref %108 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%108, %261, %262, %alloc_6, %255 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_11 = arith.constant 1 : i32
                    %c0_i32_12 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c128_i32 = arith.constant 128 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c31_i16 = arith.constant 31 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %263 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %264 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %265 = ascvec.duplicate_scalar %c255_i16, %263 : i16 -> <ui16>
                    %266 = ascvec.duplicate_scalar %c-128_i8, %264 : i8 -> <ui8>
                    %267 = ascvec.duplicate_scalar %c31_i16, %263 : i16 -> <ui16>
                    scf.for %arg14 = %c0_i32_12 to %arg13 step %c1_i32_11  : i32 {
                      %268 = arith.muli %arg14, %c256_i32 : i32
                      %269 = arith.index_cast %268 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg9[%269] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %270 = arith.muli %arg14, %c128_i32 : i32
                      %271 = arith.index_cast %270 : i32 to index
                      %272 = ascvec.loadalign %arg10[%271] : memref<?xui8, 1>, index -> <ui8>
                      %273 = ascvec.unpack %272 {part = "lower"} : <ui8> -> <ui16>
                      %274 = ascvec.and %273, %267, %263 : <ui16>, <ui16> -> <ui16>
                      %275 = ascvec.shr %273, 3, %263 : <ui16> -> <ui16>
                      %276 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %277 = ascvec.add %276, %274, %263 : <ui16>
                      %278 = ascvec.shr %277, 5, %263 : <ui16> -> <ui16>
                      %279 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %280 = ascvec.add %279, %275, %263 : <ui16>
                      %281 = ascvec.shr %280, 5, %263 : <ui16> -> <ui16>
                      %282 = ascvec.and %278, %265, %263 : <ui16>, <ui16> -> <ui16>
                      %283 = ascvec.shl %281, 8, %263 : <ui16> -> <ui16>
                      %284 = ascvec.or %282, %283, %263 : <ui16>, <ui16> -> <ui16>
                      %285 = ascvec.reinterpret %284 : <ui16> -> <ui8>
                      %286 = ascvec.add_scalar %285, %c64_i8, %264 : <ui8>, i8
                      %287 = ascvec.min_scalar %286, %c123_i8, %264 : <ui8>, i8
                      %288 = arith.muli %arg14, %c512_i32 : i32
                      %289 = arith.index_cast %288 : i32 to index
                      %result0_13, %result1_14 = ascvec.loadalign_deintlv %arg11[%289] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %290 = ascvec.and %result1_14, %266, %264 : <ui8>, <ui8> -> <ui8>
                      %291 = ascvec.or %287, %290, %264 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg12[%269], %291, %264 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
                } else {
                  scf.for %arg9 = %c0_i64 to %204 step %c16_i64  : i64 {
                    %261 = arith.addi %arg9, %c16_i64 : i64
                    %262 = arith.cmpi sgt, %261, %204 : i64
                    %263 = arith.select %262, %204, %261 : i64
                    %264 = arith.muli %arg9, %c16_i64 : i64
                    ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %265 = arith.index_cast %264 : i64 to index
                    %266 = memref.load %alloc_10[%265] : memref<2048xui16, 1>
                    ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %267 = cannir.scalar_storage_cast %266 : ui16 -> i16
                    %268 = arith.extui %267 : i16 to i64
                    %269 = arith.addi %arg9, %c4_i64 : i64
                    %270 = scf.for %arg10 = %269 to %263 step %c4_i64 iter_args(%arg11 = %268) -> (i64)  : i64 {
                      %272 = arith.muli %arg10, %c16_i64 : i64
                      ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %273 = arith.index_cast %272 : i64 to index
                      %274 = memref.load %alloc_10[%273] : memref<2048xui16, 1>
                      ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %275 = cannir.scalar_storage_cast %274 : ui16 -> i16
                      %276 = arith.extui %275 : i16 to i64
                      %277 = arith.cmpi slt, %276, %arg11 : i64
                      %278 = arith.select %277, %276, %arg11 : i64
                      scf.yield %278 : i64
                    }
                    %271 = arith.cmpi sge, %270, %c14464_i64 : i64
                    scf.if %271 {
                      %272 = arith.trunci %arg9 : i64 to i32
                      %273 = arith.trunci %263 : i64 to i32
                      ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %274 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                      %275 = ascscalar.bitcast_memref %108 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                      ascvec.vf(%108, %274, %275, %alloc_6, %272, %273 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                      ^bb0(%arg10: memref<32768xbf16, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<?xui8, 1>, %arg13: memref<32768xui8, 1>, %arg14: i32, %arg15: i32):
                        %c512_i32 = arith.constant 512 : i32
                        %c1_i32_11 = arith.constant 1 : i32
                        %c123_i8 = arith.constant 123 : i8
                        %c64_i8 = arith.constant 64 : i8
                        %c128_i32 = arith.constant 128 : i32
                        %c256_i32 = arith.constant 256 : i32
                        %c31_i16 = arith.constant 31 : i16
                        %c-128_i8 = arith.constant -128 : i8
                        %c255_i16 = arith.constant 255 : i16
                        %276 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                        %277 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                        %278 = ascvec.duplicate_scalar %c255_i16, %276 : i16 -> <ui16>
                        %279 = ascvec.duplicate_scalar %c-128_i8, %277 : i8 -> <ui8>
                        %280 = ascvec.duplicate_scalar %c31_i16, %276 : i16 -> <ui16>
                        scf.for %arg16 = %arg14 to %arg15 step %c1_i32_11  : i32 {
                          %281 = arith.muli %arg16, %c256_i32 : i32
                          %282 = arith.index_cast %281 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg10[%282] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                          %283 = arith.muli %arg16, %c128_i32 : i32
                          %284 = arith.index_cast %283 : i32 to index
                          %285 = ascvec.loadalign %arg11[%284] : memref<?xui8, 1>, index -> <ui8>
                          %286 = ascvec.unpack %285 {part = "lower"} : <ui8> -> <ui16>
                          %287 = ascvec.and %286, %280, %276 : <ui16>, <ui16> -> <ui16>
                          %288 = ascvec.shr %286, 3, %276 : <ui16> -> <ui16>
                          %289 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                          %290 = ascvec.add %289, %287, %276 : <ui16>
                          %291 = ascvec.shr %290, 5, %276 : <ui16> -> <ui16>
                          %292 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                          %293 = ascvec.add %292, %288, %276 : <ui16>
                          %294 = ascvec.shr %293, 5, %276 : <ui16> -> <ui16>
                          %295 = ascvec.and %291, %278, %276 : <ui16>, <ui16> -> <ui16>
                          %296 = ascvec.shl %294, 8, %276 : <ui16> -> <ui16>
                          %297 = ascvec.or %295, %296, %276 : <ui16>, <ui16> -> <ui16>
                          %298 = ascvec.reinterpret %297 : <ui16> -> <ui8>
                          %299 = ascvec.add_scalar %298, %c64_i8, %277 : <ui8>, i8
                          %300 = ascvec.min_scalar %299, %c123_i8, %277 : <ui8>, i8
                          %301 = arith.muli %arg16, %c512_i32 : i32
                          %302 = arith.index_cast %301 : i32 to index
                          %result0_12, %result1_13 = ascvec.loadalign_deintlv %arg12[%302] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                          %303 = ascvec.and %result1_13, %279, %277 : <ui8>, <ui8> -> <ui8>
                          %304 = ascvec.or %300, %303, %277 : <ui8>, <ui8> -> <ui8>
                          ascvec.storealign %arg13[%282], %304, %277 : memref<32768xui8, 1>, index, <ui8>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
                    } else {
                      %272 = arith.cmpi sge, %270, %c13568_i64 : i64
                      scf.if %272 {
                        %273 = arith.trunci %arg9 : i64 to i32
                        %274 = arith.trunci %263 : i64 to i32
                        ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %275 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                        %276 = ascscalar.bitcast_memref %108 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%108, %275, %276, %alloc_6, %273, %274 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg10: memref<32768xbf16, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<?xui8, 1>, %arg13: memref<32768xui8, 1>, %arg14: i32, %arg15: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_11 = arith.constant 1 : i32
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
                          %277 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %278 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %279 = ascvec.duplicate_scalar %c255_i16, %277 : i16 -> <ui16>
                          %280 = ascvec.duplicate_scalar %c-128_i8, %278 : i8 -> <ui8>
                          %281 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %282 = ascvec.shr %281, 1, %278 : <ui8> -> <ui8>
                          %283 = ascvec.shr %282, 2, %278 : <ui8> -> <ui8>
                          %284 = ascvec.shl %283, 2, %278 : <ui8> -> <ui8>
                          %285 = ascvec.duplicate_scalar %c1_i8, %278 : i8 -> <ui8>
                          %286 = ascvec.and %281, %285, %278 : <ui8>, <ui8> -> <ui8>
                          %287 = ascvec.add_scalar %282, %c3_i8, %278 : <ui8>, i8
                          %288 = ascvec.add %287, %286, %278 : <ui8>
                          %289 = ascvec.duplicate_scalar %c3_i8, %278 : i8 -> <ui8>
                          %290 = ascvec.and %288, %289, %278 : <ui8>, <ui8> -> <ui8>
                          %291 = ascvec.or %284, %290, %278 : <ui8>, <ui8> -> <ui8>
                          %292 = ascvec.duplicate_scalar %c4095_i16, %277 : i16 -> <ui16>
                          %293 = ascvec.duplicate_scalar %c32767_i16, %277 : i16 -> <ui16>
                          %294 = ascvec.duplicate_scalar %cst, %277 : bf16 -> <bf16>
                          scf.for %arg16 = %arg14 to %arg15 step %c1_i32_11  : i32 {
                            %295 = arith.muli %arg16, %c256_i32 : i32
                            %296 = arith.index_cast %295 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg10[%296] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %297 = arith.muli %arg16, %c128_i32 : i32
                            %298 = arith.index_cast %297 : i32 to index
                            %299 = ascvec.loadalign %arg11[%298] : memref<?xui8, 1>, index -> <ui8>
                            %300 = ascvec.gather_reg %299, %291 : <ui8>, <ui8> -> <ui8>
                            %301 = ascvec.reinterpret %300 : <ui8> -> <ui16>
                            %302 = ascvec.shr %301, 1, %277 : <ui16> -> <ui16>
                            %303 = ascvec.and %302, %292, %277 : <ui16>, <ui16> -> <ui16>
                            %304 = ascvec.shr %301, 4, %277 : <ui16> -> <ui16>
                            %305 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %306 = ascvec.and %305, %293, %277 : <ui16>, <ui16> -> <ui16>
                            %307 = ascvec.cmp_lt_scalar %306, %c14464_i16, %277 : <ui16>, i16 -> !ascvec.mask
                            %308 = ascvec.shr %303, 7, %277 : <ui16> -> <ui16>
                            %309 = ascvec.add %306, %308, %277 : <ui16>
                            %310 = ascvec.shr %309, 5, %277 : <ui16> -> <ui16>
                            %311 = ascvec.reinterpret %306 : <ui16> -> <bf16>
                            %312 = ascvec.mul %311, %294, %307 : <bf16>
                            %313 = ascvec.cast %312, %277 {rounding = 4 : i32} : <bf16> -> <f16>
                            %314 = ascvec.cast %313, %277 {rounding = 4 : i32} : <f16> -> <i16>
                            %315 = ascvec.reinterpret %314 : <i16> -> <ui16>
                            %316 = ascvec.add %315, %303, %277 : <ui16>
                            %317 = ascvec.shr %316, 12, %277 : <ui16> -> <ui16>
                            %318 = ascvec.add_scalar %317, %c448_i16, %277 : <ui16>, i16
                            %319 = ascvec.select %318, %310, %307 : <ui16>
                            %320 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %321 = ascvec.and %320, %293, %277 : <ui16>, <ui16> -> <ui16>
                            %322 = ascvec.cmp_lt_scalar %321, %c14464_i16, %277 : <ui16>, i16 -> !ascvec.mask
                            %323 = ascvec.shr %304, 7, %277 : <ui16> -> <ui16>
                            %324 = ascvec.add %321, %323, %277 : <ui16>
                            %325 = ascvec.shr %324, 5, %277 : <ui16> -> <ui16>
                            %326 = ascvec.reinterpret %321 : <ui16> -> <bf16>
                            %327 = ascvec.mul %326, %294, %322 : <bf16>
                            %328 = ascvec.cast %327, %277 {rounding = 4 : i32} : <bf16> -> <f16>
                            %329 = ascvec.cast %328, %277 {rounding = 4 : i32} : <f16> -> <i16>
                            %330 = ascvec.reinterpret %329 : <i16> -> <ui16>
                            %331 = ascvec.add %330, %304, %277 : <ui16>
                            %332 = ascvec.shr %331, 12, %277 : <ui16> -> <ui16>
                            %333 = ascvec.add_scalar %332, %c448_i16, %277 : <ui16>, i16
                            %334 = ascvec.select %333, %325, %322 : <ui16>
                            %335 = ascvec.and %319, %279, %277 : <ui16>, <ui16> -> <ui16>
                            %336 = ascvec.shl %334, 8, %277 : <ui16> -> <ui16>
                            %337 = ascvec.or %335, %336, %277 : <ui16>, <ui16> -> <ui16>
                            %338 = ascvec.reinterpret %337 : <ui16> -> <ui8>
                            %339 = ascvec.add_scalar %338, %c64_i8, %278 : <ui8>, i8
                            %340 = ascvec.min_scalar %339, %c123_i8, %278 : <ui8>, i8
                            %341 = arith.muli %arg16, %c512_i32 : i32
                            %342 = arith.index_cast %341 : i32 to index
                            %result0_12, %result1_13 = ascvec.loadalign_deintlv %arg12[%342] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %343 = ascvec.and %result1_13, %280, %278 : <ui8>, <ui8> -> <ui8>
                            %344 = ascvec.or %340, %343, %278 : <ui8>, <ui8> -> <ui8>
                            ascvec.storealign %arg13[%296], %344, %278 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
                      } else {
                        %273 = arith.trunci %arg9 : i64 to i32
                        %274 = arith.trunci %263 : i64 to i32
                        ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %275 = ascscalar.bitcast_memref %108 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%108, %alloc_9, %275, %256, %alloc_6, %273, %274 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg10: memref<32768xbf16, 1>, %arg11: memref<8192xi32, 1>, %arg12: memref<?xui8, 1>, %arg13: memref<?xui16, 1>, %arg14: memref<32768xui8, 1>, %arg15: i32, %arg16: i32):
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
                          %c4_i32_11 = arith.constant 4 : i32
                          %c2_i32_12 = arith.constant 2 : i32
                          %c1_i32_13 = arith.constant 1 : i32
                          %c7_i32 = arith.constant 7 : i32
                          %c32_i32 = arith.constant 32 : i32
                          %c0_i32_14 = arith.constant 0 : i32
                          %c32767_i16 = arith.constant 32767 : i16
                          %c0_15 = arith.constant 0 : index
                          %c256_i32 = arith.constant 256 : i32
                          %c3_i8 = arith.constant 3 : i8
                          %c-128_i8 = arith.constant -128 : i8
                          %c0_i8 = arith.constant 0 : i8
                          %c8_i32 = arith.constant 8 : i32
                          %276 = ascvec.create_mask : !ascvec.mask
                          %277 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %278 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                          %279 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %280 = ascvec.duplicate_scalar %c-128_i8, %278 : i8 -> <ui8>
                          %281 = ascvec.shr %279, 5, %278 : <ui8> -> <ui8>
                          %282 = ascvec.duplicate_scalar %c3_i8, %278 : i8 -> <ui8>
                          %283 = ascvec.and %279, %282, %278 : <ui8>, <ui8> -> <ui8>
                          %284 = ascvec.shl %283, 6, %278 : <ui8> -> <ui8>
                          %285 = ascvec.shr %279, 2, %278 : <ui8> -> <ui8>
                          %286 = ascvec.or %284, %285, %278 : <ui8>, <ui8> -> <ui8>
                          scf.for %arg17 = %arg15 to %arg16 step %c1_i32_13  : i32 {
                            %287 = arith.muli %arg17, %c256_i32 : i32
                            %288 = arith.index_cast %287 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg10[%288] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %289 = ascvec.duplicate_scalar %c32767_i16, %277 : i16 -> <ui16>
                            %290 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %291 = ascvec.and %290, %289, %277 : <ui16>, <ui16> -> <ui16>
                            %292 = ascvec.reinterpret %291 : <ui16> -> <bf16>
                            %293 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %294 = ascvec.and %293, %289, %277 : <ui16>, <ui16> -> <ui16>
                            %295 = ascvec.reinterpret %294 : <ui16> -> <bf16>
                            %296 = ascvec.cast %292, %277 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %297 = ascvec.cast %292, %277 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %298 = ascvec.cast %295, %277 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %299 = ascvec.cast %295, %277 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %300 = ascvec.arange %c0_i32_14 : i32 -> <ui32>
                            %301 = ascvec.shl %300, 2, %276 : <ui32> -> <ui32>
                            %302 = ascvec.add_scalar %301, %c0_i32_14, %276 : <ui32>, i32
                            %303 = ascvec.shr %302, 3, %276 : <ui32> -> <ui32>
                            %304 = arith.muli %arg17, %c32_i32 : i32
                            %305 = ascvec.add_scalar %303, %304, %276 : <ui32>, i32
                            %306 = ascvec.gather %arg11[%c0_15], %305, %276 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %307 = ascvec.reinterpret %306 : <i32> -> <ui32>
                            %308 = ascvec.duplicate_scalar %c7_i32, %276 : i32 -> <ui32>
                            %309 = ascvec.and %302, %308, %276 : <ui32>, <ui32> -> <ui32>
                            %310 = ascvec.shl %307, 11, %276 : <ui32> -> <ui32>
                            %311 = ascvec.shr %307, 21, %276 : <ui32> -> <ui32>
                            %312 = ascvec.or %310, %311, %276 : <ui32>, <ui32> -> <ui32>
                            %313 = ascvec.shr %312, 3, %276 : <ui32> -> <ui32>
                            %314 = ascvec.shl %312, 29, %276 : <ui32> -> <ui32>
                            %315 = ascvec.or %313, %314, %276 : <ui32>, <ui32> -> <ui32>
                            %316 = ascvec.duplicate_scalar %c1_i32_13, %276 : i32 -> <ui32>
                            %317 = ascvec.and %309, %316, %276 : <ui32>, <ui32> -> <ui32>
                            %318 = ascvec.cmp_ne_scalar %317, %c0_i32_14, %276 : <ui32>, i32 -> !ascvec.mask
                            %319 = ascvec.select %315, %312, %318 : <ui32>
                            %320 = ascvec.shr %319, 8, %276 : <ui32> -> <ui32>
                            %321 = ascvec.shl %319, 24, %276 : <ui32> -> <ui32>
                            %322 = ascvec.or %320, %321, %276 : <ui32>, <ui32> -> <ui32>
                            %323 = ascvec.duplicate_scalar %c2_i32_12, %276 : i32 -> <ui32>
                            %324 = ascvec.and %309, %323, %276 : <ui32>, <ui32> -> <ui32>
                            %325 = ascvec.cmp_ne_scalar %324, %c0_i32_14, %276 : <ui32>, i32 -> !ascvec.mask
                            %326 = ascvec.select %322, %319, %325 : <ui32>
                            %327 = ascvec.shr %326, 16, %276 : <ui32> -> <ui32>
                            %328 = ascvec.shl %326, 16, %276 : <ui32> -> <ui32>
                            %329 = ascvec.or %327, %328, %276 : <ui32>, <ui32> -> <ui32>
                            %330 = ascvec.duplicate_scalar %c4_i32_11, %276 : i32 -> <ui32>
                            %331 = ascvec.and %309, %330, %276 : <ui32>, <ui32> -> <ui32>
                            %332 = ascvec.cmp_ne_scalar %331, %c0_i32_14, %276 : <ui32>, i32 -> !ascvec.mask
                            %333 = ascvec.select %329, %326, %332 : <ui32>
                            %334 = ascvec.duplicate_scalar %c65535_i32, %276 : i32 -> <ui32>
                            %335 = ascvec.and %333, %334, %276 : <ui32>, <ui32> -> <ui32>
                            %336 = ascvec.add_scalar %301, %c1_i32_13, %276 : <ui32>, i32
                            %337 = ascvec.shr %336, 3, %276 : <ui32> -> <ui32>
                            %338 = ascvec.add_scalar %337, %304, %276 : <ui32>, i32
                            %339 = ascvec.gather %arg11[%c0_15], %338, %276 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %340 = ascvec.reinterpret %339 : <i32> -> <ui32>
                            %341 = ascvec.and %336, %308, %276 : <ui32>, <ui32> -> <ui32>
                            %342 = ascvec.shl %340, 11, %276 : <ui32> -> <ui32>
                            %343 = ascvec.shr %340, 21, %276 : <ui32> -> <ui32>
                            %344 = ascvec.or %342, %343, %276 : <ui32>, <ui32> -> <ui32>
                            %345 = ascvec.shr %344, 3, %276 : <ui32> -> <ui32>
                            %346 = ascvec.shl %344, 29, %276 : <ui32> -> <ui32>
                            %347 = ascvec.or %345, %346, %276 : <ui32>, <ui32> -> <ui32>
                            %348 = ascvec.and %341, %316, %276 : <ui32>, <ui32> -> <ui32>
                            %349 = ascvec.cmp_ne_scalar %348, %c0_i32_14, %276 : <ui32>, i32 -> !ascvec.mask
                            %350 = ascvec.select %347, %344, %349 : <ui32>
                            %351 = ascvec.shr %350, 8, %276 : <ui32> -> <ui32>
                            %352 = ascvec.shl %350, 24, %276 : <ui32> -> <ui32>
                            %353 = ascvec.or %351, %352, %276 : <ui32>, <ui32> -> <ui32>
                            %354 = ascvec.and %341, %323, %276 : <ui32>, <ui32> -> <ui32>
                            %355 = ascvec.cmp_ne_scalar %354, %c0_i32_14, %276 : <ui32>, i32 -> !ascvec.mask
                            %356 = ascvec.select %353, %350, %355 : <ui32>
                            %357 = ascvec.shr %356, 16, %276 : <ui32> -> <ui32>
                            %358 = ascvec.shl %356, 16, %276 : <ui32> -> <ui32>
                            %359 = ascvec.or %357, %358, %276 : <ui32>, <ui32> -> <ui32>
                            %360 = ascvec.and %341, %330, %276 : <ui32>, <ui32> -> <ui32>
                            %361 = ascvec.cmp_ne_scalar %360, %c0_i32_14, %276 : <ui32>, i32 -> !ascvec.mask
                            %362 = ascvec.select %359, %356, %361 : <ui32>
                            %363 = ascvec.and %362, %334, %276 : <ui32>, <ui32> -> <ui32>
                            %364 = ascvec.add_scalar %301, %c2_i32_12, %276 : <ui32>, i32
                            %365 = ascvec.shr %364, 3, %276 : <ui32> -> <ui32>
                            %366 = ascvec.add_scalar %365, %304, %276 : <ui32>, i32
                            %367 = ascvec.gather %arg11[%c0_15], %366, %276 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %368 = ascvec.reinterpret %367 : <i32> -> <ui32>
                            %369 = ascvec.and %364, %308, %276 : <ui32>, <ui32> -> <ui32>
                            %370 = ascvec.shl %368, 11, %276 : <ui32> -> <ui32>
                            %371 = ascvec.shr %368, 21, %276 : <ui32> -> <ui32>
                            %372 = ascvec.or %370, %371, %276 : <ui32>, <ui32> -> <ui32>
                            %373 = ascvec.shr %372, 3, %276 : <ui32> -> <ui32>
                            %374 = ascvec.shl %372, 29, %276 : <ui32> -> <ui32>
                            %375 = ascvec.or %373, %374, %276 : <ui32>, <ui32> -> <ui32>
                            %376 = ascvec.and %369, %316, %276 : <ui32>, <ui32> -> <ui32>
                            %377 = ascvec.cmp_ne_scalar %376, %c0_i32_14, %276 : <ui32>, i32 -> !ascvec.mask
                            %378 = ascvec.select %375, %372, %377 : <ui32>
                            %379 = ascvec.shr %378, 8, %276 : <ui32> -> <ui32>
                            %380 = ascvec.shl %378, 24, %276 : <ui32> -> <ui32>
                            %381 = ascvec.or %379, %380, %276 : <ui32>, <ui32> -> <ui32>
                            %382 = ascvec.and %369, %323, %276 : <ui32>, <ui32> -> <ui32>
                            %383 = ascvec.cmp_ne_scalar %382, %c0_i32_14, %276 : <ui32>, i32 -> !ascvec.mask
                            %384 = ascvec.select %381, %378, %383 : <ui32>
                            %385 = ascvec.shr %384, 16, %276 : <ui32> -> <ui32>
                            %386 = ascvec.shl %384, 16, %276 : <ui32> -> <ui32>
                            %387 = ascvec.or %385, %386, %276 : <ui32>, <ui32> -> <ui32>
                            %388 = ascvec.and %369, %330, %276 : <ui32>, <ui32> -> <ui32>
                            %389 = ascvec.cmp_ne_scalar %388, %c0_i32_14, %276 : <ui32>, i32 -> !ascvec.mask
                            %390 = ascvec.select %387, %384, %389 : <ui32>
                            %391 = ascvec.and %390, %334, %276 : <ui32>, <ui32> -> <ui32>
                            %392 = ascvec.add_scalar %301, %c3_i32, %276 : <ui32>, i32
                            %393 = ascvec.shr %392, 3, %276 : <ui32> -> <ui32>
                            %394 = ascvec.add_scalar %393, %304, %276 : <ui32>, i32
                            %395 = ascvec.gather %arg11[%c0_15], %394, %276 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %396 = ascvec.reinterpret %395 : <i32> -> <ui32>
                            %397 = ascvec.and %392, %308, %276 : <ui32>, <ui32> -> <ui32>
                            %398 = ascvec.shl %396, 11, %276 : <ui32> -> <ui32>
                            %399 = ascvec.shr %396, 21, %276 : <ui32> -> <ui32>
                            %400 = ascvec.or %398, %399, %276 : <ui32>, <ui32> -> <ui32>
                            %401 = ascvec.shr %400, 3, %276 : <ui32> -> <ui32>
                            %402 = ascvec.shl %400, 29, %276 : <ui32> -> <ui32>
                            %403 = ascvec.or %401, %402, %276 : <ui32>, <ui32> -> <ui32>
                            %404 = ascvec.and %397, %316, %276 : <ui32>, <ui32> -> <ui32>
                            %405 = ascvec.cmp_ne_scalar %404, %c0_i32_14, %276 : <ui32>, i32 -> !ascvec.mask
                            %406 = ascvec.select %403, %400, %405 : <ui32>
                            %407 = ascvec.shr %406, 8, %276 : <ui32> -> <ui32>
                            %408 = ascvec.shl %406, 24, %276 : <ui32> -> <ui32>
                            %409 = ascvec.or %407, %408, %276 : <ui32>, <ui32> -> <ui32>
                            %410 = ascvec.and %397, %323, %276 : <ui32>, <ui32> -> <ui32>
                            %411 = ascvec.cmp_ne_scalar %410, %c0_i32_14, %276 : <ui32>, i32 -> !ascvec.mask
                            %412 = ascvec.select %409, %406, %411 : <ui32>
                            %413 = ascvec.shr %412, 16, %276 : <ui32> -> <ui32>
                            %414 = ascvec.shl %412, 16, %276 : <ui32> -> <ui32>
                            %415 = ascvec.or %413, %414, %276 : <ui32>, <ui32> -> <ui32>
                            %416 = ascvec.and %397, %330, %276 : <ui32>, <ui32> -> <ui32>
                            %417 = ascvec.cmp_ne_scalar %416, %c0_i32_14, %276 : <ui32>, i32 -> !ascvec.mask
                            %418 = ascvec.select %415, %412, %417 : <ui32>
                            %419 = ascvec.and %418, %334, %276 : <ui32>, <ui32> -> <ui32>
                            %420 = ascvec.reinterpret %296 : <f32> -> <ui32>
                            %421 = ascvec.reinterpret %420 : <ui32> -> <f32>
                            %422 = ascvec.max_scalar %421, %cst, %276 : <f32>, f32
                            %423 = ascvec.add %421, %422, %276 : <f32>
                            %424 = ascvec.reinterpret %423 : <f32> -> <ui32>
                            %425 = ascvec.shr %424, 5, %276 : <ui32> -> <ui32>
                            %426 = ascvec.add %425, %335, %276 : <ui32>
                            %427 = ascvec.reinterpret %298 : <f32> -> <ui32>
                            %428 = ascvec.reinterpret %427 : <ui32> -> <f32>
                            %429 = ascvec.max_scalar %428, %cst, %276 : <f32>, f32
                            %430 = ascvec.add %428, %429, %276 : <f32>
                            %431 = ascvec.reinterpret %430 : <f32> -> <ui32>
                            %432 = ascvec.shr %431, 5, %276 : <ui32> -> <ui32>
                            %433 = ascvec.add %432, %363, %276 : <ui32>
                            %434 = ascvec.reinterpret %297 : <f32> -> <ui32>
                            %435 = ascvec.reinterpret %434 : <ui32> -> <f32>
                            %436 = ascvec.max_scalar %435, %cst, %276 : <f32>, f32
                            %437 = ascvec.add %435, %436, %276 : <f32>
                            %438 = ascvec.reinterpret %437 : <f32> -> <ui32>
                            %439 = ascvec.shr %438, 5, %276 : <ui32> -> <ui32>
                            %440 = ascvec.add %439, %391, %276 : <ui32>
                            %441 = ascvec.reinterpret %299 : <f32> -> <ui32>
                            %442 = ascvec.reinterpret %441 : <ui32> -> <f32>
                            %443 = ascvec.max_scalar %442, %cst, %276 : <f32>, f32
                            %444 = ascvec.add %442, %443, %276 : <f32>
                            %445 = ascvec.reinterpret %444 : <f32> -> <ui32>
                            %446 = ascvec.shr %445, 5, %276 : <ui32> -> <ui32>
                            %447 = ascvec.add %446, %419, %276 : <ui32>
                            %448 = ascvec.reinterpret %426 : <ui32> -> <ui16>
                            %449 = ascvec.reinterpret %433 : <ui32> -> <ui16>
                            %result0_16, %result1_17 = ascvec.deintlv %448, %449 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %450 = ascvec.reinterpret %440 : <ui32> -> <ui16>
                            %451 = ascvec.reinterpret %447 : <ui32> -> <ui16>
                            %result0_18, %result1_19 = ascvec.deintlv %450, %451 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %452 = ascvec.reinterpret %result1_17 : <ui16> -> <ui8>
                            %453 = ascvec.reinterpret %result1_19 : <ui16> -> <ui8>
                            %result0_20, %result1_21 = ascvec.deintlv %452, %453 : <ui8>, <ui8> -> <ui8>, <ui8>
                            %454 = ascvec.gather_reg %result0_20, %286 : <ui8>, <ui8> -> <ui8>
                            %455 = ascvec.add_scalar %454, %c60_i8, %278 : <ui8>, i8
                            %456 = ascvec.cmp_ne_scalar %455, %c-1_i8, %278 : <ui8>, i8 -> !ascvec.mask
                            %457 = ascvec.min_scalar %455, %c123_i8, %456 : <ui8>, i8
                            %458 = arith.muli %arg17, %c512_i32 : i32
                            %459 = arith.index_cast %458 : i32 to index
                            %result0_22, %result1_23 = ascvec.loadalign_deintlv %arg12[%459] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %460 = ascvec.and %result1_23, %280, %278 : <ui8>, <ui8> -> <ui8>
                            %461 = ascvec.or %457, %460, %278 : <ui8>, <ui8> -> <ui8>
                            %462 = ascvec.arange %c0_i16 : i16 -> <ui16>
                            %463 = arith.muli %arg17, %c16_i32 : i32
                            %464 = arith.trunci %463 : i32 to i16
                            %465 = ascvec.add_scalar %462, %464, %277 : <ui16>, i16
                            %466 = ascvec.gather %arg13[%c0_15], %465, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                            %467 = ascvec.pack %466 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                            %468 = ascvec.gather_reg %467, %281 : <ui8>, <ui8> -> <ui8>
                            %469 = ascvec.cmp_eq_scalar %468, %c-1_i8, %278 : <ui8>, i8 -> !ascvec.mask
                            %470 = ascvec.duplicate_scalar %c127_i8, %469 merge %461 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                            ascvec.storealign %arg14[%288], %470, %278 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
                      }
                    }
                  }
                }
              } else {
                %204 = arith.divsi %67, %c256_i64 : i64
                %205 = arith.addi %67, %c1023_i64 : i64
                %206 = arith.divsi %205, %c1024_i64 : i64
                %207 = arith.trunci %1 : i64 to i32
                %208 = arith.extui %207 : i32 to i64
                %209 = arith.subi %1, %208 : i64
                %210 = arith.remsi %209, %c4294967296_i64 : i64
                %211 = arith.cmpi ne, %210, %c0_i64 : i64
                %212 = arith.cmpi slt, %210, %c0_i64 : i64
                %213 = arith.andi %211, %212 : i1
                %214 = arith.divsi %209, %c4294967296_i64 : i64
                %215 = arith.extui %213 : i1 to i64
                %216 = arith.subi %214, %215 : i64
                %217 = arith.trunci %216 : i64 to i32
                %218 = arith.divui %109, %c32_i64 : i64
                %219 = arith.trunci %218 : i64 to i32
                %220 = arith.extui %219 : i32 to i64
                %221 = arith.subi %218, %220 : i64
                %222 = arith.remsi %221, %c4294967296_i64 : i64
                %223 = arith.cmpi ne, %222, %c0_i64 : i64
                %224 = arith.cmpi slt, %222, %c0_i64 : i64
                %225 = arith.andi %223, %224 : i1
                %226 = arith.divsi %221, %c4294967296_i64 : i64
                %227 = arith.extui %225 : i1 to i64
                %228 = arith.subi %226, %227 : i64
                %229 = arith.trunci %228 : i64 to i32
                %230 = arith.trunci %4 : i64 to i32
                %231 = arith.extui %230 : i32 to i64
                %232 = arith.subi %4, %231 : i64
                %233 = arith.remsi %232, %c4294967296_i64 : i64
                %234 = arith.cmpi ne, %233, %c0_i64 : i64
                %235 = arith.cmpi slt, %233, %c0_i64 : i64
                %236 = arith.andi %234, %235 : i1
                %237 = arith.divsi %232, %c4294967296_i64 : i64
                %238 = arith.extui %236 : i1 to i64
                %239 = arith.subi %237, %238 : i64
                %240 = arith.trunci %239 : i64 to i32
                %241 = arith.divui %230, %c4_i32 : i32
                %242 = arith.remui %240, %c4_i32 : i32
                %243 = arith.muli %242, %c1073741824_i32 : i32
                %244 = arith.addi %241, %243 : i32
                %245 = arith.divui %240, %c4_i32 : i32
                %246 = arith.muli %206, %c32_i64 : i64
                %247 = arith.divui %218, %c16384_i64 : i64
                %248 = arith.addi %218, %246 : i64
                %249 = arith.subi %248, %c1_i64 : i64
                %250 = arith.divui %249, %c16384_i64 : i64
                %251 = arith.cmpi eq, %247, %250 : i64
                scf.if %251 {
                  %261 = arith.trunci %247 : i64 to i32
                  %262 = arith.extui %261 : i32 to i64
                  %263 = arith.subi %247, %262 : i64
                  %264 = arith.remsi %263, %c4294967296_i64 : i64
                  %265 = arith.cmpi ne, %264, %c0_i64 : i64
                  %266 = arith.cmpi slt, %264, %c0_i64 : i64
                  %267 = arith.andi %265, %266 : i1
                  %268 = arith.divsi %263, %c4294967296_i64 : i64
                  %269 = arith.extui %267 : i1 to i64
                  %270 = arith.subi %268, %269 : i64
                  %271 = arith.trunci %270 : i64 to i32
                  %272 = arith.addi %244, %261 : i32
                  %273 = arith.addi %245, %271 : i32
                  %274 = arith.cmpi ult, %272, %244 : i32
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
                  %290 = arith.xori %276, %217 : i32
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
                  %304 = arith.addi %207, %c-1640531527_i32 : i32
                  %305 = arith.addi %217, %c-1150833019_i32 : i32
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
                  %321 = arith.xori %289, %207 : i32
                  %322 = arith.addi %207, %c1013904242_i32 : i32
                  %323 = arith.xori %293, %322 : i32
                  %324 = arith.addi %217, %c1993301258_i32 : i32
                  %325 = arith.xori %320, %324 : i32
                  ascsimt.vf(%219, %321, %305, %323, %325, %207, %217, %310, %alloc_9, %246 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: memref<8192xi32, 1>, %arg18: i64):
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
                    %c16384_i32 = arith.constant 16384 : i32
                    %326 = ascsimt.thread_idx {ascsimt.scope = 8 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %327 = arith.extsi %326 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg19 = %327 to %arg18 step %c256_i64_11  : i64 {
                      %328 = arith.trunci %arg19 : i64 to i32
                      %329 = arith.addi %arg9, %328 : i32
                      %330 = arith.remui %329, %c16384_i32 : i32
                      %331 = arith.xori %330, %arg10 : i32
                      %332 = arith.muli %331, %c-766435501_i32 : i32
                      %333 = "ascsimt.bitcast"(%331) : (i32) -> ui32
                      %334 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %335 = ascsimt.math_binary %333, %334, "mulhi" : ui32, ui32 -> ui32
                      %336 = "ascsimt.bitcast"(%335) : (ui32) -> i32
                      %337 = arith.xori %336, %arg11 : i32
                      %338 = arith.muli %337, %c-845247145_i32 : i32
                      %339 = "ascsimt.bitcast"(%337) : (i32) -> ui32
                      %340 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %341 = ascsimt.math_binary %339, %340, "mulhi" : ui32, ui32 -> ui32
                      %342 = "ascsimt.bitcast"(%341) : (ui32) -> i32
                      %343 = arith.xori %342, %arg12 : i32
                      %344 = arith.xori %332, %arg13 : i32
                      %345:4 = scf.for %arg20 = %c3_i32 to %c10_i32 step %c1_i32_16 iter_args(%arg21 = %343, %arg22 = %338, %arg23 = %344, %arg24 = %arg16) -> (i32, i32, i32, i32)  : i32 {
                        %354 = arith.muli %arg21, %c-766435501_i32 : i32
                        %355 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %356 = ascsimt.math_binary %355, %334, "mulhi" : ui32, ui32 -> ui32
                        %357 = "ascsimt.bitcast"(%356) : (ui32) -> i32
                        %358 = arith.muli %arg23, %c-845247145_i32 : i32
                        %359 = "ascsimt.bitcast"(%arg23) : (i32) -> ui32
                        %360 = ascsimt.math_binary %359, %340, "mulhi" : ui32, ui32 -> ui32
                        %361 = "ascsimt.bitcast"(%360) : (ui32) -> i32
                        %362 = arith.muli %arg20, %c-1640531527_i32_18 : i32
                        %363 = arith.addi %arg14, %362 : i32
                        %364 = arith.muli %arg20, %c-1150833019_i32_17 : i32
                        %365 = arith.addi %arg15, %364 : i32
                        %366 = arith.xori %361, %arg22 : i32
                        %367 = arith.xori %366, %363 : i32
                        %368 = arith.xori %357, %arg24 : i32
                        %369 = arith.xori %368, %365 : i32
                        scf.yield %367, %358, %369, %354 : i32, i32, i32, i32
                      }
                      %346 = arith.muli %arg19, %c4_i64_15 : i64
                      %347 = arith.index_cast %346 : i64 to index
                      ascsimt.store %arg17[%347], %345#0 : memref<8192xi32, 1>, i32
                      %348 = arith.addi %346, %c1_i64_14 : i64
                      %349 = arith.index_cast %348 : i64 to index
                      ascsimt.store %arg17[%349], %345#1 : memref<8192xi32, 1>, i32
                      %350 = arith.addi %346, %c2_i64_13 : i64
                      %351 = arith.index_cast %350 : i64 to index
                      ascsimt.store %arg17[%351], %345#2 : memref<8192xi32, 1>, i32
                      %352 = arith.addi %346, %c3_i64_12 : i64
                      %353 = arith.index_cast %352 : i64 to index
                      ascsimt.store %arg17[%353], %345#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%219, %229, %244, %245, %207, %217, %alloc_9, %246 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
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
                    %c16384_i32 = arith.constant 16384 : i32
                    %261 = ascsimt.thread_idx {ascsimt.scope = 9 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %262 = arith.extsi %261 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg17 = %262 to %arg16 step %c256_i64_11  : i64 {
                      %263 = arith.trunci %arg17 : i64 to i32
                      %264 = arith.addi %arg9, %263 : i32
                      %265 = arith.cmpi ult, %264, %arg9 : i32
                      %266 = arith.extui %265 : i1 to i32
                      %267 = arith.addi %arg10, %266 : i32
                      %268 = arith.divui %264, %c16384_i32 : i32
                      %269 = arith.remui %267, %c16384_i32 : i32
                      %270 = arith.muli %269, %c262144_i32 : i32
                      %271 = arith.addi %268, %270 : i32
                      %272 = arith.divui %267, %c16384_i32 : i32
                      %273 = arith.addi %arg11, %271 : i32
                      %274 = arith.addi %arg12, %272 : i32
                      %275 = arith.cmpi ult, %273, %arg11 : i32
                      %276 = arith.extui %275 : i1 to i32
                      %277 = arith.addi %274, %276 : i32
                      %278 = arith.remui %264, %c16384_i32 : i32
                      %279:4 = scf.for %arg18 = %c0_i32_17 to %c10_i32 step %c1_i32_16 iter_args(%arg19 = %c0_i32_17, %arg20 = %278, %arg21 = %273, %arg22 = %277) -> (i32, i32, i32, i32)  : i32 {
                        %288 = arith.muli %arg19, %c-766435501_i32 : i32
                        %289 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %290 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %291 = ascsimt.math_binary %289, %290, "mulhi" : ui32, ui32 -> ui32
                        %292 = "ascsimt.bitcast"(%291) : (ui32) -> i32
                        %293 = arith.muli %arg21, %c-845247145_i32 : i32
                        %294 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %295 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %296 = ascsimt.math_binary %294, %295, "mulhi" : ui32, ui32 -> ui32
                        %297 = "ascsimt.bitcast"(%296) : (ui32) -> i32
                        %298 = arith.muli %arg18, %c-1640531527_i32_19 : i32
                        %299 = arith.addi %arg13, %298 : i32
                        %300 = arith.muli %arg18, %c-1150833019_i32_18 : i32
                        %301 = arith.addi %arg14, %300 : i32
                        %302 = arith.xori %297, %arg20 : i32
                        %303 = arith.xori %302, %299 : i32
                        %304 = arith.xori %292, %arg22 : i32
                        %305 = arith.xori %304, %301 : i32
                        scf.yield %303, %293, %305, %288 : i32, i32, i32, i32
                      }
                      %280 = arith.muli %arg17, %c4_i64_15 : i64
                      %281 = arith.index_cast %280 : i64 to index
                      ascsimt.store %arg15[%281], %279#0 : memref<8192xi32, 1>, i32
                      %282 = arith.addi %280, %c1_i64_14 : i64
                      %283 = arith.index_cast %282 : i64 to index
                      ascsimt.store %arg15[%283], %279#1 : memref<8192xi32, 1>, i32
                      %284 = arith.addi %280, %c2_i64_13 : i64
                      %285 = arith.index_cast %284 : i64 to index
                      ascsimt.store %arg15[%285], %279#2 : memref<8192xi32, 1>, i32
                      %286 = arith.addi %280, %c3_i64_12 : i64
                      %287 = arith.index_cast %286 : i64 to index
                      ascsimt.store %arg15[%287], %279#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %252 = arith.trunci %49 : i64 to i32
                %253 = arith.divsi %252, %c2_i32 : i32
                %254 = arith.trunci %253 : i32 to i8
                %255 = arith.trunci %204 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %256 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%254, %alloc_7, %108, %256, %255, %alloc_10 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, i32, memref<2048xui16, 1>) {
                ^bb0(%arg9: i8, %arg10: memref<2048xui8, 1>, %arg11: memref<32768xbf16, 1>, %arg12: memref<?xui16, 1>, %arg13: i32, %arg14: memref<2048xui16, 1>):
                  %c1_i32_11 = arith.constant 1 : i32
                  %c0_i32_12 = arith.constant 0 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c0_13 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %261 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %262 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %263 = ascvec.duplicate_scalar %c32767_i16, %261 : i16 -> <ui16>
                  %264 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %265 = ascvec.shr %264, 4, %261 : <ui16> -> <ui16>
                  %266 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %267 = ascvec.duplicate_scalar %c1_i8, %262 : i8 -> <ui8>
                  %268 = ascvec.and %266, %267, %262 : <ui8>, <ui8> -> <ui8>
                  %269 = ascvec.cmp_lt_scalar %266, %c16_i8, %262 : <ui8>, i8 -> !ascvec.mask
                  %270 = ascvec.cmp_lt_scalar %268, %arg9, %262 : <ui8>, i8 -> !ascvec.mask
                  %271 = ascvec.mask_and %269, %270, %262
                  ascvec.store_unalign_begin %arg10 : memref<2048xui8, 1>
                  %272 = ascvec.duplicate_scalar %c-1_i16, %261 : i16 -> <ui16>
                  %273 = ascvec.duplicate_scalar %c0_i16, %261 : i16 -> <ui16>
                  %274:2 = scf.for %arg15 = %c0_i32_12 to %arg13 step %c1_i32_11 iter_args(%arg16 = %273, %arg17 = %272) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i32 {
                    %275 = arith.muli %arg15, %c256_i32 : i32
                    %276 = arith.index_cast %275 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg11[%276] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %277 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %278 = ascvec.and %277, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %279 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %280 = ascvec.and %279, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %281 = ascvec.max %278, %280, %261 : <ui16>
                    %282 = ascvec.reduce_max_datablock %281, %261 : <ui16>
                    %283 = ascvec.shr %282, 7, %261 : <ui16> -> <ui16>
                    %284 = ascvec.cmp_ge_scalar %283, %c255_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %285 = ascvec.max_scalar %283, %c15_i16, %261 : <ui16>, i16
                    %286 = ascvec.add_scalar %285, %c-15_i16, %261 : <ui16>, i16
                    %287 = ascvec.duplicate_scalar %c255_i16, %284 merge %286 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %288 = ascvec.duplicate_scalar %c254_i16, %261 : i16 -> <ui16>
                    %289 = ascvec.sub %288, %287, %261 : <ui16>
                    %290 = ascvec.cmp_ne_scalar %287, %c0_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %291 = ascvec.shl %289, 7, %290 : <ui16> -> <ui16>
                    %292 = ascvec.max %arg16, %287, %mask : <ui16>
                    %293 = ascvec.gather_reg %291, %265 : <ui16>, <ui16> -> <ui16>
                    %294 = ascvec.reinterpret %293 : <ui16> -> <bf16>
                    %295 = ascvec.mul %result0, %294, %261 : <bf16>
                    %296 = ascvec.mul %result1, %294, %261 : <bf16>
                    ascvec.storealign_intlv %arg11[%276], %295, %296 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %297 = ascvec.reinterpret %295 : <bf16> -> <ui16>
                    %298 = ascvec.and %297, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %299 = ascvec.reinterpret %296 : <bf16> -> <ui16>
                    %300 = ascvec.and %299, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %301 = ascvec.min %298, %300, %261 : <ui16>
                    %302 = ascvec.min %arg17, %301, %261 : <ui16>
                    %303 = arith.muli %arg15, %c16_i32 : i32
                    %304 = arith.index_cast %303 : i32 to index
                    ascvec.storealign %arg12[%304], %287, %mask : memref<?xui16, 1>, index, <ui16>
                    %305 = ascvec.reinterpret %287 : <ui16> -> <ui8>
                    %306 = ascvec.squeeze_store_reg %305, %271 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_13], %306 : memref<2048xui8, 1>, <ui8>
                    scf.yield %292, %302 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                  }
                  %c0_14 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg15 = %c0_14 to %c1 step %c1 {
                    %275 = ascvec.reduce_min %274#1, %261 : <ui16>
                    %276 = ascvec.reduce_max %274#0, %mask : <ui16>
                    %277 = ascvec.cmp_eq_scalar %276, %c255_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %278 = ascvec.duplicate_scalar %c0_i16, %277 merge %275 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.store_unalign_post %arg10[%c0_13] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg14[%c0_13], %278 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %257 = memref.load %alloc_10[%c0] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %258 = cannir.scalar_storage_cast %257 : ui16 -> i16
                %259 = arith.extui %258 : i16 to i64
                %260 = arith.cmpi sge, %259, %c14464_i64 : i64
                scf.if %260 {
                  ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %261 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                  %262 = ascscalar.bitcast_memref %108 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%108, %261, %262, %alloc_6, %255 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_11 = arith.constant 1 : i32
                    %c0_i32_12 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c128_i32 = arith.constant 128 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c31_i16 = arith.constant 31 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %263 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %264 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %265 = ascvec.duplicate_scalar %c255_i16, %263 : i16 -> <ui16>
                    %266 = ascvec.duplicate_scalar %c-128_i8, %264 : i8 -> <ui8>
                    %267 = ascvec.duplicate_scalar %c31_i16, %263 : i16 -> <ui16>
                    scf.for %arg14 = %c0_i32_12 to %arg13 step %c1_i32_11  : i32 {
                      %268 = arith.muli %arg14, %c256_i32 : i32
                      %269 = arith.index_cast %268 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg9[%269] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %270 = arith.muli %arg14, %c128_i32 : i32
                      %271 = arith.index_cast %270 : i32 to index
                      %272 = ascvec.loadalign %arg10[%271] : memref<?xui8, 1>, index -> <ui8>
                      %273 = ascvec.unpack %272 {part = "lower"} : <ui8> -> <ui16>
                      %274 = ascvec.and %273, %267, %263 : <ui16>, <ui16> -> <ui16>
                      %275 = ascvec.shr %273, 3, %263 : <ui16> -> <ui16>
                      %276 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %277 = ascvec.add %276, %274, %263 : <ui16>
                      %278 = ascvec.shr %277, 5, %263 : <ui16> -> <ui16>
                      %279 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %280 = ascvec.add %279, %275, %263 : <ui16>
                      %281 = ascvec.shr %280, 5, %263 : <ui16> -> <ui16>
                      %282 = ascvec.and %278, %265, %263 : <ui16>, <ui16> -> <ui16>
                      %283 = ascvec.shl %281, 8, %263 : <ui16> -> <ui16>
                      %284 = ascvec.or %282, %283, %263 : <ui16>, <ui16> -> <ui16>
                      %285 = ascvec.reinterpret %284 : <ui16> -> <ui8>
                      %286 = ascvec.add_scalar %285, %c64_i8, %264 : <ui8>, i8
                      %287 = ascvec.min_scalar %286, %c123_i8, %264 : <ui8>, i8
                      %288 = arith.muli %arg14, %c512_i32 : i32
                      %289 = arith.index_cast %288 : i32 to index
                      %result0_13, %result1_14 = ascvec.loadalign_deintlv %arg11[%289] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %290 = ascvec.and %result1_14, %266, %264 : <ui8>, <ui8> -> <ui8>
                      %291 = ascvec.or %287, %290, %264 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg12[%269], %291, %264 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
                } else {
                  %261 = arith.cmpi sge, %259, %c13568_i64 : i64
                  scf.if %261 {
                    ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %262 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                    %263 = ascscalar.bitcast_memref %108 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                    ascvec.vf(%108, %262, %263, %alloc_6, %255 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                    ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32):
                      %c512_i32 = arith.constant 512 : i32
                      %c1_i32_11 = arith.constant 1 : i32
                      %c0_i32_12 = arith.constant 0 : i32
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
                      %264 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                      %265 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                      %266 = ascvec.duplicate_scalar %c255_i16, %264 : i16 -> <ui16>
                      %267 = ascvec.duplicate_scalar %c-128_i8, %265 : i8 -> <ui8>
                      %268 = ascvec.arange %c0_i8 : i8 -> <ui8>
                      %269 = ascvec.shr %268, 1, %265 : <ui8> -> <ui8>
                      %270 = ascvec.shr %269, 2, %265 : <ui8> -> <ui8>
                      %271 = ascvec.shl %270, 2, %265 : <ui8> -> <ui8>
                      %272 = ascvec.duplicate_scalar %c1_i8, %265 : i8 -> <ui8>
                      %273 = ascvec.and %268, %272, %265 : <ui8>, <ui8> -> <ui8>
                      %274 = ascvec.add_scalar %269, %c3_i8, %265 : <ui8>, i8
                      %275 = ascvec.add %274, %273, %265 : <ui8>
                      %276 = ascvec.duplicate_scalar %c3_i8, %265 : i8 -> <ui8>
                      %277 = ascvec.and %275, %276, %265 : <ui8>, <ui8> -> <ui8>
                      %278 = ascvec.or %271, %277, %265 : <ui8>, <ui8> -> <ui8>
                      %279 = ascvec.duplicate_scalar %c4095_i16, %264 : i16 -> <ui16>
                      %280 = ascvec.duplicate_scalar %c32767_i16, %264 : i16 -> <ui16>
                      %281 = ascvec.duplicate_scalar %cst, %264 : bf16 -> <bf16>
                      scf.for %arg14 = %c0_i32_12 to %arg13 step %c1_i32_11  : i32 {
                        %282 = arith.muli %arg14, %c256_i32 : i32
                        %283 = arith.index_cast %282 : i32 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg9[%283] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                        %284 = arith.muli %arg14, %c128_i32 : i32
                        %285 = arith.index_cast %284 : i32 to index
                        %286 = ascvec.loadalign %arg10[%285] : memref<?xui8, 1>, index -> <ui8>
                        %287 = ascvec.gather_reg %286, %278 : <ui8>, <ui8> -> <ui8>
                        %288 = ascvec.reinterpret %287 : <ui8> -> <ui16>
                        %289 = ascvec.shr %288, 1, %264 : <ui16> -> <ui16>
                        %290 = ascvec.and %289, %279, %264 : <ui16>, <ui16> -> <ui16>
                        %291 = ascvec.shr %288, 4, %264 : <ui16> -> <ui16>
                        %292 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                        %293 = ascvec.and %292, %280, %264 : <ui16>, <ui16> -> <ui16>
                        %294 = ascvec.cmp_lt_scalar %293, %c14464_i16, %264 : <ui16>, i16 -> !ascvec.mask
                        %295 = ascvec.shr %290, 7, %264 : <ui16> -> <ui16>
                        %296 = ascvec.add %293, %295, %264 : <ui16>
                        %297 = ascvec.shr %296, 5, %264 : <ui16> -> <ui16>
                        %298 = ascvec.reinterpret %293 : <ui16> -> <bf16>
                        %299 = ascvec.mul %298, %281, %294 : <bf16>
                        %300 = ascvec.cast %299, %264 {rounding = 4 : i32} : <bf16> -> <f16>
                        %301 = ascvec.cast %300, %264 {rounding = 4 : i32} : <f16> -> <i16>
                        %302 = ascvec.reinterpret %301 : <i16> -> <ui16>
                        %303 = ascvec.add %302, %290, %264 : <ui16>
                        %304 = ascvec.shr %303, 12, %264 : <ui16> -> <ui16>
                        %305 = ascvec.add_scalar %304, %c448_i16, %264 : <ui16>, i16
                        %306 = ascvec.select %305, %297, %294 : <ui16>
                        %307 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                        %308 = ascvec.and %307, %280, %264 : <ui16>, <ui16> -> <ui16>
                        %309 = ascvec.cmp_lt_scalar %308, %c14464_i16, %264 : <ui16>, i16 -> !ascvec.mask
                        %310 = ascvec.shr %291, 7, %264 : <ui16> -> <ui16>
                        %311 = ascvec.add %308, %310, %264 : <ui16>
                        %312 = ascvec.shr %311, 5, %264 : <ui16> -> <ui16>
                        %313 = ascvec.reinterpret %308 : <ui16> -> <bf16>
                        %314 = ascvec.mul %313, %281, %309 : <bf16>
                        %315 = ascvec.cast %314, %264 {rounding = 4 : i32} : <bf16> -> <f16>
                        %316 = ascvec.cast %315, %264 {rounding = 4 : i32} : <f16> -> <i16>
                        %317 = ascvec.reinterpret %316 : <i16> -> <ui16>
                        %318 = ascvec.add %317, %291, %264 : <ui16>
                        %319 = ascvec.shr %318, 12, %264 : <ui16> -> <ui16>
                        %320 = ascvec.add_scalar %319, %c448_i16, %264 : <ui16>, i16
                        %321 = ascvec.select %320, %312, %309 : <ui16>
                        %322 = ascvec.and %306, %266, %264 : <ui16>, <ui16> -> <ui16>
                        %323 = ascvec.shl %321, 8, %264 : <ui16> -> <ui16>
                        %324 = ascvec.or %322, %323, %264 : <ui16>, <ui16> -> <ui16>
                        %325 = ascvec.reinterpret %324 : <ui16> -> <ui8>
                        %326 = ascvec.add_scalar %325, %c64_i8, %265 : <ui8>, i8
                        %327 = ascvec.min_scalar %326, %c123_i8, %265 : <ui8>, i8
                        %328 = arith.muli %arg14, %c512_i32 : i32
                        %329 = arith.index_cast %328 : i32 to index
                        %result0_13, %result1_14 = ascvec.loadalign_deintlv %arg11[%329] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                        %330 = ascvec.and %result1_14, %267, %265 : <ui8>, <ui8> -> <ui8>
                        %331 = ascvec.or %327, %330, %265 : <ui8>, <ui8> -> <ui8>
                        ascvec.storealign %arg12[%283], %331, %265 : memref<32768xui8, 1>, index, <ui8>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
                  } else {
                    ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %262 = ascscalar.bitcast_memref %108 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                    ascvec.vf(%108, %alloc_9, %262, %256, %alloc_6, %255 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32) {
                    ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<8192xi32, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<?xui16, 1>, %arg13: memref<32768xui8, 1>, %arg14: i32):
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
                      %c4_i32_11 = arith.constant 4 : i32
                      %c2_i32_12 = arith.constant 2 : i32
                      %c1_i32_13 = arith.constant 1 : i32
                      %c7_i32 = arith.constant 7 : i32
                      %c32_i32 = arith.constant 32 : i32
                      %c0_i32_14 = arith.constant 0 : i32
                      %c32767_i16 = arith.constant 32767 : i16
                      %c0_15 = arith.constant 0 : index
                      %c256_i32 = arith.constant 256 : i32
                      %c3_i8 = arith.constant 3 : i8
                      %c-128_i8 = arith.constant -128 : i8
                      %c0_i8 = arith.constant 0 : i8
                      %c8_i32 = arith.constant 8 : i32
                      %263 = ascvec.create_mask : !ascvec.mask
                      %264 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                      %265 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                      %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                      %266 = ascvec.arange %c0_i8 : i8 -> <ui8>
                      %267 = ascvec.duplicate_scalar %c-128_i8, %265 : i8 -> <ui8>
                      %268 = ascvec.shr %266, 5, %265 : <ui8> -> <ui8>
                      %269 = ascvec.duplicate_scalar %c3_i8, %265 : i8 -> <ui8>
                      %270 = ascvec.and %266, %269, %265 : <ui8>, <ui8> -> <ui8>
                      %271 = ascvec.shl %270, 6, %265 : <ui8> -> <ui8>
                      %272 = ascvec.shr %266, 2, %265 : <ui8> -> <ui8>
                      %273 = ascvec.or %271, %272, %265 : <ui8>, <ui8> -> <ui8>
                      scf.for %arg15 = %c0_i32_14 to %arg14 step %c1_i32_13  : i32 {
                        %274 = arith.muli %arg15, %c256_i32 : i32
                        %275 = arith.index_cast %274 : i32 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg9[%275] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                        %276 = ascvec.duplicate_scalar %c32767_i16, %264 : i16 -> <ui16>
                        %277 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                        %278 = ascvec.and %277, %276, %264 : <ui16>, <ui16> -> <ui16>
                        %279 = ascvec.reinterpret %278 : <ui16> -> <bf16>
                        %280 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                        %281 = ascvec.and %280, %276, %264 : <ui16>, <ui16> -> <ui16>
                        %282 = ascvec.reinterpret %281 : <ui16> -> <bf16>
                        %283 = ascvec.cast %279, %264 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %284 = ascvec.cast %279, %264 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %285 = ascvec.cast %282, %264 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %286 = ascvec.cast %282, %264 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %287 = ascvec.arange %c0_i32_14 : i32 -> <ui32>
                        %288 = ascvec.shl %287, 2, %263 : <ui32> -> <ui32>
                        %289 = ascvec.add_scalar %288, %c0_i32_14, %263 : <ui32>, i32
                        %290 = ascvec.shr %289, 3, %263 : <ui32> -> <ui32>
                        %291 = arith.muli %arg15, %c32_i32 : i32
                        %292 = ascvec.add_scalar %290, %291, %263 : <ui32>, i32
                        %293 = ascvec.gather %arg10[%c0_15], %292, %263 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %294 = ascvec.reinterpret %293 : <i32> -> <ui32>
                        %295 = ascvec.duplicate_scalar %c7_i32, %263 : i32 -> <ui32>
                        %296 = ascvec.and %289, %295, %263 : <ui32>, <ui32> -> <ui32>
                        %297 = ascvec.shl %294, 11, %263 : <ui32> -> <ui32>
                        %298 = ascvec.shr %294, 21, %263 : <ui32> -> <ui32>
                        %299 = ascvec.or %297, %298, %263 : <ui32>, <ui32> -> <ui32>
                        %300 = ascvec.shr %299, 3, %263 : <ui32> -> <ui32>
                        %301 = ascvec.shl %299, 29, %263 : <ui32> -> <ui32>
                        %302 = ascvec.or %300, %301, %263 : <ui32>, <ui32> -> <ui32>
                        %303 = ascvec.duplicate_scalar %c1_i32_13, %263 : i32 -> <ui32>
                        %304 = ascvec.and %296, %303, %263 : <ui32>, <ui32> -> <ui32>
                        %305 = ascvec.cmp_ne_scalar %304, %c0_i32_14, %263 : <ui32>, i32 -> !ascvec.mask
                        %306 = ascvec.select %302, %299, %305 : <ui32>
                        %307 = ascvec.shr %306, 8, %263 : <ui32> -> <ui32>
                        %308 = ascvec.shl %306, 24, %263 : <ui32> -> <ui32>
                        %309 = ascvec.or %307, %308, %263 : <ui32>, <ui32> -> <ui32>
                        %310 = ascvec.duplicate_scalar %c2_i32_12, %263 : i32 -> <ui32>
                        %311 = ascvec.and %296, %310, %263 : <ui32>, <ui32> -> <ui32>
                        %312 = ascvec.cmp_ne_scalar %311, %c0_i32_14, %263 : <ui32>, i32 -> !ascvec.mask
                        %313 = ascvec.select %309, %306, %312 : <ui32>
                        %314 = ascvec.shr %313, 16, %263 : <ui32> -> <ui32>
                        %315 = ascvec.shl %313, 16, %263 : <ui32> -> <ui32>
                        %316 = ascvec.or %314, %315, %263 : <ui32>, <ui32> -> <ui32>
                        %317 = ascvec.duplicate_scalar %c4_i32_11, %263 : i32 -> <ui32>
                        %318 = ascvec.and %296, %317, %263 : <ui32>, <ui32> -> <ui32>
                        %319 = ascvec.cmp_ne_scalar %318, %c0_i32_14, %263 : <ui32>, i32 -> !ascvec.mask
                        %320 = ascvec.select %316, %313, %319 : <ui32>
                        %321 = ascvec.duplicate_scalar %c65535_i32, %263 : i32 -> <ui32>
                        %322 = ascvec.and %320, %321, %263 : <ui32>, <ui32> -> <ui32>
                        %323 = ascvec.add_scalar %288, %c1_i32_13, %263 : <ui32>, i32
                        %324 = ascvec.shr %323, 3, %263 : <ui32> -> <ui32>
                        %325 = ascvec.add_scalar %324, %291, %263 : <ui32>, i32
                        %326 = ascvec.gather %arg10[%c0_15], %325, %263 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %327 = ascvec.reinterpret %326 : <i32> -> <ui32>
                        %328 = ascvec.and %323, %295, %263 : <ui32>, <ui32> -> <ui32>
                        %329 = ascvec.shl %327, 11, %263 : <ui32> -> <ui32>
                        %330 = ascvec.shr %327, 21, %263 : <ui32> -> <ui32>
                        %331 = ascvec.or %329, %330, %263 : <ui32>, <ui32> -> <ui32>
                        %332 = ascvec.shr %331, 3, %263 : <ui32> -> <ui32>
                        %333 = ascvec.shl %331, 29, %263 : <ui32> -> <ui32>
                        %334 = ascvec.or %332, %333, %263 : <ui32>, <ui32> -> <ui32>
                        %335 = ascvec.and %328, %303, %263 : <ui32>, <ui32> -> <ui32>
                        %336 = ascvec.cmp_ne_scalar %335, %c0_i32_14, %263 : <ui32>, i32 -> !ascvec.mask
                        %337 = ascvec.select %334, %331, %336 : <ui32>
                        %338 = ascvec.shr %337, 8, %263 : <ui32> -> <ui32>
                        %339 = ascvec.shl %337, 24, %263 : <ui32> -> <ui32>
                        %340 = ascvec.or %338, %339, %263 : <ui32>, <ui32> -> <ui32>
                        %341 = ascvec.and %328, %310, %263 : <ui32>, <ui32> -> <ui32>
                        %342 = ascvec.cmp_ne_scalar %341, %c0_i32_14, %263 : <ui32>, i32 -> !ascvec.mask
                        %343 = ascvec.select %340, %337, %342 : <ui32>
                        %344 = ascvec.shr %343, 16, %263 : <ui32> -> <ui32>
                        %345 = ascvec.shl %343, 16, %263 : <ui32> -> <ui32>
                        %346 = ascvec.or %344, %345, %263 : <ui32>, <ui32> -> <ui32>
                        %347 = ascvec.and %328, %317, %263 : <ui32>, <ui32> -> <ui32>
                        %348 = ascvec.cmp_ne_scalar %347, %c0_i32_14, %263 : <ui32>, i32 -> !ascvec.mask
                        %349 = ascvec.select %346, %343, %348 : <ui32>
                        %350 = ascvec.and %349, %321, %263 : <ui32>, <ui32> -> <ui32>
                        %351 = ascvec.add_scalar %288, %c2_i32_12, %263 : <ui32>, i32
                        %352 = ascvec.shr %351, 3, %263 : <ui32> -> <ui32>
                        %353 = ascvec.add_scalar %352, %291, %263 : <ui32>, i32
                        %354 = ascvec.gather %arg10[%c0_15], %353, %263 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %355 = ascvec.reinterpret %354 : <i32> -> <ui32>
                        %356 = ascvec.and %351, %295, %263 : <ui32>, <ui32> -> <ui32>
                        %357 = ascvec.shl %355, 11, %263 : <ui32> -> <ui32>
                        %358 = ascvec.shr %355, 21, %263 : <ui32> -> <ui32>
                        %359 = ascvec.or %357, %358, %263 : <ui32>, <ui32> -> <ui32>
                        %360 = ascvec.shr %359, 3, %263 : <ui32> -> <ui32>
                        %361 = ascvec.shl %359, 29, %263 : <ui32> -> <ui32>
                        %362 = ascvec.or %360, %361, %263 : <ui32>, <ui32> -> <ui32>
                        %363 = ascvec.and %356, %303, %263 : <ui32>, <ui32> -> <ui32>
                        %364 = ascvec.cmp_ne_scalar %363, %c0_i32_14, %263 : <ui32>, i32 -> !ascvec.mask
                        %365 = ascvec.select %362, %359, %364 : <ui32>
                        %366 = ascvec.shr %365, 8, %263 : <ui32> -> <ui32>
                        %367 = ascvec.shl %365, 24, %263 : <ui32> -> <ui32>
                        %368 = ascvec.or %366, %367, %263 : <ui32>, <ui32> -> <ui32>
                        %369 = ascvec.and %356, %310, %263 : <ui32>, <ui32> -> <ui32>
                        %370 = ascvec.cmp_ne_scalar %369, %c0_i32_14, %263 : <ui32>, i32 -> !ascvec.mask
                        %371 = ascvec.select %368, %365, %370 : <ui32>
                        %372 = ascvec.shr %371, 16, %263 : <ui32> -> <ui32>
                        %373 = ascvec.shl %371, 16, %263 : <ui32> -> <ui32>
                        %374 = ascvec.or %372, %373, %263 : <ui32>, <ui32> -> <ui32>
                        %375 = ascvec.and %356, %317, %263 : <ui32>, <ui32> -> <ui32>
                        %376 = ascvec.cmp_ne_scalar %375, %c0_i32_14, %263 : <ui32>, i32 -> !ascvec.mask
                        %377 = ascvec.select %374, %371, %376 : <ui32>
                        %378 = ascvec.and %377, %321, %263 : <ui32>, <ui32> -> <ui32>
                        %379 = ascvec.add_scalar %288, %c3_i32, %263 : <ui32>, i32
                        %380 = ascvec.shr %379, 3, %263 : <ui32> -> <ui32>
                        %381 = ascvec.add_scalar %380, %291, %263 : <ui32>, i32
                        %382 = ascvec.gather %arg10[%c0_15], %381, %263 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %383 = ascvec.reinterpret %382 : <i32> -> <ui32>
                        %384 = ascvec.and %379, %295, %263 : <ui32>, <ui32> -> <ui32>
                        %385 = ascvec.shl %383, 11, %263 : <ui32> -> <ui32>
                        %386 = ascvec.shr %383, 21, %263 : <ui32> -> <ui32>
                        %387 = ascvec.or %385, %386, %263 : <ui32>, <ui32> -> <ui32>
                        %388 = ascvec.shr %387, 3, %263 : <ui32> -> <ui32>
                        %389 = ascvec.shl %387, 29, %263 : <ui32> -> <ui32>
                        %390 = ascvec.or %388, %389, %263 : <ui32>, <ui32> -> <ui32>
                        %391 = ascvec.and %384, %303, %263 : <ui32>, <ui32> -> <ui32>
                        %392 = ascvec.cmp_ne_scalar %391, %c0_i32_14, %263 : <ui32>, i32 -> !ascvec.mask
                        %393 = ascvec.select %390, %387, %392 : <ui32>
                        %394 = ascvec.shr %393, 8, %263 : <ui32> -> <ui32>
                        %395 = ascvec.shl %393, 24, %263 : <ui32> -> <ui32>
                        %396 = ascvec.or %394, %395, %263 : <ui32>, <ui32> -> <ui32>
                        %397 = ascvec.and %384, %310, %263 : <ui32>, <ui32> -> <ui32>
                        %398 = ascvec.cmp_ne_scalar %397, %c0_i32_14, %263 : <ui32>, i32 -> !ascvec.mask
                        %399 = ascvec.select %396, %393, %398 : <ui32>
                        %400 = ascvec.shr %399, 16, %263 : <ui32> -> <ui32>
                        %401 = ascvec.shl %399, 16, %263 : <ui32> -> <ui32>
                        %402 = ascvec.or %400, %401, %263 : <ui32>, <ui32> -> <ui32>
                        %403 = ascvec.and %384, %317, %263 : <ui32>, <ui32> -> <ui32>
                        %404 = ascvec.cmp_ne_scalar %403, %c0_i32_14, %263 : <ui32>, i32 -> !ascvec.mask
                        %405 = ascvec.select %402, %399, %404 : <ui32>
                        %406 = ascvec.and %405, %321, %263 : <ui32>, <ui32> -> <ui32>
                        %407 = ascvec.reinterpret %283 : <f32> -> <ui32>
                        %408 = ascvec.reinterpret %407 : <ui32> -> <f32>
                        %409 = ascvec.max_scalar %408, %cst, %263 : <f32>, f32
                        %410 = ascvec.add %408, %409, %263 : <f32>
                        %411 = ascvec.reinterpret %410 : <f32> -> <ui32>
                        %412 = ascvec.shr %411, 5, %263 : <ui32> -> <ui32>
                        %413 = ascvec.add %412, %322, %263 : <ui32>
                        %414 = ascvec.reinterpret %285 : <f32> -> <ui32>
                        %415 = ascvec.reinterpret %414 : <ui32> -> <f32>
                        %416 = ascvec.max_scalar %415, %cst, %263 : <f32>, f32
                        %417 = ascvec.add %415, %416, %263 : <f32>
                        %418 = ascvec.reinterpret %417 : <f32> -> <ui32>
                        %419 = ascvec.shr %418, 5, %263 : <ui32> -> <ui32>
                        %420 = ascvec.add %419, %350, %263 : <ui32>
                        %421 = ascvec.reinterpret %284 : <f32> -> <ui32>
                        %422 = ascvec.reinterpret %421 : <ui32> -> <f32>
                        %423 = ascvec.max_scalar %422, %cst, %263 : <f32>, f32
                        %424 = ascvec.add %422, %423, %263 : <f32>
                        %425 = ascvec.reinterpret %424 : <f32> -> <ui32>
                        %426 = ascvec.shr %425, 5, %263 : <ui32> -> <ui32>
                        %427 = ascvec.add %426, %378, %263 : <ui32>
                        %428 = ascvec.reinterpret %286 : <f32> -> <ui32>
                        %429 = ascvec.reinterpret %428 : <ui32> -> <f32>
                        %430 = ascvec.max_scalar %429, %cst, %263 : <f32>, f32
                        %431 = ascvec.add %429, %430, %263 : <f32>
                        %432 = ascvec.reinterpret %431 : <f32> -> <ui32>
                        %433 = ascvec.shr %432, 5, %263 : <ui32> -> <ui32>
                        %434 = ascvec.add %433, %406, %263 : <ui32>
                        %435 = ascvec.reinterpret %413 : <ui32> -> <ui16>
                        %436 = ascvec.reinterpret %420 : <ui32> -> <ui16>
                        %result0_16, %result1_17 = ascvec.deintlv %435, %436 : <ui16>, <ui16> -> <ui16>, <ui16>
                        %437 = ascvec.reinterpret %427 : <ui32> -> <ui16>
                        %438 = ascvec.reinterpret %434 : <ui32> -> <ui16>
                        %result0_18, %result1_19 = ascvec.deintlv %437, %438 : <ui16>, <ui16> -> <ui16>, <ui16>
                        %439 = ascvec.reinterpret %result1_17 : <ui16> -> <ui8>
                        %440 = ascvec.reinterpret %result1_19 : <ui16> -> <ui8>
                        %result0_20, %result1_21 = ascvec.deintlv %439, %440 : <ui8>, <ui8> -> <ui8>, <ui8>
                        %441 = ascvec.gather_reg %result0_20, %273 : <ui8>, <ui8> -> <ui8>
                        %442 = ascvec.add_scalar %441, %c60_i8, %265 : <ui8>, i8
                        %443 = ascvec.cmp_ne_scalar %442, %c-1_i8, %265 : <ui8>, i8 -> !ascvec.mask
                        %444 = ascvec.min_scalar %442, %c123_i8, %443 : <ui8>, i8
                        %445 = arith.muli %arg15, %c512_i32 : i32
                        %446 = arith.index_cast %445 : i32 to index
                        %result0_22, %result1_23 = ascvec.loadalign_deintlv %arg11[%446] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                        %447 = ascvec.and %result1_23, %267, %265 : <ui8>, <ui8> -> <ui8>
                        %448 = ascvec.or %444, %447, %265 : <ui8>, <ui8> -> <ui8>
                        %449 = ascvec.arange %c0_i16 : i16 -> <ui16>
                        %450 = arith.muli %arg15, %c16_i32 : i32
                        %451 = arith.trunci %450 : i32 to i16
                        %452 = ascvec.add_scalar %449, %451, %264 : <ui16>, i16
                        %453 = ascvec.gather %arg12[%c0_15], %452, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %454 = ascvec.pack %453 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                        %455 = ascvec.gather_reg %454, %268 : <ui8>, <ui8> -> <ui8>
                        %456 = ascvec.cmp_eq_scalar %455, %c-1_i8, %265 : <ui8>, i8 -> !ascvec.mask
                        %457 = ascvec.duplicate_scalar %c127_i8, %456 merge %448 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                        ascvec.storealign %arg13[%275], %457, %265 : memref<32768xui8, 1>, index, <ui8>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
                  }
                }
              }
            } else {
              %200 = arith.cmpi sge, %67, %c1024_i64 : i64
              %201 = arith.remsi %67, %c1024_i64 : i64
              %202 = arith.cmpi eq, %201, %c0_i64 : i64
              %203 = arith.select %200, %202, %false : i1
              scf.if %203 {
                %204 = arith.divsi %67, %c256_i64 : i64
                %205 = arith.addi %67, %c1023_i64 : i64
                %206 = arith.divsi %205, %c1024_i64 : i64
                %207 = arith.trunci %1 : i64 to i32
                %208 = arith.extui %207 : i32 to i64
                %209 = arith.subi %1, %208 : i64
                %210 = arith.remsi %209, %c4294967296_i64 : i64
                %211 = arith.cmpi ne, %210, %c0_i64 : i64
                %212 = arith.cmpi slt, %210, %c0_i64 : i64
                %213 = arith.andi %211, %212 : i1
                %214 = arith.divsi %209, %c4294967296_i64 : i64
                %215 = arith.extui %213 : i1 to i64
                %216 = arith.subi %214, %215 : i64
                %217 = arith.trunci %216 : i64 to i32
                %218 = arith.divui %109, %c16_i64 : i64
                %219 = arith.trunci %218 : i64 to i32
                %220 = arith.extui %219 : i32 to i64
                %221 = arith.subi %218, %220 : i64
                %222 = arith.remsi %221, %c4294967296_i64 : i64
                %223 = arith.cmpi ne, %222, %c0_i64 : i64
                %224 = arith.cmpi slt, %222, %c0_i64 : i64
                %225 = arith.andi %223, %224 : i1
                %226 = arith.divsi %221, %c4294967296_i64 : i64
                %227 = arith.extui %225 : i1 to i64
                %228 = arith.subi %226, %227 : i64
                %229 = arith.trunci %228 : i64 to i32
                %230 = arith.trunci %4 : i64 to i32
                %231 = arith.extui %230 : i32 to i64
                %232 = arith.subi %4, %231 : i64
                %233 = arith.remsi %232, %c4294967296_i64 : i64
                %234 = arith.cmpi ne, %233, %c0_i64 : i64
                %235 = arith.cmpi slt, %233, %c0_i64 : i64
                %236 = arith.andi %234, %235 : i1
                %237 = arith.divsi %232, %c4294967296_i64 : i64
                %238 = arith.extui %236 : i1 to i64
                %239 = arith.subi %237, %238 : i64
                %240 = arith.trunci %239 : i64 to i32
                %241 = arith.divui %230, %c4_i32 : i32
                %242 = arith.remui %240, %c4_i32 : i32
                %243 = arith.muli %242, %c1073741824_i32 : i32
                %244 = arith.addi %241, %243 : i32
                %245 = arith.divui %240, %c4_i32 : i32
                %246 = arith.muli %206, %c64_i64 : i64
                %247 = arith.divui %218, %c16384_i64 : i64
                %248 = arith.addi %218, %246 : i64
                %249 = arith.subi %248, %c1_i64 : i64
                %250 = arith.divui %249, %c16384_i64 : i64
                %251 = arith.cmpi eq, %247, %250 : i64
                scf.if %251 {
                  %261 = arith.trunci %247 : i64 to i32
                  %262 = arith.extui %261 : i32 to i64
                  %263 = arith.subi %247, %262 : i64
                  %264 = arith.remsi %263, %c4294967296_i64 : i64
                  %265 = arith.cmpi ne, %264, %c0_i64 : i64
                  %266 = arith.cmpi slt, %264, %c0_i64 : i64
                  %267 = arith.andi %265, %266 : i1
                  %268 = arith.divsi %263, %c4294967296_i64 : i64
                  %269 = arith.extui %267 : i1 to i64
                  %270 = arith.subi %268, %269 : i64
                  %271 = arith.trunci %270 : i64 to i32
                  %272 = arith.addi %244, %261 : i32
                  %273 = arith.addi %245, %271 : i32
                  %274 = arith.cmpi ult, %272, %244 : i32
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
                  %290 = arith.xori %276, %217 : i32
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
                  %304 = arith.addi %207, %c-1640531527_i32 : i32
                  %305 = arith.addi %217, %c-1150833019_i32 : i32
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
                  %321 = arith.xori %289, %207 : i32
                  %322 = arith.addi %207, %c1013904242_i32 : i32
                  %323 = arith.xori %293, %322 : i32
                  %324 = arith.addi %217, %c1993301258_i32 : i32
                  %325 = arith.xori %320, %324 : i32
                  ascsimt.vf(%219, %321, %305, %323, %325, %207, %217, %310, %alloc_9, %246 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: memref<8192xi32, 1>, %arg18: i64):
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
                    %c16384_i32 = arith.constant 16384 : i32
                    %326 = ascsimt.thread_idx {ascsimt.scope = 14 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %327 = arith.extsi %326 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg19 = %327 to %arg18 step %c256_i64_11  : i64 {
                      %328 = arith.trunci %arg19 : i64 to i32
                      %329 = arith.addi %arg9, %328 : i32
                      %330 = arith.remui %329, %c16384_i32 : i32
                      %331 = arith.xori %330, %arg10 : i32
                      %332 = arith.muli %331, %c-766435501_i32 : i32
                      %333 = "ascsimt.bitcast"(%331) : (i32) -> ui32
                      %334 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %335 = ascsimt.math_binary %333, %334, "mulhi" : ui32, ui32 -> ui32
                      %336 = "ascsimt.bitcast"(%335) : (ui32) -> i32
                      %337 = arith.xori %336, %arg11 : i32
                      %338 = arith.muli %337, %c-845247145_i32 : i32
                      %339 = "ascsimt.bitcast"(%337) : (i32) -> ui32
                      %340 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %341 = ascsimt.math_binary %339, %340, "mulhi" : ui32, ui32 -> ui32
                      %342 = "ascsimt.bitcast"(%341) : (ui32) -> i32
                      %343 = arith.xori %342, %arg12 : i32
                      %344 = arith.xori %332, %arg13 : i32
                      %345:4 = scf.for %arg20 = %c3_i32 to %c10_i32 step %c1_i32_16 iter_args(%arg21 = %343, %arg22 = %338, %arg23 = %344, %arg24 = %arg16) -> (i32, i32, i32, i32)  : i32 {
                        %354 = arith.muli %arg21, %c-766435501_i32 : i32
                        %355 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %356 = ascsimt.math_binary %355, %334, "mulhi" : ui32, ui32 -> ui32
                        %357 = "ascsimt.bitcast"(%356) : (ui32) -> i32
                        %358 = arith.muli %arg23, %c-845247145_i32 : i32
                        %359 = "ascsimt.bitcast"(%arg23) : (i32) -> ui32
                        %360 = ascsimt.math_binary %359, %340, "mulhi" : ui32, ui32 -> ui32
                        %361 = "ascsimt.bitcast"(%360) : (ui32) -> i32
                        %362 = arith.muli %arg20, %c-1640531527_i32_18 : i32
                        %363 = arith.addi %arg14, %362 : i32
                        %364 = arith.muli %arg20, %c-1150833019_i32_17 : i32
                        %365 = arith.addi %arg15, %364 : i32
                        %366 = arith.xori %361, %arg22 : i32
                        %367 = arith.xori %366, %363 : i32
                        %368 = arith.xori %357, %arg24 : i32
                        %369 = arith.xori %368, %365 : i32
                        scf.yield %367, %358, %369, %354 : i32, i32, i32, i32
                      }
                      %346 = arith.muli %arg19, %c4_i64_15 : i64
                      %347 = arith.index_cast %346 : i64 to index
                      ascsimt.store %arg17[%347], %345#0 : memref<8192xi32, 1>, i32
                      %348 = arith.addi %346, %c1_i64_14 : i64
                      %349 = arith.index_cast %348 : i64 to index
                      ascsimt.store %arg17[%349], %345#1 : memref<8192xi32, 1>, i32
                      %350 = arith.addi %346, %c2_i64_13 : i64
                      %351 = arith.index_cast %350 : i64 to index
                      ascsimt.store %arg17[%351], %345#2 : memref<8192xi32, 1>, i32
                      %352 = arith.addi %346, %c3_i64_12 : i64
                      %353 = arith.index_cast %352 : i64 to index
                      ascsimt.store %arg17[%353], %345#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%219, %229, %244, %245, %207, %217, %alloc_9, %246 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
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
                    %c16384_i32 = arith.constant 16384 : i32
                    %261 = ascsimt.thread_idx {ascsimt.scope = 15 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %262 = arith.extsi %261 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg17 = %262 to %arg16 step %c256_i64_11  : i64 {
                      %263 = arith.trunci %arg17 : i64 to i32
                      %264 = arith.addi %arg9, %263 : i32
                      %265 = arith.cmpi ult, %264, %arg9 : i32
                      %266 = arith.extui %265 : i1 to i32
                      %267 = arith.addi %arg10, %266 : i32
                      %268 = arith.divui %264, %c16384_i32 : i32
                      %269 = arith.remui %267, %c16384_i32 : i32
                      %270 = arith.muli %269, %c262144_i32 : i32
                      %271 = arith.addi %268, %270 : i32
                      %272 = arith.divui %267, %c16384_i32 : i32
                      %273 = arith.addi %arg11, %271 : i32
                      %274 = arith.addi %arg12, %272 : i32
                      %275 = arith.cmpi ult, %273, %arg11 : i32
                      %276 = arith.extui %275 : i1 to i32
                      %277 = arith.addi %274, %276 : i32
                      %278 = arith.remui %264, %c16384_i32 : i32
                      %279:4 = scf.for %arg18 = %c0_i32_17 to %c10_i32 step %c1_i32_16 iter_args(%arg19 = %c0_i32_17, %arg20 = %278, %arg21 = %273, %arg22 = %277) -> (i32, i32, i32, i32)  : i32 {
                        %288 = arith.muli %arg19, %c-766435501_i32 : i32
                        %289 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %290 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %291 = ascsimt.math_binary %289, %290, "mulhi" : ui32, ui32 -> ui32
                        %292 = "ascsimt.bitcast"(%291) : (ui32) -> i32
                        %293 = arith.muli %arg21, %c-845247145_i32 : i32
                        %294 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %295 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %296 = ascsimt.math_binary %294, %295, "mulhi" : ui32, ui32 -> ui32
                        %297 = "ascsimt.bitcast"(%296) : (ui32) -> i32
                        %298 = arith.muli %arg18, %c-1640531527_i32_19 : i32
                        %299 = arith.addi %arg13, %298 : i32
                        %300 = arith.muli %arg18, %c-1150833019_i32_18 : i32
                        %301 = arith.addi %arg14, %300 : i32
                        %302 = arith.xori %297, %arg20 : i32
                        %303 = arith.xori %302, %299 : i32
                        %304 = arith.xori %292, %arg22 : i32
                        %305 = arith.xori %304, %301 : i32
                        scf.yield %303, %293, %305, %288 : i32, i32, i32, i32
                      }
                      %280 = arith.muli %arg17, %c4_i64_15 : i64
                      %281 = arith.index_cast %280 : i64 to index
                      ascsimt.store %arg15[%281], %279#0 : memref<8192xi32, 1>, i32
                      %282 = arith.addi %280, %c1_i64_14 : i64
                      %283 = arith.index_cast %282 : i64 to index
                      ascsimt.store %arg15[%283], %279#1 : memref<8192xi32, 1>, i32
                      %284 = arith.addi %280, %c2_i64_13 : i64
                      %285 = arith.index_cast %284 : i64 to index
                      ascsimt.store %arg15[%285], %279#2 : memref<8192xi32, 1>, i32
                      %286 = arith.addi %280, %c3_i64_12 : i64
                      %287 = arith.index_cast %286 : i64 to index
                      ascsimt.store %arg15[%287], %279#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %252 = arith.trunci %49 : i64 to i32
                %253 = arith.divsi %252, %c2_i32 : i32
                %254 = arith.trunci %253 : i32 to i8
                %255 = arith.trunci %204 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %256 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%254, %alloc_7, %108, %256, %alloc_10, %255 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, memref<2048xui16, 1>, i32) {
                ^bb0(%arg9: i8, %arg10: memref<2048xui8, 1>, %arg11: memref<32768xbf16, 1>, %arg12: memref<?xui16, 1>, %arg13: memref<2048xui16, 1>, %arg14: i32):
                  %c16_11 = arith.constant 16 : index
                  %c4_i32_12 = arith.constant 4 : i32
                  %c0_i32_13 = arith.constant 0 : i32
                  %c3_i32 = arith.constant 3 : i32
                  %c2_i32_14 = arith.constant 2 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c1_i32_15 = arith.constant 1 : i32
                  %c0_16 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %261 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %262 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %263 = ascvec.duplicate_scalar %c32767_i16, %261 : i16 -> <ui16>
                  %264 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %265 = ascvec.shr %264, 4, %261 : <ui16> -> <ui16>
                  %266 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %267 = ascvec.duplicate_scalar %c1_i8, %262 : i8 -> <ui8>
                  %268 = ascvec.and %266, %267, %262 : <ui8>, <ui8> -> <ui8>
                  %269 = ascvec.cmp_lt_scalar %266, %c16_i8, %262 : <ui8>, i8 -> !ascvec.mask
                  %270 = ascvec.cmp_lt_scalar %268, %arg9, %262 : <ui8>, i8 -> !ascvec.mask
                  %271 = ascvec.mask_and %269, %270, %262
                  ascvec.store_unalign_begin %arg10 : memref<2048xui8, 1>
                  %mask_17, %remaining_out_18 = ascvec.update_mask %c1_i32_15 elem_bits 16 : !ascvec.mask
                  %272 = ascvec.duplicate_scalar %c-1_i16, %mask_17 : i16 -> <ui16>
                  %273 = scf.for %arg15 = %c0_i32_13 to %arg14 step %c4_i32_12 iter_args(%arg16 = %272) -> (!ascvec.vreg<ui16>)  : i32 {
                    %274 = ascvec.duplicate_scalar %c-1_i16, %261 : i16 -> <ui16>
                    %275 = ascvec.duplicate_scalar %c0_i16, %261 : i16 -> <ui16>
                    %276 = arith.muli %arg15, %c256_i32 : i32
                    %277 = arith.index_cast %276 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg11[%277] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %278 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %279 = ascvec.and %278, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %280 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %281 = ascvec.and %280, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %282 = ascvec.max %279, %281, %261 : <ui16>
                    %283 = ascvec.reduce_max_datablock %282, %261 : <ui16>
                    %284 = ascvec.shr %283, 7, %261 : <ui16> -> <ui16>
                    %285 = ascvec.cmp_ge_scalar %284, %c255_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %286 = ascvec.max_scalar %284, %c15_i16, %261 : <ui16>, i16
                    %287 = ascvec.add_scalar %286, %c-15_i16, %261 : <ui16>, i16
                    %288 = ascvec.duplicate_scalar %c255_i16, %285 merge %287 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %289 = ascvec.duplicate_scalar %c254_i16, %261 : i16 -> <ui16>
                    %290 = ascvec.sub %289, %288, %261 : <ui16>
                    %291 = ascvec.cmp_ne_scalar %288, %c0_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %292 = ascvec.shl %290, 7, %291 : <ui16> -> <ui16>
                    %293 = ascvec.max %275, %288, %mask : <ui16>
                    %294 = ascvec.gather_reg %292, %265 : <ui16>, <ui16> -> <ui16>
                    %295 = ascvec.reinterpret %294 : <ui16> -> <bf16>
                    %296 = ascvec.mul %result0, %295, %261 : <bf16>
                    %297 = ascvec.mul %result1, %295, %261 : <bf16>
                    ascvec.storealign_intlv %arg11[%277], %296, %297 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %298 = ascvec.reinterpret %296 : <bf16> -> <ui16>
                    %299 = ascvec.and %298, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %300 = ascvec.reinterpret %297 : <bf16> -> <ui16>
                    %301 = ascvec.and %300, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %302 = ascvec.min %299, %301, %261 : <ui16>
                    %303 = ascvec.min %274, %302, %261 : <ui16>
                    %304 = arith.muli %arg15, %c16_i32 : i32
                    %305 = arith.index_cast %304 : i32 to index
                    ascvec.storealign %arg12[%305], %288, %mask : memref<?xui16, 1>, index, <ui16>
                    %306 = ascvec.reinterpret %288 : <ui16> -> <ui8>
                    %307 = ascvec.squeeze_store_reg %306, %271 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_16], %307 : memref<2048xui8, 1>, <ui8>
                    %308 = arith.addi %arg15, %c1_i32_15 : i32
                    %309 = arith.muli %308, %c256_i32 : i32
                    %310 = arith.index_cast %309 : i32 to index
                    %result0_20, %result1_21 = ascvec.loadalign_deintlv %arg11[%310] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %311 = ascvec.reinterpret %result0_20 : <bf16> -> <ui16>
                    %312 = ascvec.and %311, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %313 = ascvec.reinterpret %result1_21 : <bf16> -> <ui16>
                    %314 = ascvec.and %313, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %315 = ascvec.max %312, %314, %261 : <ui16>
                    %316 = ascvec.reduce_max_datablock %315, %261 : <ui16>
                    %317 = ascvec.shr %316, 7, %261 : <ui16> -> <ui16>
                    %318 = ascvec.cmp_ge_scalar %317, %c255_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %319 = ascvec.max_scalar %317, %c15_i16, %261 : <ui16>, i16
                    %320 = ascvec.add_scalar %319, %c-15_i16, %261 : <ui16>, i16
                    %321 = ascvec.duplicate_scalar %c255_i16, %318 merge %320 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %322 = ascvec.sub %289, %321, %261 : <ui16>
                    %323 = ascvec.cmp_ne_scalar %321, %c0_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %324 = ascvec.shl %322, 7, %323 : <ui16> -> <ui16>
                    %325 = ascvec.max %293, %321, %mask : <ui16>
                    %326 = ascvec.gather_reg %324, %265 : <ui16>, <ui16> -> <ui16>
                    %327 = ascvec.reinterpret %326 : <ui16> -> <bf16>
                    %328 = ascvec.mul %result0_20, %327, %261 : <bf16>
                    %329 = ascvec.mul %result1_21, %327, %261 : <bf16>
                    ascvec.storealign_intlv %arg11[%310], %328, %329 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %330 = ascvec.reinterpret %328 : <bf16> -> <ui16>
                    %331 = ascvec.and %330, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %332 = ascvec.reinterpret %329 : <bf16> -> <ui16>
                    %333 = ascvec.and %332, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %334 = ascvec.min %331, %333, %261 : <ui16>
                    %335 = ascvec.min %303, %334, %261 : <ui16>
                    %336 = arith.muli %308, %c16_i32 : i32
                    %337 = arith.index_cast %336 : i32 to index
                    ascvec.storealign %arg12[%337], %321, %mask : memref<?xui16, 1>, index, <ui16>
                    %338 = ascvec.reinterpret %321 : <ui16> -> <ui8>
                    %339 = ascvec.squeeze_store_reg %338, %271 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_16], %339 : memref<2048xui8, 1>, <ui8>
                    %340 = arith.addi %arg15, %c2_i32_14 : i32
                    %341 = arith.muli %340, %c256_i32 : i32
                    %342 = arith.index_cast %341 : i32 to index
                    %result0_22, %result1_23 = ascvec.loadalign_deintlv %arg11[%342] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %343 = ascvec.reinterpret %result0_22 : <bf16> -> <ui16>
                    %344 = ascvec.and %343, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %345 = ascvec.reinterpret %result1_23 : <bf16> -> <ui16>
                    %346 = ascvec.and %345, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %347 = ascvec.max %344, %346, %261 : <ui16>
                    %348 = ascvec.reduce_max_datablock %347, %261 : <ui16>
                    %349 = ascvec.shr %348, 7, %261 : <ui16> -> <ui16>
                    %350 = ascvec.cmp_ge_scalar %349, %c255_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %351 = ascvec.max_scalar %349, %c15_i16, %261 : <ui16>, i16
                    %352 = ascvec.add_scalar %351, %c-15_i16, %261 : <ui16>, i16
                    %353 = ascvec.duplicate_scalar %c255_i16, %350 merge %352 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %354 = ascvec.sub %289, %353, %261 : <ui16>
                    %355 = ascvec.cmp_ne_scalar %353, %c0_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %356 = ascvec.shl %354, 7, %355 : <ui16> -> <ui16>
                    %357 = ascvec.max %325, %353, %mask : <ui16>
                    %358 = ascvec.gather_reg %356, %265 : <ui16>, <ui16> -> <ui16>
                    %359 = ascvec.reinterpret %358 : <ui16> -> <bf16>
                    %360 = ascvec.mul %result0_22, %359, %261 : <bf16>
                    %361 = ascvec.mul %result1_23, %359, %261 : <bf16>
                    ascvec.storealign_intlv %arg11[%342], %360, %361 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %362 = ascvec.reinterpret %360 : <bf16> -> <ui16>
                    %363 = ascvec.and %362, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %364 = ascvec.reinterpret %361 : <bf16> -> <ui16>
                    %365 = ascvec.and %364, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %366 = ascvec.min %363, %365, %261 : <ui16>
                    %367 = ascvec.min %335, %366, %261 : <ui16>
                    %368 = arith.muli %340, %c16_i32 : i32
                    %369 = arith.index_cast %368 : i32 to index
                    ascvec.storealign %arg12[%369], %353, %mask : memref<?xui16, 1>, index, <ui16>
                    %370 = ascvec.reinterpret %353 : <ui16> -> <ui8>
                    %371 = ascvec.squeeze_store_reg %370, %271 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_16], %371 : memref<2048xui8, 1>, <ui8>
                    %372 = arith.addi %arg15, %c3_i32 : i32
                    %373 = arith.muli %372, %c256_i32 : i32
                    %374 = arith.index_cast %373 : i32 to index
                    %result0_24, %result1_25 = ascvec.loadalign_deintlv %arg11[%374] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %375 = ascvec.reinterpret %result0_24 : <bf16> -> <ui16>
                    %376 = ascvec.and %375, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %377 = ascvec.reinterpret %result1_25 : <bf16> -> <ui16>
                    %378 = ascvec.and %377, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %379 = ascvec.max %376, %378, %261 : <ui16>
                    %380 = ascvec.reduce_max_datablock %379, %261 : <ui16>
                    %381 = ascvec.shr %380, 7, %261 : <ui16> -> <ui16>
                    %382 = ascvec.cmp_ge_scalar %381, %c255_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %383 = ascvec.max_scalar %381, %c15_i16, %261 : <ui16>, i16
                    %384 = ascvec.add_scalar %383, %c-15_i16, %261 : <ui16>, i16
                    %385 = ascvec.duplicate_scalar %c255_i16, %382 merge %384 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %386 = ascvec.sub %289, %385, %261 : <ui16>
                    %387 = ascvec.cmp_ne_scalar %385, %c0_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %388 = ascvec.shl %386, 7, %387 : <ui16> -> <ui16>
                    %389 = ascvec.max %357, %385, %mask : <ui16>
                    %390 = ascvec.gather_reg %388, %265 : <ui16>, <ui16> -> <ui16>
                    %391 = ascvec.reinterpret %390 : <ui16> -> <bf16>
                    %392 = ascvec.mul %result0_24, %391, %261 : <bf16>
                    %393 = ascvec.mul %result1_25, %391, %261 : <bf16>
                    ascvec.storealign_intlv %arg11[%374], %392, %393 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %394 = ascvec.reinterpret %392 : <bf16> -> <ui16>
                    %395 = ascvec.and %394, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %396 = ascvec.reinterpret %393 : <bf16> -> <ui16>
                    %397 = ascvec.and %396, %263, %261 : <ui16>, <ui16> -> <ui16>
                    %398 = ascvec.min %395, %397, %261 : <ui16>
                    %399 = ascvec.min %367, %398, %261 : <ui16>
                    %400 = arith.muli %372, %c16_i32 : i32
                    %401 = arith.index_cast %400 : i32 to index
                    ascvec.storealign %arg12[%401], %385, %mask : memref<?xui16, 1>, index, <ui16>
                    %402 = ascvec.reinterpret %385 : <ui16> -> <ui8>
                    %403 = ascvec.squeeze_store_reg %402, %271 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_16], %403 : memref<2048xui8, 1>, <ui8>
                    %404 = ascvec.reduce_min %399, %261 : <ui16>
                    %405 = ascvec.reduce_max %389, %mask : <ui16>
                    %406 = ascvec.cmp_eq_scalar %405, %c255_i16, %261 : <ui16>, i16 -> !ascvec.mask
                    %407 = ascvec.duplicate_scalar %c0_i16, %406 merge %404 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.storealign_1st %arg13[%305], %407 : memref<2048xui16, 1>, index, <ui16>
                    %408 = ascvec.min %arg16, %407, %mask_17 : <ui16>
                    scf.yield %408 : !ascvec.vreg<ui16>
                  }
                  %c0_19 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg15 = %c0_19 to %c1 step %c1 {
                    ascvec.store_unalign_post %arg10[%c0_16] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg13[%c16_11], %273 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %257 = memref.load %alloc_10[%c16] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %258 = cannir.scalar_storage_cast %257 : ui16 -> i16
                %259 = arith.extui %258 : i16 to i64
                %260 = arith.cmpi sge, %259, %c14464_i64 : i64
                scf.if %260 {
                  ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %261 = ascscalar.bitcast_memref %108 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%108, %alloc_9, %261, %alloc_6, %255 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<8192xi32, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_11 = arith.constant 1 : i32
                    %c0_i32_12 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c64_i32 = arith.constant 64 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c1_i16 = arith.constant 1 : i16
                    %c0_i16 = arith.constant 0 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %262 = ascvec.create_mask : !ascvec.mask
                    %263 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %264 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %265 = ascvec.duplicate_scalar %c255_i16, %263 : i16 -> <ui16>
                    %266 = ascvec.duplicate_scalar %c-128_i8, %264 : i8 -> <ui8>
                    %267 = ascvec.arange %c0_i16 : i16 -> <ui16>
                    %268 = ascvec.duplicate_scalar %c1_i16, %263 : i16 -> <ui16>
                    %269 = ascvec.xor %267, %268, %263 : <ui16>, <ui16> -> <ui16>
                    scf.for %arg14 = %c0_i32_12 to %arg13 step %c1_i32_11  : i32 {
                      %270 = arith.muli %arg14, %c256_i32 : i32
                      %271 = arith.index_cast %270 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg9[%271] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %272 = arith.muli %arg14, %c64_i32 : i32
                      %273 = arith.index_cast %272 : i32 to index
                      %274 = ascvec.loadalign %arg10[%273] : memref<8192xi32, 1>, index -> <i32>
                      %275 = ascvec.reinterpret %274 : <i32> -> <ui16>
                      %276 = ascvec.reinterpret %275 : <ui16> -> <ui32>
                      %277 = ascvec.shl %276, 8, %262 : <ui32> -> <ui32>
                      %278 = ascvec.shr %276, 24, %262 : <ui32> -> <ui32>
                      %279 = ascvec.or %277, %278, %262 : <ui32>, <ui32> -> <ui32>
                      %280 = ascvec.reinterpret %279 : <ui32> -> <ui16>
                      %281 = ascvec.gather_reg %275, %269 : <ui16>, <ui16> -> <ui16>
                      %282 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %283 = ascvec.shr %281, 11, %263 : <ui16> -> <ui16>
                      %284 = ascvec.add %282, %283, %263 : <ui16>
                      %285 = ascvec.shr %284, 5, %263 : <ui16> -> <ui16>
                      %286 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %287 = ascvec.shr %280, 11, %263 : <ui16> -> <ui16>
                      %288 = ascvec.add %286, %287, %263 : <ui16>
                      %289 = ascvec.shr %288, 5, %263 : <ui16> -> <ui16>
                      %290 = ascvec.and %285, %265, %263 : <ui16>, <ui16> -> <ui16>
                      %291 = ascvec.shl %289, 8, %263 : <ui16> -> <ui16>
                      %292 = ascvec.or %290, %291, %263 : <ui16>, <ui16> -> <ui16>
                      %293 = ascvec.reinterpret %292 : <ui16> -> <ui8>
                      %294 = ascvec.add_scalar %293, %c64_i8, %264 : <ui8>, i8
                      %295 = ascvec.min_scalar %294, %c123_i8, %264 : <ui8>, i8
                      %296 = arith.muli %arg14, %c512_i32 : i32
                      %297 = arith.index_cast %296 : i32 to index
                      %result0_13, %result1_14 = ascvec.loadalign_deintlv %arg11[%297] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %298 = ascvec.and %result1_14, %266, %264 : <ui8>, <ui8> -> <ui8>
                      %299 = ascvec.or %295, %298, %264 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg12[%271], %299, %264 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
                } else {
                  scf.for %arg9 = %c0_i64 to %204 step %c16_i64  : i64 {
                    %261 = arith.addi %arg9, %c16_i64 : i64
                    %262 = arith.cmpi sgt, %261, %204 : i64
                    %263 = arith.select %262, %204, %261 : i64
                    %264 = arith.muli %arg9, %c16_i64 : i64
                    ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %265 = arith.index_cast %264 : i64 to index
                    %266 = memref.load %alloc_10[%265] : memref<2048xui16, 1>
                    ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %267 = cannir.scalar_storage_cast %266 : ui16 -> i16
                    %268 = arith.extui %267 : i16 to i64
                    %269 = arith.addi %arg9, %c4_i64 : i64
                    %270 = scf.for %arg10 = %269 to %263 step %c4_i64 iter_args(%arg11 = %268) -> (i64)  : i64 {
                      %272 = arith.muli %arg10, %c16_i64 : i64
                      ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %273 = arith.index_cast %272 : i64 to index
                      %274 = memref.load %alloc_10[%273] : memref<2048xui16, 1>
                      ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %275 = cannir.scalar_storage_cast %274 : ui16 -> i16
                      %276 = arith.extui %275 : i16 to i64
                      %277 = arith.cmpi slt, %276, %arg11 : i64
                      %278 = arith.select %277, %276, %arg11 : i64
                      scf.yield %278 : i64
                    }
                    %271 = arith.cmpi sge, %270, %c14464_i64 : i64
                    scf.if %271 {
                      %272 = arith.trunci %arg9 : i64 to i32
                      %273 = arith.trunci %263 : i64 to i32
                      ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %274 = ascscalar.bitcast_memref %108 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                      ascvec.vf(%108, %alloc_9, %274, %alloc_6, %272, %273 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                      ^bb0(%arg10: memref<32768xbf16, 1>, %arg11: memref<8192xi32, 1>, %arg12: memref<?xui8, 1>, %arg13: memref<32768xui8, 1>, %arg14: i32, %arg15: i32):
                        %c512_i32 = arith.constant 512 : i32
                        %c1_i32_11 = arith.constant 1 : i32
                        %c123_i8 = arith.constant 123 : i8
                        %c64_i8 = arith.constant 64 : i8
                        %c64_i32 = arith.constant 64 : i32
                        %c256_i32 = arith.constant 256 : i32
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
                        scf.for %arg16 = %arg14 to %arg15 step %c1_i32_11  : i32 {
                          %283 = arith.muli %arg16, %c256_i32 : i32
                          %284 = arith.index_cast %283 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg10[%284] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                          %285 = arith.muli %arg16, %c64_i32 : i32
                          %286 = arith.index_cast %285 : i32 to index
                          %287 = ascvec.loadalign %arg11[%286] : memref<8192xi32, 1>, index -> <i32>
                          %288 = ascvec.reinterpret %287 : <i32> -> <ui16>
                          %289 = ascvec.reinterpret %288 : <ui16> -> <ui32>
                          %290 = ascvec.shl %289, 8, %275 : <ui32> -> <ui32>
                          %291 = ascvec.shr %289, 24, %275 : <ui32> -> <ui32>
                          %292 = ascvec.or %290, %291, %275 : <ui32>, <ui32> -> <ui32>
                          %293 = ascvec.reinterpret %292 : <ui32> -> <ui16>
                          %294 = ascvec.gather_reg %288, %282 : <ui16>, <ui16> -> <ui16>
                          %295 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                          %296 = ascvec.shr %294, 11, %276 : <ui16> -> <ui16>
                          %297 = ascvec.add %295, %296, %276 : <ui16>
                          %298 = ascvec.shr %297, 5, %276 : <ui16> -> <ui16>
                          %299 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                          %300 = ascvec.shr %293, 11, %276 : <ui16> -> <ui16>
                          %301 = ascvec.add %299, %300, %276 : <ui16>
                          %302 = ascvec.shr %301, 5, %276 : <ui16> -> <ui16>
                          %303 = ascvec.and %298, %278, %276 : <ui16>, <ui16> -> <ui16>
                          %304 = ascvec.shl %302, 8, %276 : <ui16> -> <ui16>
                          %305 = ascvec.or %303, %304, %276 : <ui16>, <ui16> -> <ui16>
                          %306 = ascvec.reinterpret %305 : <ui16> -> <ui8>
                          %307 = ascvec.add_scalar %306, %c64_i8, %277 : <ui8>, i8
                          %308 = ascvec.min_scalar %307, %c123_i8, %277 : <ui8>, i8
                          %309 = arith.muli %arg16, %c512_i32 : i32
                          %310 = arith.index_cast %309 : i32 to index
                          %result0_12, %result1_13 = ascvec.loadalign_deintlv %arg12[%310] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                          %311 = ascvec.and %result1_13, %279, %277 : <ui8>, <ui8> -> <ui8>
                          %312 = ascvec.or %308, %311, %277 : <ui8>, <ui8> -> <ui8>
                          ascvec.storealign %arg13[%284], %312, %277 : memref<32768xui8, 1>, index, <ui8>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
                    } else {
                      %272 = arith.cmpi sge, %270, %c13568_i64 : i64
                      scf.if %272 {
                        %273 = arith.trunci %arg9 : i64 to i32
                        %274 = arith.trunci %263 : i64 to i32
                        ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %275 = ascscalar.bitcast_memref %108 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%108, %alloc_9, %275, %alloc_6, %273, %274 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg10: memref<32768xbf16, 1>, %arg11: memref<8192xi32, 1>, %arg12: memref<?xui8, 1>, %arg13: memref<32768xui8, 1>, %arg14: i32, %arg15: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_11 = arith.constant 1 : i32
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
                          %276 = ascvec.create_mask : !ascvec.mask
                          %277 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %278 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %279 = ascvec.duplicate_scalar %c255_i16, %277 : i16 -> <ui16>
                          %280 = ascvec.duplicate_scalar %c-128_i8, %278 : i8 -> <ui8>
                          %281 = ascvec.arange %c0_i16 : i16 -> <ui16>
                          %282 = ascvec.duplicate_scalar %c1_i16, %277 : i16 -> <ui16>
                          %283 = ascvec.xor %281, %282, %277 : <ui16>, <ui16> -> <ui16>
                          %284 = ascvec.duplicate_scalar %c32767_i16, %277 : i16 -> <ui16>
                          %285 = ascvec.duplicate_scalar %cst, %277 : bf16 -> <bf16>
                          scf.for %arg16 = %arg14 to %arg15 step %c1_i32_11  : i32 {
                            %286 = arith.muli %arg16, %c256_i32 : i32
                            %287 = arith.index_cast %286 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg10[%287] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %288 = arith.muli %arg16, %c64_i32 : i32
                            %289 = arith.index_cast %288 : i32 to index
                            %290 = ascvec.loadalign %arg11[%289] : memref<8192xi32, 1>, index -> <i32>
                            %291 = ascvec.reinterpret %290 : <i32> -> <ui16>
                            %292 = ascvec.reinterpret %291 : <ui16> -> <ui32>
                            %293 = ascvec.shl %292, 8, %276 : <ui32> -> <ui32>
                            %294 = ascvec.shr %292, 24, %276 : <ui32> -> <ui32>
                            %295 = ascvec.or %293, %294, %276 : <ui32>, <ui32> -> <ui32>
                            %296 = ascvec.reinterpret %295 : <ui32> -> <ui16>
                            %297 = ascvec.gather_reg %291, %283 : <ui16>, <ui16> -> <ui16>
                            %298 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %299 = ascvec.and %298, %284, %277 : <ui16>, <ui16> -> <ui16>
                            %300 = ascvec.cmp_lt_scalar %299, %c14464_i16, %277 : <ui16>, i16 -> !ascvec.mask
                            %301 = ascvec.shr %297, 11, %277 : <ui16> -> <ui16>
                            %302 = ascvec.add %299, %301, %277 : <ui16>
                            %303 = ascvec.shr %302, 5, %277 : <ui16> -> <ui16>
                            %304 = ascvec.reinterpret %299 : <ui16> -> <bf16>
                            %305 = ascvec.mul %304, %285, %300 : <bf16>
                            %306 = ascvec.cast %305, %277 {rounding = 4 : i32} : <bf16> -> <f16>
                            %307 = ascvec.cast %306, %277 {rounding = 4 : i32} : <f16> -> <i16>
                            %308 = ascvec.reinterpret %307 : <i16> -> <ui16>
                            %309 = ascvec.shr %297, 4, %277 : <ui16> -> <ui16>
                            %310 = ascvec.add %308, %309, %277 : <ui16>
                            %311 = ascvec.shr %310, 12, %277 : <ui16> -> <ui16>
                            %312 = ascvec.add_scalar %311, %c448_i16, %277 : <ui16>, i16
                            %313 = ascvec.select %312, %303, %300 : <ui16>
                            %314 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %315 = ascvec.and %314, %284, %277 : <ui16>, <ui16> -> <ui16>
                            %316 = ascvec.cmp_lt_scalar %315, %c14464_i16, %277 : <ui16>, i16 -> !ascvec.mask
                            %317 = ascvec.shr %296, 11, %277 : <ui16> -> <ui16>
                            %318 = ascvec.add %315, %317, %277 : <ui16>
                            %319 = ascvec.shr %318, 5, %277 : <ui16> -> <ui16>
                            %320 = ascvec.reinterpret %315 : <ui16> -> <bf16>
                            %321 = ascvec.mul %320, %285, %316 : <bf16>
                            %322 = ascvec.cast %321, %277 {rounding = 4 : i32} : <bf16> -> <f16>
                            %323 = ascvec.cast %322, %277 {rounding = 4 : i32} : <f16> -> <i16>
                            %324 = ascvec.reinterpret %323 : <i16> -> <ui16>
                            %325 = ascvec.shr %296, 4, %277 : <ui16> -> <ui16>
                            %326 = ascvec.add %324, %325, %277 : <ui16>
                            %327 = ascvec.shr %326, 12, %277 : <ui16> -> <ui16>
                            %328 = ascvec.add_scalar %327, %c448_i16, %277 : <ui16>, i16
                            %329 = ascvec.select %328, %319, %316 : <ui16>
                            %330 = ascvec.and %313, %279, %277 : <ui16>, <ui16> -> <ui16>
                            %331 = ascvec.shl %329, 8, %277 : <ui16> -> <ui16>
                            %332 = ascvec.or %330, %331, %277 : <ui16>, <ui16> -> <ui16>
                            %333 = ascvec.reinterpret %332 : <ui16> -> <ui8>
                            %334 = ascvec.add_scalar %333, %c64_i8, %278 : <ui8>, i8
                            %335 = ascvec.min_scalar %334, %c123_i8, %278 : <ui8>, i8
                            %336 = arith.muli %arg16, %c512_i32 : i32
                            %337 = arith.index_cast %336 : i32 to index
                            %result0_12, %result1_13 = ascvec.loadalign_deintlv %arg12[%337] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %338 = ascvec.and %result1_13, %280, %278 : <ui8>, <ui8> -> <ui8>
                            %339 = ascvec.or %335, %338, %278 : <ui8>, <ui8> -> <ui8>
                            ascvec.storealign %arg13[%287], %339, %278 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
                      } else {
                        %273 = arith.trunci %arg9 : i64 to i32
                        %274 = arith.trunci %263 : i64 to i32
                        ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %275 = ascscalar.bitcast_memref %108 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%108, %alloc_9, %275, %256, %alloc_6, %273, %274 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg10: memref<32768xbf16, 1>, %arg11: memref<8192xi32, 1>, %arg12: memref<?xui8, 1>, %arg13: memref<?xui16, 1>, %arg14: memref<32768xui8, 1>, %arg15: i32, %arg16: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_11 = arith.constant 1 : i32
                          %c127_i8 = arith.constant 127 : i8
                          %c16_i32 = arith.constant 16 : i32
                          %c0_i16 = arith.constant 0 : i16
                          %c123_i8 = arith.constant 123 : i8
                          %c-1_i8 = arith.constant -1 : i8
                          %c60_i8 = arith.constant 60 : i8
                          %cst = arith.constant 6.10351526E-5 : f32
                          %c64_i32 = arith.constant 64 : i32
                          %c32767_i16 = arith.constant 32767 : i16
                          %c0_12 = arith.constant 0 : index
                          %c256_i32 = arith.constant 256 : i32
                          %c65535_i32 = arith.constant 65535 : i32
                          %c3_i8 = arith.constant 3 : i8
                          %c-128_i8 = arith.constant -128 : i8
                          %c0_i8 = arith.constant 0 : i8
                          %c8_i32 = arith.constant 8 : i32
                          %276 = ascvec.create_mask : !ascvec.mask
                          %277 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %278 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                          %279 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %280 = ascvec.duplicate_scalar %c-128_i8, %278 : i8 -> <ui8>
                          %281 = ascvec.shr %279, 5, %278 : <ui8> -> <ui8>
                          %282 = ascvec.duplicate_scalar %c3_i8, %278 : i8 -> <ui8>
                          %283 = ascvec.and %279, %282, %278 : <ui8>, <ui8> -> <ui8>
                          %284 = ascvec.shl %283, 6, %278 : <ui8> -> <ui8>
                          %285 = ascvec.shr %279, 2, %278 : <ui8> -> <ui8>
                          %286 = ascvec.or %284, %285, %278 : <ui8>, <ui8> -> <ui8>
                          %287 = ascvec.duplicate_scalar %c65535_i32, %276 : i32 -> <ui32>
                          scf.for %arg17 = %arg15 to %arg16 step %c1_i32_11  : i32 {
                            %288 = arith.muli %arg17, %c256_i32 : i32
                            %289 = arith.index_cast %288 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg10[%289] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %290 = ascvec.duplicate_scalar %c32767_i16, %277 : i16 -> <ui16>
                            %291 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %292 = ascvec.and %291, %290, %277 : <ui16>, <ui16> -> <ui16>
                            %293 = ascvec.reinterpret %292 : <ui16> -> <bf16>
                            %294 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %295 = ascvec.and %294, %290, %277 : <ui16>, <ui16> -> <ui16>
                            %296 = ascvec.reinterpret %295 : <ui16> -> <bf16>
                            %297 = ascvec.cast %293, %277 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %298 = ascvec.cast %293, %277 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %299 = ascvec.cast %296, %277 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %300 = ascvec.cast %296, %277 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %301 = arith.muli %arg17, %c64_i32 : i32
                            %302 = arith.index_cast %301 : i32 to index
                            %303 = ascvec.loadalign %arg11[%302] : memref<8192xi32, 1>, index -> <i32>
                            %304 = ascvec.reinterpret %303 : <i32> -> <ui32>
                            %305 = ascvec.shl %304, 8, %276 : <ui32> -> <ui32>
                            %306 = ascvec.shr %304, 24, %276 : <ui32> -> <ui32>
                            %307 = ascvec.or %305, %306, %276 : <ui32>, <ui32> -> <ui32>
                            %308 = ascvec.shr %304, 16, %276 : <ui32> -> <ui32>
                            %309 = ascvec.and %307, %287, %276 : <ui32>, <ui32> -> <ui32>
                            %310 = ascvec.and %304, %287, %276 : <ui32>, <ui32> -> <ui32>
                            %311 = ascvec.shr %307, 16, %276 : <ui32> -> <ui32>
                            %312 = ascvec.reinterpret %297 : <f32> -> <ui32>
                            %313 = ascvec.reinterpret %312 : <ui32> -> <f32>
                            %314 = ascvec.max_scalar %313, %cst, %276 : <f32>, f32
                            %315 = ascvec.add %313, %314, %276 : <f32>
                            %316 = ascvec.reinterpret %315 : <f32> -> <ui32>
                            %317 = ascvec.shr %316, 5, %276 : <ui32> -> <ui32>
                            %318 = ascvec.add %317, %308, %276 : <ui32>
                            %319 = ascvec.reinterpret %299 : <f32> -> <ui32>
                            %320 = ascvec.reinterpret %319 : <ui32> -> <f32>
                            %321 = ascvec.max_scalar %320, %cst, %276 : <f32>, f32
                            %322 = ascvec.add %320, %321, %276 : <f32>
                            %323 = ascvec.reinterpret %322 : <f32> -> <ui32>
                            %324 = ascvec.shr %323, 5, %276 : <ui32> -> <ui32>
                            %325 = ascvec.add %324, %309, %276 : <ui32>
                            %326 = ascvec.reinterpret %298 : <f32> -> <ui32>
                            %327 = ascvec.reinterpret %326 : <ui32> -> <f32>
                            %328 = ascvec.max_scalar %327, %cst, %276 : <f32>, f32
                            %329 = ascvec.add %327, %328, %276 : <f32>
                            %330 = ascvec.reinterpret %329 : <f32> -> <ui32>
                            %331 = ascvec.shr %330, 5, %276 : <ui32> -> <ui32>
                            %332 = ascvec.add %331, %310, %276 : <ui32>
                            %333 = ascvec.reinterpret %300 : <f32> -> <ui32>
                            %334 = ascvec.reinterpret %333 : <ui32> -> <f32>
                            %335 = ascvec.max_scalar %334, %cst, %276 : <f32>, f32
                            %336 = ascvec.add %334, %335, %276 : <f32>
                            %337 = ascvec.reinterpret %336 : <f32> -> <ui32>
                            %338 = ascvec.shr %337, 5, %276 : <ui32> -> <ui32>
                            %339 = ascvec.add %338, %311, %276 : <ui32>
                            %340 = ascvec.reinterpret %318 : <ui32> -> <ui16>
                            %341 = ascvec.reinterpret %325 : <ui32> -> <ui16>
                            %result0_13, %result1_14 = ascvec.deintlv %340, %341 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %342 = ascvec.reinterpret %332 : <ui32> -> <ui16>
                            %343 = ascvec.reinterpret %339 : <ui32> -> <ui16>
                            %result0_15, %result1_16 = ascvec.deintlv %342, %343 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %344 = ascvec.reinterpret %result1_14 : <ui16> -> <ui8>
                            %345 = ascvec.reinterpret %result1_16 : <ui16> -> <ui8>
                            %result0_17, %result1_18 = ascvec.deintlv %344, %345 : <ui8>, <ui8> -> <ui8>, <ui8>
                            %346 = ascvec.gather_reg %result0_17, %286 : <ui8>, <ui8> -> <ui8>
                            %347 = ascvec.add_scalar %346, %c60_i8, %278 : <ui8>, i8
                            %348 = ascvec.cmp_ne_scalar %347, %c-1_i8, %278 : <ui8>, i8 -> !ascvec.mask
                            %349 = ascvec.min_scalar %347, %c123_i8, %348 : <ui8>, i8
                            %350 = arith.muli %arg17, %c512_i32 : i32
                            %351 = arith.index_cast %350 : i32 to index
                            %result0_19, %result1_20 = ascvec.loadalign_deintlv %arg12[%351] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %352 = ascvec.and %result1_20, %280, %278 : <ui8>, <ui8> -> <ui8>
                            %353 = ascvec.or %349, %352, %278 : <ui8>, <ui8> -> <ui8>
                            %354 = ascvec.arange %c0_i16 : i16 -> <ui16>
                            %355 = arith.muli %arg17, %c16_i32 : i32
                            %356 = arith.trunci %355 : i32 to i16
                            %357 = ascvec.add_scalar %354, %356, %277 : <ui16>, i16
                            %358 = ascvec.gather %arg13[%c0_12], %357, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                            %359 = ascvec.pack %358 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                            %360 = ascvec.gather_reg %359, %281 : <ui8>, <ui8> -> <ui8>
                            %361 = ascvec.cmp_eq_scalar %360, %c-1_i8, %278 : <ui8>, i8 -> !ascvec.mask
                            %362 = ascvec.duplicate_scalar %c127_i8, %361 merge %353 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                            ascvec.storealign %arg14[%289], %362, %278 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
                      }
                    }
                  }
                }
              } else {
                %204 = arith.divsi %67, %c256_i64 : i64
                %205 = arith.addi %67, %c1023_i64 : i64
                %206 = arith.divsi %205, %c1024_i64 : i64
                %207 = arith.trunci %1 : i64 to i32
                %208 = arith.extui %207 : i32 to i64
                %209 = arith.subi %1, %208 : i64
                %210 = arith.remsi %209, %c4294967296_i64 : i64
                %211 = arith.cmpi ne, %210, %c0_i64 : i64
                %212 = arith.cmpi slt, %210, %c0_i64 : i64
                %213 = arith.andi %211, %212 : i1
                %214 = arith.divsi %209, %c4294967296_i64 : i64
                %215 = arith.extui %213 : i1 to i64
                %216 = arith.subi %214, %215 : i64
                %217 = arith.trunci %216 : i64 to i32
                %218 = arith.divui %109, %c16_i64 : i64
                %219 = arith.trunci %218 : i64 to i32
                %220 = arith.extui %219 : i32 to i64
                %221 = arith.subi %218, %220 : i64
                %222 = arith.remsi %221, %c4294967296_i64 : i64
                %223 = arith.cmpi ne, %222, %c0_i64 : i64
                %224 = arith.cmpi slt, %222, %c0_i64 : i64
                %225 = arith.andi %223, %224 : i1
                %226 = arith.divsi %221, %c4294967296_i64 : i64
                %227 = arith.extui %225 : i1 to i64
                %228 = arith.subi %226, %227 : i64
                %229 = arith.trunci %228 : i64 to i32
                %230 = arith.trunci %4 : i64 to i32
                %231 = arith.extui %230 : i32 to i64
                %232 = arith.subi %4, %231 : i64
                %233 = arith.remsi %232, %c4294967296_i64 : i64
                %234 = arith.cmpi ne, %233, %c0_i64 : i64
                %235 = arith.cmpi slt, %233, %c0_i64 : i64
                %236 = arith.andi %234, %235 : i1
                %237 = arith.divsi %232, %c4294967296_i64 : i64
                %238 = arith.extui %236 : i1 to i64
                %239 = arith.subi %237, %238 : i64
                %240 = arith.trunci %239 : i64 to i32
                %241 = arith.divui %230, %c4_i32 : i32
                %242 = arith.remui %240, %c4_i32 : i32
                %243 = arith.muli %242, %c1073741824_i32 : i32
                %244 = arith.addi %241, %243 : i32
                %245 = arith.divui %240, %c4_i32 : i32
                %246 = arith.muli %206, %c64_i64 : i64
                %247 = arith.divui %218, %c16384_i64 : i64
                %248 = arith.addi %218, %246 : i64
                %249 = arith.subi %248, %c1_i64 : i64
                %250 = arith.divui %249, %c16384_i64 : i64
                %251 = arith.cmpi eq, %247, %250 : i64
                scf.if %251 {
                  %257 = arith.trunci %247 : i64 to i32
                  %258 = arith.extui %257 : i32 to i64
                  %259 = arith.subi %247, %258 : i64
                  %260 = arith.remsi %259, %c4294967296_i64 : i64
                  %261 = arith.cmpi ne, %260, %c0_i64 : i64
                  %262 = arith.cmpi slt, %260, %c0_i64 : i64
                  %263 = arith.andi %261, %262 : i1
                  %264 = arith.divsi %259, %c4294967296_i64 : i64
                  %265 = arith.extui %263 : i1 to i64
                  %266 = arith.subi %264, %265 : i64
                  %267 = arith.trunci %266 : i64 to i32
                  %268 = arith.addi %244, %257 : i32
                  %269 = arith.addi %245, %267 : i32
                  %270 = arith.cmpi ult, %268, %244 : i32
                  %271 = arith.extui %270 : i1 to i32
                  %272 = arith.addi %269, %271 : i32
                  %273 = arith.extui %268 : i32 to i64
                  %274 = arith.muli %273, %c3449720151_i64 : i64
                  %275 = arith.trunci %274 : i64 to i32
                  %276 = arith.extui %275 : i32 to i64
                  %277 = arith.subi %274, %276 : i64
                  %278 = arith.remsi %277, %c4294967296_i64 : i64
                  %279 = arith.cmpi ne, %278, %c0_i64 : i64
                  %280 = arith.cmpi slt, %278, %c0_i64 : i64
                  %281 = arith.andi %279, %280 : i1
                  %282 = arith.divsi %277, %c4294967296_i64 : i64
                  %283 = arith.extui %281 : i1 to i64
                  %284 = arith.subi %282, %283 : i64
                  %285 = arith.trunci %284 : i64 to i32
                  %286 = arith.xori %272, %217 : i32
                  %287 = arith.extui %286 : i32 to i64
                  %288 = arith.muli %287, %c3449720151_i64 : i64
                  %289 = arith.trunci %288 : i64 to i32
                  %290 = arith.extui %289 : i32 to i64
                  %291 = arith.subi %288, %290 : i64
                  %292 = arith.remsi %291, %c4294967296_i64 : i64
                  %293 = arith.cmpi ne, %292, %c0_i64 : i64
                  %294 = arith.cmpi slt, %292, %c0_i64 : i64
                  %295 = arith.andi %293, %294 : i1
                  %296 = arith.divsi %291, %c4294967296_i64 : i64
                  %297 = arith.extui %295 : i1 to i64
                  %298 = arith.subi %296, %297 : i64
                  %299 = arith.trunci %298 : i64 to i32
                  %300 = arith.addi %207, %c-1640531527_i32 : i32
                  %301 = arith.addi %217, %c-1150833019_i32 : i32
                  %302 = arith.xori %299, %275 : i32
                  %303 = arith.xori %302, %300 : i32
                  %304 = arith.extui %303 : i32 to i64
                  %305 = arith.muli %304, %c3528531795_i64 : i64
                  %306 = arith.trunci %305 : i64 to i32
                  %307 = arith.extui %306 : i32 to i64
                  %308 = arith.subi %305, %307 : i64
                  %309 = arith.remsi %308, %c4294967296_i64 : i64
                  %310 = arith.cmpi ne, %309, %c0_i64 : i64
                  %311 = arith.cmpi slt, %309, %c0_i64 : i64
                  %312 = arith.andi %310, %311 : i1
                  %313 = arith.divsi %308, %c4294967296_i64 : i64
                  %314 = arith.extui %312 : i1 to i64
                  %315 = arith.subi %313, %314 : i64
                  %316 = arith.trunci %315 : i64 to i32
                  %317 = arith.xori %285, %207 : i32
                  %318 = arith.addi %207, %c1013904242_i32 : i32
                  %319 = arith.xori %289, %318 : i32
                  %320 = arith.addi %217, %c1993301258_i32 : i32
                  %321 = arith.xori %316, %320 : i32
                  ascsimt.vf(%219, %317, %301, %319, %321, %207, %217, %306, %alloc_9, %246 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: memref<8192xi32, 1>, %arg18: i64):
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
                    %c16384_i32 = arith.constant 16384 : i32
                    %322 = ascsimt.thread_idx {ascsimt.scope = 21 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %323 = arith.extsi %322 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg19 = %323 to %arg18 step %c256_i64_11  : i64 {
                      %324 = arith.trunci %arg19 : i64 to i32
                      %325 = arith.addi %arg9, %324 : i32
                      %326 = arith.remui %325, %c16384_i32 : i32
                      %327 = arith.xori %326, %arg10 : i32
                      %328 = arith.muli %327, %c-766435501_i32 : i32
                      %329 = "ascsimt.bitcast"(%327) : (i32) -> ui32
                      %330 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %331 = ascsimt.math_binary %329, %330, "mulhi" : ui32, ui32 -> ui32
                      %332 = "ascsimt.bitcast"(%331) : (ui32) -> i32
                      %333 = arith.xori %332, %arg11 : i32
                      %334 = arith.muli %333, %c-845247145_i32 : i32
                      %335 = "ascsimt.bitcast"(%333) : (i32) -> ui32
                      %336 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %337 = ascsimt.math_binary %335, %336, "mulhi" : ui32, ui32 -> ui32
                      %338 = "ascsimt.bitcast"(%337) : (ui32) -> i32
                      %339 = arith.xori %338, %arg12 : i32
                      %340 = arith.xori %328, %arg13 : i32
                      %341:4 = scf.for %arg20 = %c3_i32 to %c10_i32 step %c1_i32_16 iter_args(%arg21 = %339, %arg22 = %334, %arg23 = %340, %arg24 = %arg16) -> (i32, i32, i32, i32)  : i32 {
                        %350 = arith.muli %arg21, %c-766435501_i32 : i32
                        %351 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %352 = ascsimt.math_binary %351, %330, "mulhi" : ui32, ui32 -> ui32
                        %353 = "ascsimt.bitcast"(%352) : (ui32) -> i32
                        %354 = arith.muli %arg23, %c-845247145_i32 : i32
                        %355 = "ascsimt.bitcast"(%arg23) : (i32) -> ui32
                        %356 = ascsimt.math_binary %355, %336, "mulhi" : ui32, ui32 -> ui32
                        %357 = "ascsimt.bitcast"(%356) : (ui32) -> i32
                        %358 = arith.muli %arg20, %c-1640531527_i32_18 : i32
                        %359 = arith.addi %arg14, %358 : i32
                        %360 = arith.muli %arg20, %c-1150833019_i32_17 : i32
                        %361 = arith.addi %arg15, %360 : i32
                        %362 = arith.xori %357, %arg22 : i32
                        %363 = arith.xori %362, %359 : i32
                        %364 = arith.xori %353, %arg24 : i32
                        %365 = arith.xori %364, %361 : i32
                        scf.yield %363, %354, %365, %350 : i32, i32, i32, i32
                      }
                      %342 = arith.muli %arg19, %c4_i64_15 : i64
                      %343 = arith.index_cast %342 : i64 to index
                      ascsimt.store %arg17[%343], %341#0 : memref<8192xi32, 1>, i32
                      %344 = arith.addi %342, %c1_i64_14 : i64
                      %345 = arith.index_cast %344 : i64 to index
                      ascsimt.store %arg17[%345], %341#1 : memref<8192xi32, 1>, i32
                      %346 = arith.addi %342, %c2_i64_13 : i64
                      %347 = arith.index_cast %346 : i64 to index
                      ascsimt.store %arg17[%347], %341#2 : memref<8192xi32, 1>, i32
                      %348 = arith.addi %342, %c3_i64_12 : i64
                      %349 = arith.index_cast %348 : i64 to index
                      ascsimt.store %arg17[%349], %341#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%219, %229, %244, %245, %207, %217, %alloc_9, %246 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
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
                    %c16384_i32 = arith.constant 16384 : i32
                    %257 = ascsimt.thread_idx {ascsimt.scope = 22 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %258 = arith.extsi %257 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg17 = %258 to %arg16 step %c256_i64_11  : i64 {
                      %259 = arith.trunci %arg17 : i64 to i32
                      %260 = arith.addi %arg9, %259 : i32
                      %261 = arith.cmpi ult, %260, %arg9 : i32
                      %262 = arith.extui %261 : i1 to i32
                      %263 = arith.addi %arg10, %262 : i32
                      %264 = arith.divui %260, %c16384_i32 : i32
                      %265 = arith.remui %263, %c16384_i32 : i32
                      %266 = arith.muli %265, %c262144_i32 : i32
                      %267 = arith.addi %264, %266 : i32
                      %268 = arith.divui %263, %c16384_i32 : i32
                      %269 = arith.addi %arg11, %267 : i32
                      %270 = arith.addi %arg12, %268 : i32
                      %271 = arith.cmpi ult, %269, %arg11 : i32
                      %272 = arith.extui %271 : i1 to i32
                      %273 = arith.addi %270, %272 : i32
                      %274 = arith.remui %260, %c16384_i32 : i32
                      %275:4 = scf.for %arg18 = %c0_i32_17 to %c10_i32 step %c1_i32_16 iter_args(%arg19 = %c0_i32_17, %arg20 = %274, %arg21 = %269, %arg22 = %273) -> (i32, i32, i32, i32)  : i32 {
                        %284 = arith.muli %arg19, %c-766435501_i32 : i32
                        %285 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %286 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %287 = ascsimt.math_binary %285, %286, "mulhi" : ui32, ui32 -> ui32
                        %288 = "ascsimt.bitcast"(%287) : (ui32) -> i32
                        %289 = arith.muli %arg21, %c-845247145_i32 : i32
                        %290 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %291 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %292 = ascsimt.math_binary %290, %291, "mulhi" : ui32, ui32 -> ui32
                        %293 = "ascsimt.bitcast"(%292) : (ui32) -> i32
                        %294 = arith.muli %arg18, %c-1640531527_i32_19 : i32
                        %295 = arith.addi %arg13, %294 : i32
                        %296 = arith.muli %arg18, %c-1150833019_i32_18 : i32
                        %297 = arith.addi %arg14, %296 : i32
                        %298 = arith.xori %293, %arg20 : i32
                        %299 = arith.xori %298, %295 : i32
                        %300 = arith.xori %288, %arg22 : i32
                        %301 = arith.xori %300, %297 : i32
                        scf.yield %299, %289, %301, %284 : i32, i32, i32, i32
                      }
                      %276 = arith.muli %arg17, %c4_i64_15 : i64
                      %277 = arith.index_cast %276 : i64 to index
                      ascsimt.store %arg15[%277], %275#0 : memref<8192xi32, 1>, i32
                      %278 = arith.addi %276, %c1_i64_14 : i64
                      %279 = arith.index_cast %278 : i64 to index
                      ascsimt.store %arg15[%279], %275#1 : memref<8192xi32, 1>, i32
                      %280 = arith.addi %276, %c2_i64_13 : i64
                      %281 = arith.index_cast %280 : i64 to index
                      ascsimt.store %arg15[%281], %275#2 : memref<8192xi32, 1>, i32
                      %282 = arith.addi %276, %c3_i64_12 : i64
                      %283 = arith.index_cast %282 : i64 to index
                      ascsimt.store %arg15[%283], %275#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %252 = arith.trunci %49 : i64 to i32
                %253 = arith.divsi %252, %c2_i32 : i32
                %254 = arith.trunci %253 : i32 to i8
                %255 = arith.trunci %204 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                %256 = ascscalar.bitcast_memref %108 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                ascvec.vf(%254, %alloc_7, %108, %256, %alloc_9, %alloc_6, %255 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui8, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, i32) {
                ^bb0(%arg9: i8, %arg10: memref<2048xui8, 1>, %arg11: memref<32768xbf16, 1>, %arg12: memref<?xui8, 1>, %arg13: memref<8192xi32, 1>, %arg14: memref<32768xui8, 1>, %arg15: i32):
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
                  %c32_i8 = arith.constant 32 : i8
                  %c65535_i32 = arith.constant 65535 : i32
                  %c3_i8 = arith.constant 3 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c-128_i8 = arith.constant -128 : i8
                  %257 = ascvec.create_mask : !ascvec.mask
                  %258 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %259 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %260 = ascvec.duplicate_scalar %c-128_i8, %259 : i8 -> <ui8>
                  %261 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %262 = ascvec.duplicate_scalar %c3_i8, %259 : i8 -> <ui8>
                  %263 = ascvec.and %261, %262, %259 : <ui8>, <ui8> -> <ui8>
                  %264 = ascvec.shl %263, 6, %259 : <ui8> -> <ui8>
                  %265 = ascvec.shr %261, 2, %259 : <ui8> -> <ui8>
                  %266 = ascvec.or %264, %265, %259 : <ui8>, <ui8> -> <ui8>
                  %267 = ascvec.duplicate_scalar %c65535_i32, %257 : i32 -> <ui32>
                  %268 = ascvec.cmp_lt_scalar %261, %c32_i8, %259 : <ui8>, i8 -> !ascvec.mask
                  %269 = ascvec.cmp_lt_scalar %263, %arg9, %259 : <ui8>, i8 -> !ascvec.mask
                  %270 = ascvec.mask_and %268, %269, %259
                  ascvec.store_unalign_begin %arg10 : memref<2048xui8, 1>
                  scf.for %arg16 = %c0_i32_12 to %arg15 step %c1_i32_11  : i32 {
                    %271 = arith.muli %arg16, %c256_i32 : i32
                    %272 = arith.index_cast %271 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg11[%272] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %273 = ascvec.duplicate_scalar %c32767_i16, %258 : i16 -> <ui16>
                    %274 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %275 = ascvec.and %274, %273, %258 : <ui16>, <ui16> -> <ui16>
                    %276 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %277 = ascvec.and %276, %273, %258 : <ui16>, <ui16> -> <ui16>
                    %278 = ascvec.max %275, %277, %258 : <ui16>
                    %279 = ascvec.reduce_max_datablock %278, %258 : <ui16>
                    %280 = ascvec.unpack %279 {part = "lower"} : <ui16> -> <ui32>
                    %281 = ascvec.shl %280, 16, %257 : <ui32> -> <ui32>
                    %282 = arith.muli %arg16, %c512_i32 : i32
                    %283 = arith.index_cast %282 : i32 to index
                    %result0_15, %result1_16 = ascvec.loadalign_deintlv %arg12[%283] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                    %284 = ascvec.and %result1_16, %260, %259 : <ui8>, <ui8> -> <ui8>
                    %285 = ascvec.reinterpret %284 : <ui8> -> <ui32>
                    %286 = ascvec.shr %281, 23, %257 : <ui32> -> <ui32>
                    %287 = ascvec.cmp_ge_scalar %286, %c255_i32, %257 : <ui32>, i32 -> !ascvec.mask
                    %288 = ascvec.max_scalar %286, %c15_i32, %257 : <ui32>, i32
                    %289 = ascvec.add_scalar %288, %c-15_i32, %257 : <ui32>, i32
                    %290 = ascvec.duplicate_scalar %c255_i32, %287 merge %289 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                    %291 = ascvec.duplicate_scalar %c254_i32, %257 : i32 -> <ui32>
                    %292 = ascvec.sub %291, %290, %257 : <ui32>
                    %293 = ascvec.cmp_ne_scalar %290, %c0_i32_12, %257 : <ui32>, i32 -> !ascvec.mask
                    %294 = ascvec.shl %292, 23, %293 : <ui32> -> <ui32>
                    %295 = ascvec.reinterpret %294 : <ui32> -> <f32>
                    %296 = ascvec.reinterpret %295 : <f32> -> <ui32>
                    %297 = ascvec.shr %296, 16, %257 : <ui32> -> <ui32>
                    %298 = ascvec.pack %297 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                    %299 = ascvec.arange %c0_i16 : i16 -> <ui16>
                    %300 = ascvec.shr %299, 4, %258 : <ui16> -> <ui16>
                    %301 = ascvec.gather_reg %298, %300 : <ui16>, <ui16> -> <ui16>
                    %302 = ascvec.reinterpret %301 : <ui16> -> <bf16>
                    %303 = ascvec.reinterpret %275 : <ui16> -> <bf16>
                    %304 = ascvec.mul %303, %302, %258 : <bf16>
                    %305 = ascvec.reinterpret %277 : <ui16> -> <bf16>
                    %306 = ascvec.mul %305, %302, %258 : <bf16>
                    %307 = ascvec.cast %304, %258 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %308 = ascvec.cast %304, %258 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %309 = ascvec.cast %306, %258 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %310 = ascvec.cast %306, %258 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %311 = ascvec.reinterpret %302 : <bf16> -> <ui32>
                    %312 = arith.muli %arg16, %c64_i32 : i32
                    %313 = arith.index_cast %312 : i32 to index
                    %314 = ascvec.loadalign %arg13[%313] : memref<8192xi32, 1>, index -> <i32>
                    %315 = ascvec.reinterpret %314 : <i32> -> <ui32>
                    %316 = ascvec.shl %315, 8, %257 : <ui32> -> <ui32>
                    %317 = ascvec.shr %315, 24, %257 : <ui32> -> <ui32>
                    %318 = ascvec.or %316, %317, %257 : <ui32>, <ui32> -> <ui32>
                    %319 = ascvec.shr %315, 16, %257 : <ui32> -> <ui32>
                    %320 = ascvec.and %318, %267, %257 : <ui32>, <ui32> -> <ui32>
                    %321 = ascvec.and %315, %267, %257 : <ui32>, <ui32> -> <ui32>
                    %322 = ascvec.shr %318, 16, %257 : <ui32> -> <ui32>
                    %323 = ascvec.reinterpret %307 : <f32> -> <ui32>
                    %324 = ascvec.reinterpret %323 : <ui32> -> <f32>
                    %325 = ascvec.max_scalar %324, %cst, %257 : <f32>, f32
                    %326 = ascvec.add %324, %325, %257 : <f32>
                    %327 = ascvec.reinterpret %326 : <f32> -> <ui32>
                    %328 = ascvec.shr %327, 5, %257 : <ui32> -> <ui32>
                    %329 = ascvec.add %328, %319, %257 : <ui32>
                    %330 = ascvec.reinterpret %309 : <f32> -> <ui32>
                    %331 = ascvec.reinterpret %330 : <ui32> -> <f32>
                    %332 = ascvec.max_scalar %331, %cst, %257 : <f32>, f32
                    %333 = ascvec.add %331, %332, %257 : <f32>
                    %334 = ascvec.reinterpret %333 : <f32> -> <ui32>
                    %335 = ascvec.shr %334, 5, %257 : <ui32> -> <ui32>
                    %336 = ascvec.add %335, %320, %257 : <ui32>
                    %337 = ascvec.reinterpret %308 : <f32> -> <ui32>
                    %338 = ascvec.reinterpret %337 : <ui32> -> <f32>
                    %339 = ascvec.max_scalar %338, %cst, %257 : <f32>, f32
                    %340 = ascvec.add %338, %339, %257 : <f32>
                    %341 = ascvec.reinterpret %340 : <f32> -> <ui32>
                    %342 = ascvec.shr %341, 5, %257 : <ui32> -> <ui32>
                    %343 = ascvec.add %342, %321, %257 : <ui32>
                    %344 = ascvec.reinterpret %310 : <f32> -> <ui32>
                    %345 = ascvec.reinterpret %344 : <ui32> -> <f32>
                    %346 = ascvec.max_scalar %345, %cst, %257 : <f32>, f32
                    %347 = ascvec.add %345, %346, %257 : <f32>
                    %348 = ascvec.reinterpret %347 : <f32> -> <ui32>
                    %349 = ascvec.shr %348, 5, %257 : <ui32> -> <ui32>
                    %350 = ascvec.add %349, %322, %257 : <ui32>
                    %351 = ascvec.reinterpret %329 : <ui32> -> <ui16>
                    %352 = ascvec.reinterpret %336 : <ui32> -> <ui16>
                    %result0_17, %result1_18 = ascvec.deintlv %351, %352 : <ui16>, <ui16> -> <ui16>, <ui16>
                    %353 = ascvec.reinterpret %343 : <ui32> -> <ui16>
                    %354 = ascvec.reinterpret %350 : <ui32> -> <ui16>
                    %result0_19, %result1_20 = ascvec.deintlv %353, %354 : <ui16>, <ui16> -> <ui16>, <ui16>
                    %355 = ascvec.reinterpret %result1_18 : <ui16> -> <ui8>
                    %356 = ascvec.reinterpret %result1_20 : <ui16> -> <ui8>
                    %result0_21, %result1_22 = ascvec.deintlv %355, %356 : <ui8>, <ui8> -> <ui8>, <ui8>
                    %357 = ascvec.gather_reg %result0_21, %266 : <ui8>, <ui8> -> <ui8>
                    %358 = ascvec.reinterpret %357 : <ui8> -> <ui32>
                    %359 = ascvec.reinterpret %358 : <ui32> -> <ui8>
                    %360 = ascvec.add_scalar %359, %c60_i8, %259 : <ui8>, i8
                    %361 = ascvec.cmp_ne_scalar %360, %c-1_i8, %259 : <ui8>, i8 -> !ascvec.mask
                    %362 = ascvec.min_scalar %360, %c123_i8, %361 : <ui8>, i8
                    %363 = ascvec.reinterpret %362 : <ui8> -> <ui32>
                    %364 = ascvec.or %363, %285, %257 : <ui32>, <ui32> -> <ui32>
                    %365 = ascvec.reinterpret %311 : <ui32> -> <ui32>
                    %366 = ascvec.cmp_eq_scalar %365, %c-8323200_i32, %257 : <ui32>, i32 -> !ascvec.mask
                    %367 = ascvec.duplicate_scalar %c2139062143_i32, %366 merge %364 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                    %368 = ascvec.reinterpret %367 : <ui32> -> <ui8>
                    ascvec.storealign %arg14[%272], %368, %259 : memref<32768xui8, 1>, index, <ui8>
                    %369 = ascvec.reinterpret %290 : <ui32> -> <ui8>
                    %370 = ascvec.squeeze_store_reg %369, %270 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_13], %370 : memref<2048xui8, 1>, <ui8>
                  }
                  %c0_14 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg16 = %c0_14 to %c1 step %c1 {
                    ascvec.store_unalign_post %arg10[%c0_13] : memref<2048xui8, 1>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              }
            }
          } else {
            %193 = arith.remsi %arg3, %c256_i64 : i64
            %194 = arith.cmpi ne, %193, %c0_i64 : i64
            %195 = arith.cmpi slt, %193, %c0_i64 : i64
            %196 = arith.andi %194, %195 : i1
            %197 = arith.select %196, %c256_i64, %c0_i64 : i64
            %198 = arith.addi %193, %197 overflow<nsw> : i64
            %199 = arith.cmpi eq, %198, %c0_i64 : i64
            scf.if %199 {
              %200 = arith.addi %67, %c63_i64 : i64
              %201 = arith.remsi %200, %c64_i64 : i64
              %202 = arith.cmpi ne, %201, %c0_i64 : i64
              %203 = arith.cmpi slt, %201, %c0_i64 : i64
              %204 = arith.andi %202, %203 : i1
              %205 = arith.divsi %200, %c64_i64 : i64
              %206 = arith.extui %204 : i1 to i64
              %207 = arith.subi %205, %206 : i64
              %208 = arith.addi %67, %c1023_i64 : i64
              %209 = arith.remsi %208, %c1024_i64 : i64
              %210 = arith.cmpi ne, %209, %c0_i64 : i64
              %211 = arith.cmpi slt, %209, %c0_i64 : i64
              %212 = arith.andi %210, %211 : i1
              %213 = arith.divsi %208, %c1024_i64 : i64
              %214 = arith.extui %212 : i1 to i64
              %215 = arith.subi %213, %214 : i64
              %216 = arith.trunci %1 : i64 to i32
              %217 = arith.extui %216 : i32 to i64
              %218 = arith.subi %1, %217 : i64
              %219 = arith.remsi %218, %c4294967296_i64 : i64
              %220 = arith.cmpi ne, %219, %c0_i64 : i64
              %221 = arith.cmpi slt, %219, %c0_i64 : i64
              %222 = arith.andi %220, %221 : i1
              %223 = arith.divsi %218, %c4294967296_i64 : i64
              %224 = arith.extui %222 : i1 to i64
              %225 = arith.subi %223, %224 : i64
              %226 = arith.trunci %225 : i64 to i32
              %227 = arith.divui %109, %c32_i64 : i64
              %228 = arith.trunci %227 : i64 to i32
              %229 = arith.extui %228 : i32 to i64
              %230 = arith.subi %227, %229 : i64
              %231 = arith.remsi %230, %c4294967296_i64 : i64
              %232 = arith.cmpi ne, %231, %c0_i64 : i64
              %233 = arith.cmpi slt, %231, %c0_i64 : i64
              %234 = arith.andi %232, %233 : i1
              %235 = arith.divsi %230, %c4294967296_i64 : i64
              %236 = arith.extui %234 : i1 to i64
              %237 = arith.subi %235, %236 : i64
              %238 = arith.trunci %237 : i64 to i32
              %239 = arith.trunci %4 : i64 to i32
              %240 = arith.extui %239 : i32 to i64
              %241 = arith.subi %4, %240 : i64
              %242 = arith.remsi %241, %c4294967296_i64 : i64
              %243 = arith.cmpi ne, %242, %c0_i64 : i64
              %244 = arith.cmpi slt, %242, %c0_i64 : i64
              %245 = arith.andi %243, %244 : i1
              %246 = arith.divsi %241, %c4294967296_i64 : i64
              %247 = arith.extui %245 : i1 to i64
              %248 = arith.subi %246, %247 : i64
              %249 = arith.trunci %248 : i64 to i32
              %250 = arith.divui %239, %c4_i32 : i32
              %251 = arith.remui %249, %c4_i32 : i32
              %252 = arith.muli %251, %c1073741824_i32 : i32
              %253 = arith.addi %250, %252 : i32
              %254 = arith.divui %249, %c4_i32 : i32
              %255 = arith.muli %215, %c32_i64 : i64
              %256 = arith.divui %227, %c16384_i64 : i64
              %257 = arith.addi %227, %255 : i64
              %258 = arith.subi %257, %c1_i64 : i64
              %259 = arith.divui %258, %c16384_i64 : i64
              %260 = arith.cmpi eq, %256, %259 : i64
              scf.if %260 {
                %266 = arith.trunci %256 : i64 to i32
                %267 = arith.extui %266 : i32 to i64
                %268 = arith.subi %256, %267 : i64
                %269 = arith.remsi %268, %c4294967296_i64 : i64
                %270 = arith.cmpi ne, %269, %c0_i64 : i64
                %271 = arith.cmpi slt, %269, %c0_i64 : i64
                %272 = arith.andi %270, %271 : i1
                %273 = arith.divsi %268, %c4294967296_i64 : i64
                %274 = arith.extui %272 : i1 to i64
                %275 = arith.subi %273, %274 : i64
                %276 = arith.trunci %275 : i64 to i32
                %277 = arith.addi %253, %266 : i32
                %278 = arith.addi %254, %276 : i32
                %279 = arith.cmpi ult, %277, %253 : i32
                %280 = arith.extui %279 : i1 to i32
                %281 = arith.addi %278, %280 : i32
                %282 = arith.extui %277 : i32 to i64
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
                %295 = arith.xori %281, %226 : i32
                %296 = arith.extui %295 : i32 to i64
                %297 = arith.muli %296, %c3449720151_i64 : i64
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
                %309 = arith.addi %216, %c-1640531527_i32 : i32
                %310 = arith.addi %226, %c-1150833019_i32 : i32
                %311 = arith.xori %308, %284 : i32
                %312 = arith.xori %311, %309 : i32
                %313 = arith.extui %312 : i32 to i64
                %314 = arith.muli %313, %c3528531795_i64 : i64
                %315 = arith.trunci %314 : i64 to i32
                %316 = arith.extui %315 : i32 to i64
                %317 = arith.subi %314, %316 : i64
                %318 = arith.remsi %317, %c4294967296_i64 : i64
                %319 = arith.cmpi ne, %318, %c0_i64 : i64
                %320 = arith.cmpi slt, %318, %c0_i64 : i64
                %321 = arith.andi %319, %320 : i1
                %322 = arith.divsi %317, %c4294967296_i64 : i64
                %323 = arith.extui %321 : i1 to i64
                %324 = arith.subi %322, %323 : i64
                %325 = arith.trunci %324 : i64 to i32
                %326 = arith.xori %294, %216 : i32
                %327 = arith.addi %216, %c1013904242_i32 : i32
                %328 = arith.xori %298, %327 : i32
                %329 = arith.addi %226, %c1993301258_i32 : i32
                %330 = arith.xori %325, %329 : i32
                ascsimt.vf(%228, %326, %310, %328, %330, %216, %226, %315, %alloc_9, %255 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: memref<8192xi32, 1>, %arg18: i64):
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
                  %c16384_i32 = arith.constant 16384 : i32
                  %331 = ascsimt.thread_idx {ascsimt.scope = 24 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                  %332 = arith.extsi %331 {cannir.vf_threads = 256 : i64} : i32 to i64
                  scf.for %arg19 = %332 to %arg18 step %c256_i64_11  : i64 {
                    %333 = arith.trunci %arg19 : i64 to i32
                    %334 = arith.addi %arg9, %333 : i32
                    %335 = arith.remui %334, %c16384_i32 : i32
                    %336 = arith.xori %335, %arg10 : i32
                    %337 = arith.muli %336, %c-766435501_i32 : i32
                    %338 = "ascsimt.bitcast"(%336) : (i32) -> ui32
                    %339 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                    %340 = ascsimt.math_binary %338, %339, "mulhi" : ui32, ui32 -> ui32
                    %341 = "ascsimt.bitcast"(%340) : (ui32) -> i32
                    %342 = arith.xori %341, %arg11 : i32
                    %343 = arith.muli %342, %c-845247145_i32 : i32
                    %344 = "ascsimt.bitcast"(%342) : (i32) -> ui32
                    %345 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                    %346 = ascsimt.math_binary %344, %345, "mulhi" : ui32, ui32 -> ui32
                    %347 = "ascsimt.bitcast"(%346) : (ui32) -> i32
                    %348 = arith.xori %347, %arg12 : i32
                    %349 = arith.xori %337, %arg13 : i32
                    %350:4 = scf.for %arg20 = %c3_i32 to %c10_i32 step %c1_i32_16 iter_args(%arg21 = %348, %arg22 = %343, %arg23 = %349, %arg24 = %arg16) -> (i32, i32, i32, i32)  : i32 {
                      %359 = arith.muli %arg21, %c-766435501_i32 : i32
                      %360 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                      %361 = ascsimt.math_binary %360, %339, "mulhi" : ui32, ui32 -> ui32
                      %362 = "ascsimt.bitcast"(%361) : (ui32) -> i32
                      %363 = arith.muli %arg23, %c-845247145_i32 : i32
                      %364 = "ascsimt.bitcast"(%arg23) : (i32) -> ui32
                      %365 = ascsimt.math_binary %364, %345, "mulhi" : ui32, ui32 -> ui32
                      %366 = "ascsimt.bitcast"(%365) : (ui32) -> i32
                      %367 = arith.muli %arg20, %c-1640531527_i32_18 : i32
                      %368 = arith.addi %arg14, %367 : i32
                      %369 = arith.muli %arg20, %c-1150833019_i32_17 : i32
                      %370 = arith.addi %arg15, %369 : i32
                      %371 = arith.xori %366, %arg22 : i32
                      %372 = arith.xori %371, %368 : i32
                      %373 = arith.xori %362, %arg24 : i32
                      %374 = arith.xori %373, %370 : i32
                      scf.yield %372, %363, %374, %359 : i32, i32, i32, i32
                    }
                    %351 = arith.muli %arg19, %c4_i64_15 : i64
                    %352 = arith.index_cast %351 : i64 to index
                    ascsimt.store %arg17[%352], %350#0 : memref<8192xi32, 1>, i32
                    %353 = arith.addi %351, %c1_i64_14 : i64
                    %354 = arith.index_cast %353 : i64 to index
                    ascsimt.store %arg17[%354], %350#1 : memref<8192xi32, 1>, i32
                    %355 = arith.addi %351, %c2_i64_13 : i64
                    %356 = arith.index_cast %355 : i64 to index
                    ascsimt.store %arg17[%356], %350#2 : memref<8192xi32, 1>, i32
                    %357 = arith.addi %351, %c3_i64_12 : i64
                    %358 = arith.index_cast %357 : i64 to index
                    ascsimt.store %arg17[%358], %350#3 : memref<8192xi32, 1>, i32
                  } {cannir.vf_threads = 256 : i64}
                } {threads = 256 : i64}
              } else {
                ascsimt.vf(%228, %238, %253, %254, %216, %226, %alloc_9, %255 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
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
                  %c16384_i32 = arith.constant 16384 : i32
                  %266 = ascsimt.thread_idx {ascsimt.scope = 25 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                  %267 = arith.extsi %266 {cannir.vf_threads = 256 : i64} : i32 to i64
                  scf.for %arg17 = %267 to %arg16 step %c256_i64_11  : i64 {
                    %268 = arith.trunci %arg17 : i64 to i32
                    %269 = arith.addi %arg9, %268 : i32
                    %270 = arith.cmpi ult, %269, %arg9 : i32
                    %271 = arith.extui %270 : i1 to i32
                    %272 = arith.addi %arg10, %271 : i32
                    %273 = arith.divui %269, %c16384_i32 : i32
                    %274 = arith.remui %272, %c16384_i32 : i32
                    %275 = arith.muli %274, %c262144_i32 : i32
                    %276 = arith.addi %273, %275 : i32
                    %277 = arith.divui %272, %c16384_i32 : i32
                    %278 = arith.addi %arg11, %276 : i32
                    %279 = arith.addi %arg12, %277 : i32
                    %280 = arith.cmpi ult, %278, %arg11 : i32
                    %281 = arith.extui %280 : i1 to i32
                    %282 = arith.addi %279, %281 : i32
                    %283 = arith.remui %269, %c16384_i32 : i32
                    %284:4 = scf.for %arg18 = %c0_i32_17 to %c10_i32 step %c1_i32_16 iter_args(%arg19 = %c0_i32_17, %arg20 = %283, %arg21 = %278, %arg22 = %282) -> (i32, i32, i32, i32)  : i32 {
                      %293 = arith.muli %arg19, %c-766435501_i32 : i32
                      %294 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                      %295 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %296 = ascsimt.math_binary %294, %295, "mulhi" : ui32, ui32 -> ui32
                      %297 = "ascsimt.bitcast"(%296) : (ui32) -> i32
                      %298 = arith.muli %arg21, %c-845247145_i32 : i32
                      %299 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                      %300 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %301 = ascsimt.math_binary %299, %300, "mulhi" : ui32, ui32 -> ui32
                      %302 = "ascsimt.bitcast"(%301) : (ui32) -> i32
                      %303 = arith.muli %arg18, %c-1640531527_i32_19 : i32
                      %304 = arith.addi %arg13, %303 : i32
                      %305 = arith.muli %arg18, %c-1150833019_i32_18 : i32
                      %306 = arith.addi %arg14, %305 : i32
                      %307 = arith.xori %302, %arg20 : i32
                      %308 = arith.xori %307, %304 : i32
                      %309 = arith.xori %297, %arg22 : i32
                      %310 = arith.xori %309, %306 : i32
                      scf.yield %308, %298, %310, %293 : i32, i32, i32, i32
                    }
                    %285 = arith.muli %arg17, %c4_i64_15 : i64
                    %286 = arith.index_cast %285 : i64 to index
                    ascsimt.store %arg15[%286], %284#0 : memref<8192xi32, 1>, i32
                    %287 = arith.addi %285, %c1_i64_14 : i64
                    %288 = arith.index_cast %287 : i64 to index
                    ascsimt.store %arg15[%288], %284#1 : memref<8192xi32, 1>, i32
                    %289 = arith.addi %285, %c2_i64_13 : i64
                    %290 = arith.index_cast %289 : i64 to index
                    ascsimt.store %arg15[%290], %284#2 : memref<8192xi32, 1>, i32
                    %291 = arith.addi %285, %c3_i64_12 : i64
                    %292 = arith.index_cast %291 : i64 to index
                    ascsimt.store %arg15[%292], %284#3 : memref<8192xi32, 1>, i32
                  } {cannir.vf_threads = 256 : i64}
                } {threads = 256 : i64}
              }
              %261 = arith.trunci %207 : i64 to i32
              %262 = arith.trunci %67 : i64 to i32
              %263 = arith.trunci %49 : i64 to i32
              %264 = arith.muli %261, %c2_i32 : i32
              %265 = arith.cmpi eq, %263, %c4_i32 : i32
              ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%67, %108, %alloc_9, %alloc_6, %alloc_8, %261, %262, %264, %265, %alloc_7 : i64, memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, memref<1024xui32, 1>, i32, i32, i32, i1, memref<2048xui8, 1>) {
              ^bb0(%arg9: i64, %arg10: memref<32768xbf16, 1>, %arg11: memref<8192xi32, 1>, %arg12: memref<32768xui8, 1>, %arg13: memref<1024xui32, 1>, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i1, %arg18: memref<2048xui8, 1>):
                %c2139095040_i32 = arith.constant 2139095040 : i32
                %c127_i32 = arith.constant 127 : i32
                %c123_i32 = arith.constant 123 : i32
                %c947912704_i32 = arith.constant 947912704 : i32
                %cst = arith.constant 4.2949673E+9 : f32
                %c29360128_i32 = arith.constant 29360128 : i32
                %c128_i32 = arith.constant 128 : i32
                %c65535_i32 = arith.constant 65535 : i32
                %c2_i32_11 = arith.constant 2 : i32
                %c7_i32 = arith.constant 7 : i32
                %c1_i32_12 = arith.constant 1 : i32
                %c4_i32_13 = arith.constant 4 : i32
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
                %266 = ascvec.create_mask : !ascvec.mask
                %267 = ascvec.arange %c0_i32_16 : i32 -> <ui32>
                %268 = ascvec.cmp_lt_scalar %267, %c32_i32, %266 : <ui32>, i32 -> !ascvec.mask
                %269 = ascvec.cmp_ge_scalar %267, %c32_i32, %266 : <ui32>, i32 -> !ascvec.mask
                scf.for %arg19 = %c0_i32_16 to %arg14 step %c1_i32_12  : i32 {
                  %271 = arith.muli %arg19, %c64_i32 : i32
                  %272 = arith.trunci %arg9 : i64 to i32
                  %273 = arith.subi %272, %271 : i32
                  %mask, %remaining_out = ascvec.update_mask %273 : !ascvec.mask
                  %274 = arith.index_cast %271 : i32 to index
                  %275 = ascvec.loadalign_unpack %arg10[%274] {mode = "b16_to_b32"} : memref<32768xbf16, 1>, index -> <bf16>
                  %276 = ascvec.cast %275, %266 {rounding = 6 : i32} : <bf16> -> <f32>
                  %277 = ascvec.duplicate_scalar %cst_14, %266 : f32 -> <f32>
                  %278 = ascvec.select %276, %277, %mask : <f32>
                  %279 = ascvec.reinterpret %278 : <f32> -> <ui32>
                  %280 = ascvec.duplicate_scalar %c2147483647_i32, %266 : i32 -> <ui32>
                  %281 = ascvec.and %279, %280, %266 : <ui32>, <ui32> -> <ui32>
                  %282 = ascvec.reduce_max %281, %268 : <ui32>
                  %283 = ascvec.duplicate %282, %266 : <ui32> -> <ui32>
                  %284 = ascvec.reduce_max %281, %269 : <ui32>
                  %285 = ascvec.duplicate %284, %266 : <ui32> -> <ui32>
                  %286 = ascvec.select %283, %285, %268 : <ui32>
                  %287 = ascvec.shr %286, 23, %266 : <ui32> -> <ui32>
                  %288 = ascvec.cmp_ge_scalar %287, %c255_i32, %266 : <ui32>, i32 -> !ascvec.mask
                  %289 = ascvec.max_scalar %287, %c15_i32, %266 : <ui32>, i32
                  %290 = ascvec.add_scalar %289, %c-15_i32, %266 : <ui32>, i32
                  %291 = ascvec.duplicate_scalar %c255_i32, %288 merge %290 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %292 = ascvec.duplicate_scalar %c254_i32, %266 : i32 -> <ui32>
                  %293 = ascvec.sub %292, %291, %266 : <ui32>
                  %294 = ascvec.cmp_ne_scalar %291, %c0_i32_16, %266 : <ui32>, i32 -> !ascvec.mask
                  %295 = ascvec.shl %293, 23, %294 : <ui32> -> <ui32>
                  %296 = ascvec.duplicate_scalar %c2139160576_i32, %288 merge %295 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %297 = ascvec.reinterpret %296 : <ui32> -> <f32>
                  %298 = ascvec.mul %278, %297, %266 : <f32>
                  %299 = ascvec.cast %298, %266 : <f32> -> <bf16>
                  %300 = ascvec.cast %299, %266 {rounding = 6 : i32} : <bf16> -> <f32>
                  %301 = arith.remsi %arg19, %c4_i32_13 : i32
                  %302 = arith.cmpi ne, %301, %c0_i32_16 : i32
                  %303 = arith.cmpi slt, %301, %c0_i32_16 : i32
                  %304 = arith.andi %302, %303 : i1
                  %305 = arith.select %304, %c4_i32_13, %c0_i32_16 : i32
                  %306 = arith.addi %301, %305 overflow<nsw> : i32
                  %307 = arith.muli %306, %c64_i32 overflow<nsw> : i32
                  %308 = ascvec.add_scalar %267, %307, %266 : <ui32>, i32
                  %309 = arith.divsi %arg19, %c4_i32_13 : i32
                  %310 = arith.extui %304 : i1 to i32
                  %311 = arith.subi %309, %310 : i32
                  %312 = ascvec.shr %308, 3, %266 : <ui32> -> <ui32>
                  %313 = arith.muli %311, %c32_i32 : i32
                  %314 = ascvec.add_scalar %312, %313, %266 : <ui32>, i32
                  %315 = ascvec.gather %arg11[%c0_15], %314, %266 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                  %316 = ascvec.reinterpret %315 : <i32> -> <ui32>
                  %317 = ascvec.duplicate_scalar %c7_i32, %266 : i32 -> <ui32>
                  %318 = ascvec.and %308, %317, %266 : <ui32>, <ui32> -> <ui32>
                  %319 = ascvec.shl %316, 11, %266 : <ui32> -> <ui32>
                  %320 = ascvec.shr %316, 21, %266 : <ui32> -> <ui32>
                  %321 = ascvec.or %319, %320, %266 : <ui32>, <ui32> -> <ui32>
                  %322 = ascvec.shr %321, 3, %266 : <ui32> -> <ui32>
                  %323 = ascvec.shl %321, 29, %266 : <ui32> -> <ui32>
                  %324 = ascvec.or %322, %323, %266 : <ui32>, <ui32> -> <ui32>
                  %325 = ascvec.duplicate_scalar %c1_i32_12, %266 : i32 -> <ui32>
                  %326 = ascvec.and %318, %325, %266 : <ui32>, <ui32> -> <ui32>
                  %327 = ascvec.cmp_ne_scalar %326, %c0_i32_16, %266 : <ui32>, i32 -> !ascvec.mask
                  %328 = ascvec.select %324, %321, %327 : <ui32>
                  %329 = ascvec.shr %328, 8, %266 : <ui32> -> <ui32>
                  %330 = ascvec.shl %328, 24, %266 : <ui32> -> <ui32>
                  %331 = ascvec.or %329, %330, %266 : <ui32>, <ui32> -> <ui32>
                  %332 = ascvec.duplicate_scalar %c2_i32_11, %266 : i32 -> <ui32>
                  %333 = ascvec.and %318, %332, %266 : <ui32>, <ui32> -> <ui32>
                  %334 = ascvec.cmp_ne_scalar %333, %c0_i32_16, %266 : <ui32>, i32 -> !ascvec.mask
                  %335 = ascvec.select %331, %328, %334 : <ui32>
                  %336 = ascvec.shr %335, 16, %266 : <ui32> -> <ui32>
                  %337 = ascvec.shl %335, 16, %266 : <ui32> -> <ui32>
                  %338 = ascvec.or %336, %337, %266 : <ui32>, <ui32> -> <ui32>
                  %339 = ascvec.duplicate_scalar %c4_i32_13, %266 : i32 -> <ui32>
                  %340 = ascvec.and %318, %339, %266 : <ui32>, <ui32> -> <ui32>
                  %341 = ascvec.cmp_ne_scalar %340, %c0_i32_16, %266 : <ui32>, i32 -> !ascvec.mask
                  %342 = ascvec.select %338, %335, %341 : <ui32>
                  %343 = ascvec.duplicate_scalar %c65535_i32, %266 : i32 -> <ui32>
                  %344 = ascvec.and %342, %343, %266 : <ui32>, <ui32> -> <ui32>
                  %345 = ascvec.reinterpret %300 : <f32> -> <ui32>
                  %346 = ascvec.shr %345, 24, %266 : <ui32> -> <ui32>
                  %347 = ascvec.duplicate_scalar %c128_i32, %266 : i32 -> <ui32>
                  %348 = ascvec.and %346, %347, %266 : <ui32>, <ui32> -> <ui32>
                  %349 = ascvec.and %345, %280, %266 : <ui32>, <ui32> -> <ui32>
                  %350 = ascvec.shr %349, 5, %266 : <ui32> -> <ui32>
                  %351 = ascvec.duplicate_scalar %c29360128_i32, %266 : i32 -> <ui32>
                  %352 = ascvec.sub %350, %351, %266 : <ui32>
                  %353 = ascvec.reinterpret %349 : <ui32> -> <f32>
                  %354 = ascvec.mul_scalar %353, %cst, %266 : <f32>, f32
                  %355 = ascvec.cast %354, %266 {rounding = 4 : i32} : <f32> -> <i32>
                  %356 = ascvec.reinterpret %355 : <i32> -> <ui32>
                  %357 = ascvec.cmp_ge_scalar %349, %c947912704_i32, %266 : <ui32>, i32 -> !ascvec.mask
                  %358 = ascvec.select %352, %356, %357 : <ui32>
                  %359 = ascvec.and %344, %343, %266 : <ui32>, <ui32> -> <ui32>
                  %360 = ascvec.add %358, %359, %266 : <ui32>
                  %361 = ascvec.shr %360, 16, %266 : <ui32> -> <ui32>
                  %362 = ascvec.min_scalar %361, %c123_i32, %266 : <ui32>, i32
                  %363 = ascvec.or %362, %348, %266 : <ui32>, <ui32> -> <ui32>
                  %364 = ascvec.duplicate_scalar %c127_i32, %266 : i32 -> <ui32>
                  %365 = ascvec.cmp_gt_scalar %349, %c2139095040_i32, %266 : <ui32>, i32 -> !ascvec.mask
                  %366 = ascvec.select %364, %363, %365 : <ui32>
                  ascvec.storealign_pack %arg12[%274], %366, %mask {mode = "b32_to_b8"} : memref<32768xui8, 1>, index, <ui32>
                  %367 = arith.muli %arg19, %c2_i32_11 : i32
                  %368 = arith.index_cast %367 : i32 to index
                  ascvec.storealign_1st %arg13[%368], %291 : memref<1024xui32, 1>, index, <ui32>
                  %369 = ascvec.duplicate_scalar %c32_i32, %266 : i32 -> <ui32>
                  %370 = ascvec.gather_reg %291, %369 : <ui32>, <ui32> -> <ui32>
                  %371 = arith.addi %367, %c1_i32_12 : i32
                  %372 = arith.index_cast %371 : i32 to index
                  ascvec.storealign_1st %arg13[%372], %370 : memref<1024xui32, 1>, index, <ui32>
                }
                ascvec.mem_bar vst_vld
                %270 = scf.if %arg17 -> (i32) {
                  %271 = arith.remsi %arg15, %c32_i32 : i32
                  %272 = arith.cmpi ne, %271, %c0_i32_16 : i32
                  %273 = arith.cmpi slt, %271, %c0_i32_16 : i32
                  %274 = arith.andi %272, %273 : i1
                  %275 = arith.divsi %arg15, %c32_i32 : i32
                  %276 = arith.extui %274 : i1 to i32
                  %277 = arith.subi %275, %276 : i32
                  %278 = arith.muli %277, %c2_i32_11 : i32
                  scf.yield %278 : i32
                } else {
                  scf.yield %arg16 : i32
                }
                scf.for %arg19 = %c0_i32_16 to %270 step %c64_i32  : i32 {
                  %271 = arith.subi %270, %arg19 : i32
                  %mask, %remaining_out = ascvec.update_mask %271 : !ascvec.mask
                  %272 = ascvec.add_scalar %267, %arg19, %266 : <ui32>, i32
                  scf.if %arg17 {
                    %273 = ascvec.shr %272, 1, %266 : <ui32> -> <ui32>
                    %274 = ascvec.gather %arg13[%c0_15], %273, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %275 = ascvec.duplicate_scalar %c1_i32_12, %266 : i32 -> <ui32>
                    %276 = ascvec.and %267, %275, %266 : <ui32>, <ui32> -> <ui32>
                    %277 = ascvec.cmp_eq_scalar %276, %c0_i32_16, %266 : <ui32>, i32 -> !ascvec.mask
                    %278 = ascvec.duplicate_scalar %c0_i32_16, %266 : i32 -> <ui32>
                    %279 = ascvec.select %274, %278, %277 : <ui32>
                    %280 = arith.index_cast %arg19 : i32 to index
                    %c0_17 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg20 = %c0_17 to %c1 step %c1 {
                      ascvec.storealign_pack %arg18[%280], %279, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  } else {
                    %273 = ascvec.gather %arg13[%c0_15], %272, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %274 = arith.index_cast %arg19 : i32 to index
                    %c0_17 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg20 = %c0_17 to %c1 step %c1 {
                      ascvec.storealign_pack %arg18[%274], %273, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  }
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
            } else {
              %200 = arith.addi %67, %c63_i64 : i64
              %201 = arith.remsi %200, %c64_i64 : i64
              %202 = arith.cmpi ne, %201, %c0_i64 : i64
              %203 = arith.cmpi slt, %201, %c0_i64 : i64
              %204 = arith.andi %202, %203 : i1
              %205 = arith.divsi %200, %c64_i64 : i64
              %206 = arith.extui %204 : i1 to i64
              %207 = arith.subi %205, %206 : i64
              %208 = arith.addi %67, %c1023_i64 : i64
              %209 = arith.remsi %208, %c1024_i64 : i64
              %210 = arith.cmpi ne, %209, %c0_i64 : i64
              %211 = arith.cmpi slt, %209, %c0_i64 : i64
              %212 = arith.andi %210, %211 : i1
              %213 = arith.divsi %208, %c1024_i64 : i64
              %214 = arith.extui %212 : i1 to i64
              %215 = arith.subi %213, %214 : i64
              %216 = arith.trunci %1 : i64 to i32
              %217 = arith.extui %216 : i32 to i64
              %218 = arith.subi %1, %217 : i64
              %219 = arith.remsi %218, %c4294967296_i64 : i64
              %220 = arith.cmpi ne, %219, %c0_i64 : i64
              %221 = arith.cmpi slt, %219, %c0_i64 : i64
              %222 = arith.andi %220, %221 : i1
              %223 = arith.divsi %218, %c4294967296_i64 : i64
              %224 = arith.extui %222 : i1 to i64
              %225 = arith.subi %223, %224 : i64
              %226 = arith.trunci %225 : i64 to i32
              %227 = arith.remsi %109, %c16_i64 : i64
              %228 = arith.cmpi ne, %227, %c0_i64 : i64
              %229 = arith.cmpi slt, %227, %c0_i64 : i64
              %230 = arith.andi %228, %229 : i1
              %231 = arith.divsi %109, %c16_i64 : i64
              %232 = arith.extui %230 : i1 to i64
              %233 = arith.subi %231, %232 : i64
              %234 = arith.remsi %arg3, %c16_i64 : i64
              %235 = arith.cmpi ne, %234, %c0_i64 : i64
              %236 = arith.cmpi slt, %234, %c0_i64 : i64
              %237 = arith.andi %235, %236 : i1
              %238 = arith.divsi %arg3, %c16_i64 : i64
              %239 = arith.extui %237 : i1 to i64
              %240 = arith.subi %238, %239 : i64
              %241 = arith.remsi %233, %240 : i64
              %242 = arith.cmpi ne, %241, %c0_i64 : i64
              %243 = arith.cmpi slt, %241, %c0_i64 : i64
              %244 = arith.cmpi slt, %240, %c0_i64 : i64
              %245 = arith.xori %243, %244 : i1
              %246 = arith.andi %242, %245 : i1
              %247 = arith.divsi %233, %240 : i64
              %248 = arith.extui %246 : i1 to i64
              %249 = arith.subi %247, %248 : i64
              %250 = arith.select %246, %240, %c0_i64 : i64
              %251 = arith.addi %241, %250 : i64
              %252 = arith.trunci %249 : i64 to i32
              %253 = arith.extui %252 : i32 to i64
              %254 = arith.subi %249, %253 : i64
              %255 = arith.remsi %254, %c4294967296_i64 : i64
              %256 = arith.cmpi ne, %255, %c0_i64 : i64
              %257 = arith.cmpi slt, %255, %c0_i64 : i64
              %258 = arith.andi %256, %257 : i1
              %259 = arith.divsi %254, %c4294967296_i64 : i64
              %260 = arith.extui %258 : i1 to i64
              %261 = arith.subi %259, %260 : i64
              %262 = arith.trunci %261 : i64 to i32
              %263 = arith.trunci %251 : i64 to i32
              %264 = arith.extui %263 : i32 to i64
              %265 = arith.subi %251, %264 : i64
              %266 = arith.remsi %265, %c4294967296_i64 : i64
              %267 = arith.cmpi ne, %266, %c0_i64 : i64
              %268 = arith.cmpi slt, %266, %c0_i64 : i64
              %269 = arith.andi %267, %268 : i1
              %270 = arith.divsi %265, %c4294967296_i64 : i64
              %271 = arith.extui %269 : i1 to i64
              %272 = arith.subi %270, %271 : i64
              %273 = arith.trunci %272 : i64 to i32
              %274 = arith.remsi %c64_i64, %240 : i64
              %275 = arith.cmpi ne, %274, %c0_i64 : i64
              %276 = arith.cmpi slt, %274, %c0_i64 : i64
              %277 = arith.xori %276, %244 : i1
              %278 = arith.andi %275, %277 : i1
              %279 = arith.divsi %c64_i64, %240 : i64
              %280 = arith.extui %278 : i1 to i64
              %281 = arith.subi %279, %280 : i64
              %282 = arith.select %278, %240, %c0_i64 : i64
              %283 = arith.addi %274, %282 : i64
              %284 = arith.trunci %215 : i64 to i32
              ascvec.vf(%240, %4, %216, %226, %alloc_9, %283, %281, %284, %252, %262, %263, %273 : i64, i64, i32, i32, memref<8192xi32, 1>, i64, i64, i32, i32, i32, i32, i32) {
              ^bb0(%arg9: i64, %arg10: i64, %arg11: i32, %arg12: i32, %arg13: memref<8192xi32, 1>, %arg14: i64, %arg15: i64, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32):
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
                %290:4 = scf.for %arg21 = %c0_i32_19 to %arg16 step %c1_i32_11 iter_args(%arg22 = %arg17, %arg23 = %arg18, %arg24 = %arg19, %arg25 = %arg20) -> (i32, i32, i32, i32)  : i32 {
                  %291 = arith.extui %arg22 : i32 to i64
                  %292 = arith.extui %arg23 : i32 to i64
                  %293 = arith.muli %292, %c4294967296_i64_20 : i64
                  %294 = arith.addi %291, %293 : i64
                  %295 = arith.extui %arg24 : i32 to i64
                  %296 = arith.extui %arg25 : i32 to i64
                  %297 = arith.muli %296, %c4294967296_i64_20 : i64
                  %298 = arith.addi %295, %297 : i64
                  %299 = arith.muli %arg21, %c256_i32 : i32
                  %300 = ascvec.create_mask : !ascvec.mask
                  %301 = ascvec.arange %c0_i32_19 : i32 -> <ui32>
                  %302 = arith.muli %294, %arg9 : i64
                  %303 = arith.addi %302, %298 : i64
                  %304 = arith.trunci %303 : i64 to i32
                  %305 = arith.extui %304 : i32 to i64
                  %306 = arith.subi %303, %305 : i64
                  %307 = arith.remsi %306, %c4294967296_i64_20 : i64
                  %308 = arith.cmpi ne, %307, %c0_i64_18 : i64
                  %309 = arith.cmpi slt, %307, %c0_i64_18 : i64
                  %310 = arith.andi %308, %309 : i1
                  %311 = arith.divsi %306, %c4294967296_i64_20 : i64
                  %312 = arith.extui %310 : i1 to i64
                  %313 = arith.subi %311, %312 : i64
                  %314 = arith.trunci %313 : i64 to i32
                  %315 = ascvec.duplicate_scalar %304, %300 : i32 -> <ui32>
                  %316 = ascvec.duplicate_scalar %314, %300 : i32 -> <ui32>
                  %317 = ascvec.duplicate_scalar %c0_i32_19, %300 : i32 -> <ui32>
                  %carry, %dst = ascvec.addc %315, %301, %300 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %carry_21, %dst_22 = ascvec.addc %316, %317, %carry, %300 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %318 = arith.trunci %arg10 : i64 to i32
                  %319 = arith.extui %318 : i32 to i64
                  %320 = arith.subi %arg10, %319 : i64
                  %321 = arith.remsi %320, %c4294967296_i64_20 : i64
                  %322 = arith.cmpi ne, %321, %c0_i64_18 : i64
                  %323 = arith.cmpi slt, %321, %c0_i64_18 : i64
                  %324 = arith.andi %322, %323 : i1
                  %325 = arith.divsi %320, %c4294967296_i64_20 : i64
                  %326 = arith.extui %324 : i1 to i64
                  %327 = arith.subi %325, %326 : i64
                  %328 = arith.trunci %327 : i64 to i32
                  %329 = arith.divui %318, %c4_i32_17 : i32
                  %330 = arith.remui %328, %c4_i32_17 : i32
                  %331 = arith.muli %330, %c1073741824_i32_16 : i32
                  %332 = arith.addi %329, %331 : i32
                  %333 = arith.divui %328, %c4_i32_17 : i32
                  %334 = ascvec.shr %dst, 14, %300 : <ui32> -> <ui32>
                  %335 = ascvec.shl %dst_22, 18, %300 : <ui32> -> <ui32>
                  %336 = ascvec.or %334, %335, %300 : <ui32>, <ui32> -> <ui32>
                  %337 = ascvec.shr %dst_22, 14, %300 : <ui32> -> <ui32>
                  %338 = ascvec.duplicate_scalar %332, %300 : i32 -> <ui32>
                  %339 = ascvec.duplicate_scalar %333, %300 : i32 -> <ui32>
                  %carry_23, %dst_24 = ascvec.addc %338, %336, %300 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %carry_25, %dst_26 = ascvec.addc %339, %337, %carry_23, %300 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %340 = ascvec.duplicate_scalar %c16383_i32, %300 : i32 -> <ui32>
                  %341 = ascvec.and %dst, %340, %300 : <ui32>, <ui32> -> <ui32>
                  %342 = ascvec.duplicate_scalar %c-766435501_i32, %300 : i32 -> <ui32>
                  %343 = ascvec.duplicate_scalar %c-845247145_i32, %300 : i32 -> <ui32>
                  %dst0, %dst1 = ascvec.mull %317, %342, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_27, %dst1_28 = ascvec.mull %dst_24, %343, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %344 = ascvec.xor %dst1_28, %341, %300 : <ui32>, <ui32> -> <ui32>
                  %345 = ascvec.xor %dst1, %dst_26, %300 : <ui32>, <ui32> -> <ui32>
                  %346 = ascvec.duplicate_scalar %arg11, %300 : i32 -> <ui32>
                  %347 = ascvec.duplicate_scalar %arg12, %300 : i32 -> <ui32>
                  %348 = ascvec.xor %344, %346, %300 : <ui32>, <ui32> -> <ui32>
                  %349 = ascvec.xor %345, %347, %300 : <ui32>, <ui32> -> <ui32>
                  %dst0_29, %dst1_30 = ascvec.mull %348, %342, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_31, %dst1_32 = ascvec.mull %349, %343, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %350 = ascvec.xor %dst1_32, %dst0_27, %300 : <ui32>, <ui32> -> <ui32>
                  %351 = ascvec.xor %dst1_30, %dst0, %300 : <ui32>, <ui32> -> <ui32>
                  %352 = arith.addi %arg11, %c-1640531527_i32_15 : i32
                  %353 = ascvec.duplicate_scalar %352, %300 : i32 -> <ui32>
                  %354 = arith.addi %arg12, %c-1150833019_i32_14 : i32
                  %355 = ascvec.duplicate_scalar %354, %300 : i32 -> <ui32>
                  %356 = ascvec.xor %350, %353, %300 : <ui32>, <ui32> -> <ui32>
                  %357 = ascvec.xor %351, %355, %300 : <ui32>, <ui32> -> <ui32>
                  %dst0_33, %dst1_34 = ascvec.mull %356, %342, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_35, %dst1_36 = ascvec.mull %357, %343, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %358 = ascvec.xor %dst1_36, %dst0_31, %300 : <ui32>, <ui32> -> <ui32>
                  %359 = ascvec.xor %dst1_34, %dst0_29, %300 : <ui32>, <ui32> -> <ui32>
                  %360 = arith.addi %arg11, %c1013904242_i32_13 : i32
                  %361 = ascvec.duplicate_scalar %360, %300 : i32 -> <ui32>
                  %362 = arith.addi %arg12, %c1993301258_i32_12 : i32
                  %363 = ascvec.duplicate_scalar %362, %300 : i32 -> <ui32>
                  %364 = ascvec.xor %358, %361, %300 : <ui32>, <ui32> -> <ui32>
                  %365 = ascvec.xor %359, %363, %300 : <ui32>, <ui32> -> <ui32>
                  %dst0_37, %dst1_38 = ascvec.mull %364, %342, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_39, %dst1_40 = ascvec.mull %365, %343, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %366 = ascvec.xor %dst1_40, %dst0_35, %300 : <ui32>, <ui32> -> <ui32>
                  %367 = ascvec.xor %dst1_38, %dst0_33, %300 : <ui32>, <ui32> -> <ui32>
                  %368 = arith.addi %arg11, %c-626627285_i32 : i32
                  %369 = ascvec.duplicate_scalar %368, %300 : i32 -> <ui32>
                  %370 = arith.addi %arg12, %c842468239_i32 : i32
                  %371 = ascvec.duplicate_scalar %370, %300 : i32 -> <ui32>
                  %372 = ascvec.xor %366, %369, %300 : <ui32>, <ui32> -> <ui32>
                  %373 = ascvec.xor %367, %371, %300 : <ui32>, <ui32> -> <ui32>
                  %dst0_41, %dst1_42 = ascvec.mull %372, %342, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_43, %dst1_44 = ascvec.mull %373, %343, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %374 = ascvec.xor %dst1_44, %dst0_39, %300 : <ui32>, <ui32> -> <ui32>
                  %375 = ascvec.xor %dst1_42, %dst0_37, %300 : <ui32>, <ui32> -> <ui32>
                  %376 = arith.addi %arg11, %c2027808484_i32 : i32
                  %377 = ascvec.duplicate_scalar %376, %300 : i32 -> <ui32>
                  %378 = arith.addi %arg12, %c-308364780_i32 : i32
                  %379 = ascvec.duplicate_scalar %378, %300 : i32 -> <ui32>
                  %380 = ascvec.xor %374, %377, %300 : <ui32>, <ui32> -> <ui32>
                  %381 = ascvec.xor %375, %379, %300 : <ui32>, <ui32> -> <ui32>
                  %dst0_45, %dst1_46 = ascvec.mull %380, %342, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_47, %dst1_48 = ascvec.mull %381, %343, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %382 = ascvec.xor %dst1_48, %dst0_43, %300 : <ui32>, <ui32> -> <ui32>
                  %383 = ascvec.xor %dst1_46, %dst0_41, %300 : <ui32>, <ui32> -> <ui32>
                  %384 = arith.addi %arg11, %c387276957_i32 : i32
                  %385 = ascvec.duplicate_scalar %384, %300 : i32 -> <ui32>
                  %386 = arith.addi %arg12, %c-1459197799_i32 : i32
                  %387 = ascvec.duplicate_scalar %386, %300 : i32 -> <ui32>
                  %388 = ascvec.xor %382, %385, %300 : <ui32>, <ui32> -> <ui32>
                  %389 = ascvec.xor %383, %387, %300 : <ui32>, <ui32> -> <ui32>
                  %dst0_49, %dst1_50 = ascvec.mull %388, %342, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_51, %dst1_52 = ascvec.mull %389, %343, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %390 = ascvec.xor %dst1_52, %dst0_47, %300 : <ui32>, <ui32> -> <ui32>
                  %391 = ascvec.xor %dst1_50, %dst0_45, %300 : <ui32>, <ui32> -> <ui32>
                  %392 = arith.addi %arg11, %c-1253254570_i32 : i32
                  %393 = ascvec.duplicate_scalar %392, %300 : i32 -> <ui32>
                  %394 = arith.addi %arg12, %c1684936478_i32 : i32
                  %395 = ascvec.duplicate_scalar %394, %300 : i32 -> <ui32>
                  %396 = ascvec.xor %390, %393, %300 : <ui32>, <ui32> -> <ui32>
                  %397 = ascvec.xor %391, %395, %300 : <ui32>, <ui32> -> <ui32>
                  %dst0_53, %dst1_54 = ascvec.mull %396, %342, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_55, %dst1_56 = ascvec.mull %397, %343, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %398 = ascvec.xor %dst1_56, %dst0_51, %300 : <ui32>, <ui32> -> <ui32>
                  %399 = ascvec.xor %dst1_54, %dst0_49, %300 : <ui32>, <ui32> -> <ui32>
                  %400 = arith.addi %arg11, %c1401181199_i32 : i32
                  %401 = ascvec.duplicate_scalar %400, %300 : i32 -> <ui32>
                  %402 = arith.addi %arg12, %c534103459_i32 : i32
                  %403 = ascvec.duplicate_scalar %402, %300 : i32 -> <ui32>
                  %404 = ascvec.xor %398, %401, %300 : <ui32>, <ui32> -> <ui32>
                  %405 = ascvec.xor %399, %403, %300 : <ui32>, <ui32> -> <ui32>
                  %dst0_57, %dst1_58 = ascvec.mull %404, %342, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_59, %dst1_60 = ascvec.mull %405, %343, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %406 = ascvec.xor %dst1_60, %dst0_55, %300 : <ui32>, <ui32> -> <ui32>
                  %407 = ascvec.xor %dst1_58, %dst0_53, %300 : <ui32>, <ui32> -> <ui32>
                  %408 = arith.addi %arg11, %c-239350328_i32 : i32
                  %409 = ascvec.duplicate_scalar %408, %300 : i32 -> <ui32>
                  %410 = arith.addi %arg12, %c-616729560_i32 : i32
                  %411 = ascvec.duplicate_scalar %410, %300 : i32 -> <ui32>
                  %412 = ascvec.xor %406, %409, %300 : <ui32>, <ui32> -> <ui32>
                  %413 = ascvec.xor %407, %411, %300 : <ui32>, <ui32> -> <ui32>
                  %dst0_61, %dst1_62 = ascvec.mull %412, %342, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_63, %dst1_64 = ascvec.mull %413, %343, %300 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %414 = ascvec.xor %dst1_64, %dst0_59, %300 : <ui32>, <ui32> -> <ui32>
                  %415 = ascvec.xor %dst1_62, %dst0_57, %300 : <ui32>, <ui32> -> <ui32>
                  %416 = arith.addi %arg11, %c-1879881855_i32 : i32
                  %417 = ascvec.duplicate_scalar %416, %300 : i32 -> <ui32>
                  %418 = arith.addi %arg12, %c-1767562579_i32 : i32
                  %419 = ascvec.duplicate_scalar %418, %300 : i32 -> <ui32>
                  %420 = ascvec.xor %414, %417, %300 : <ui32>, <ui32> -> <ui32>
                  %421 = ascvec.xor %415, %419, %300 : <ui32>, <ui32> -> <ui32>
                  %422 = ascvec.reinterpret %420 : <ui32> -> <i32>
                  %423 = arith.index_cast %299 : i32 to index
                  ascvec.storealign %arg13[%423], %422, %300 : memref<8192xi32, 1>, index, <i32>
                  %424 = arith.addi %299, %c64_i32 : i32
                  %425 = ascvec.reinterpret %dst0_63 : <ui32> -> <i32>
                  %426 = arith.index_cast %424 : i32 to index
                  ascvec.storealign %arg13[%426], %425, %300 : memref<8192xi32, 1>, index, <i32>
                  %427 = arith.addi %299, %c128_i32 : i32
                  %428 = ascvec.reinterpret %421 : <ui32> -> <i32>
                  %429 = arith.index_cast %427 : i32 to index
                  ascvec.storealign %arg13[%429], %428, %300 : memref<8192xi32, 1>, index, <i32>
                  %430 = arith.addi %299, %c192_i32 : i32
                  %431 = ascvec.reinterpret %dst0_61 : <ui32> -> <i32>
                  %432 = arith.index_cast %430 : i32 to index
                  ascvec.storealign %arg13[%432], %431, %300 : memref<8192xi32, 1>, index, <i32>
                  %433 = arith.addi %298, %arg14 : i64
                  %434 = arith.cmpi sge, %433, %arg9 : i64
                  %435 = arith.extui %434 : i1 to i64
                  %436 = arith.addi %294, %arg15 : i64
                  %437 = arith.addi %436, %435 : i64
                  %438 = arith.muli %435, %arg9 : i64
                  %439 = arith.subi %433, %438 : i64
                  %440 = arith.trunci %437 : i64 to i32
                  %441 = arith.extui %440 : i32 to i64
                  %442 = arith.subi %437, %441 : i64
                  %443 = arith.remsi %442, %c4294967296_i64_20 : i64
                  %444 = arith.cmpi ne, %443, %c0_i64_18 : i64
                  %445 = arith.cmpi slt, %443, %c0_i64_18 : i64
                  %446 = arith.andi %444, %445 : i1
                  %447 = arith.divsi %442, %c4294967296_i64_20 : i64
                  %448 = arith.extui %446 : i1 to i64
                  %449 = arith.subi %447, %448 : i64
                  %450 = arith.trunci %449 : i64 to i32
                  %451 = arith.trunci %439 : i64 to i32
                  %452 = arith.extui %451 : i32 to i64
                  %453 = arith.subi %439, %452 : i64
                  %454 = arith.remsi %453, %c4294967296_i64_20 : i64
                  %455 = arith.cmpi ne, %454, %c0_i64_18 : i64
                  %456 = arith.cmpi slt, %454, %c0_i64_18 : i64
                  %457 = arith.andi %455, %456 : i1
                  %458 = arith.divsi %453, %c4294967296_i64_20 : i64
                  %459 = arith.extui %457 : i1 to i64
                  %460 = arith.subi %458, %459 : i64
                  %461 = arith.trunci %460 : i64 to i32
                  scf.yield %440, %450, %451, %461 : i32, i32, i32, i32
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              %285 = arith.trunci %207 : i64 to i32
              %286 = arith.trunci %67 : i64 to i32
              %287 = arith.trunci %49 : i64 to i32
              %288 = arith.muli %285, %c2_i32 : i32
              %289 = arith.cmpi eq, %287, %c4_i32 : i32
              ascvec.get_buf v, %73, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%67, %108, %alloc_9, %alloc_6, %alloc_8, %285, %286, %288, %289, %alloc_7 : i64, memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, memref<1024xui32, 1>, i32, i32, i32, i1, memref<2048xui8, 1>) {
              ^bb0(%arg9: i64, %arg10: memref<32768xbf16, 1>, %arg11: memref<8192xi32, 1>, %arg12: memref<32768xui8, 1>, %arg13: memref<1024xui32, 1>, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i1, %arg18: memref<2048xui8, 1>):
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
                %290 = ascvec.create_mask : !ascvec.mask
                %291 = ascvec.arange %c0_i32_16 : i32 -> <ui32>
                %292 = ascvec.cmp_lt_scalar %291, %c32_i32, %290 : <ui32>, i32 -> !ascvec.mask
                %293 = ascvec.cmp_ge_scalar %291, %c32_i32, %290 : <ui32>, i32 -> !ascvec.mask
                scf.for %arg19 = %c0_i32_16 to %arg14 step %c1_i32_13  : i32 {
                  %295 = arith.muli %arg19, %c64_i32 : i32
                  %296 = arith.trunci %arg9 : i64 to i32
                  %297 = arith.subi %296, %295 : i32
                  %mask, %remaining_out = ascvec.update_mask %297 : !ascvec.mask
                  %298 = arith.index_cast %295 : i32 to index
                  %299 = ascvec.loadalign_unpack %arg10[%298] {mode = "b16_to_b32"} : memref<32768xbf16, 1>, index -> <bf16>
                  %300 = ascvec.cast %299, %290 {rounding = 6 : i32} : <bf16> -> <f32>
                  %301 = ascvec.duplicate_scalar %cst_14, %290 : f32 -> <f32>
                  %302 = ascvec.select %300, %301, %mask : <f32>
                  %303 = ascvec.reinterpret %302 : <f32> -> <ui32>
                  %304 = ascvec.duplicate_scalar %c2147483647_i32, %290 : i32 -> <ui32>
                  %305 = ascvec.and %303, %304, %290 : <ui32>, <ui32> -> <ui32>
                  %306 = ascvec.reduce_max %305, %292 : <ui32>
                  %307 = ascvec.duplicate %306, %290 : <ui32> -> <ui32>
                  %308 = ascvec.reduce_max %305, %293 : <ui32>
                  %309 = ascvec.duplicate %308, %290 : <ui32> -> <ui32>
                  %310 = ascvec.select %307, %309, %292 : <ui32>
                  %311 = ascvec.shr %310, 23, %290 : <ui32> -> <ui32>
                  %312 = ascvec.cmp_ge_scalar %311, %c255_i32, %290 : <ui32>, i32 -> !ascvec.mask
                  %313 = ascvec.max_scalar %311, %c15_i32, %290 : <ui32>, i32
                  %314 = ascvec.add_scalar %313, %c-15_i32, %290 : <ui32>, i32
                  %315 = ascvec.duplicate_scalar %c255_i32, %312 merge %314 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %316 = ascvec.duplicate_scalar %c254_i32, %290 : i32 -> <ui32>
                  %317 = ascvec.sub %316, %315, %290 : <ui32>
                  %318 = ascvec.cmp_ne_scalar %315, %c0_i32_16, %290 : <ui32>, i32 -> !ascvec.mask
                  %319 = ascvec.shl %317, 23, %318 : <ui32> -> <ui32>
                  %320 = ascvec.duplicate_scalar %c2139160576_i32, %312 merge %319 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %321 = ascvec.reinterpret %320 : <ui32> -> <f32>
                  %322 = ascvec.mul %302, %321, %290 : <f32>
                  %323 = ascvec.shr %291, 2, %290 : <ui32> -> <ui32>
                  %324 = ascvec.duplicate_scalar %c3_i32, %290 : i32 -> <ui32>
                  %325 = ascvec.and %323, %324, %290 : <ui32>, <ui32> -> <ui32>
                  %326 = ascvec.shl %325, 6, %290 : <ui32> -> <ui32>
                  %327 = ascvec.shr %291, 4, %290 : <ui32> -> <ui32>
                  %328 = ascvec.add %326, %327, %290 : <ui32>
                  %329 = arith.remsi %arg19, %c16_i32 : i32
                  %330 = arith.cmpi ne, %329, %c0_i32_16 : i32
                  %331 = arith.cmpi slt, %329, %c0_i32_16 : i32
                  %332 = arith.andi %330, %331 : i1
                  %333 = arith.divsi %arg19, %c16_i32 : i32
                  %334 = arith.extui %332 : i1 to i32
                  %335 = arith.subi %333, %334 : i32
                  %336 = arith.muli %335, %c256_i32 : i32
                  %337 = arith.select %332, %c16_i32, %c0_i32_16 : i32
                  %338 = arith.addi %329, %337 overflow<nsw> : i32
                  %339 = arith.muli %338, %c4_i32_12 overflow<nsw> : i32
                  %340 = arith.addi %336, %339 : i32
                  %341 = ascvec.add_scalar %328, %340, %290 : <ui32>, i32
                  %342 = ascvec.gather %arg11[%c0_15], %341, %290 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                  %343 = ascvec.reinterpret %342 : <i32> -> <ui32>
                  %344 = ascvec.shl %343, 8, %290 : <ui32> -> <ui32>
                  %345 = ascvec.shr %343, 24, %290 : <ui32> -> <ui32>
                  %346 = ascvec.or %344, %345, %290 : <ui32>, <ui32> -> <ui32>
                  %347 = ascvec.and %291, %324, %290 : <ui32>, <ui32> -> <ui32>
                  %348 = ascvec.duplicate_scalar %c1_i32_13, %290 : i32 -> <ui32>
                  %349 = ascvec.and %347, %348, %290 : <ui32>, <ui32> -> <ui32>
                  %350 = ascvec.cmp_eq_scalar %349, %c0_i32_16, %290 : <ui32>, i32 -> !ascvec.mask
                  %351 = ascvec.select %343, %346, %350 : <ui32>
                  %352 = ascvec.cmp_eq_scalar %347, %c0_i32_16, %290 : <ui32>, i32 -> !ascvec.mask
                  %353 = ascvec.cmp_eq_scalar %347, %c3_i32, %290 : <ui32>, i32 -> !ascvec.mask
                  %354 = ascvec.mask_or %352, %353, %290
                  %355 = ascvec.shr %351, 16, %290 : <ui32> -> <ui32>
                  %356 = ascvec.duplicate_scalar %c65535_i32, %290 : i32 -> <ui32>
                  %357 = ascvec.and %351, %356, %290 : <ui32>, <ui32> -> <ui32>
                  %358 = ascvec.select %355, %357, %354 : <ui32>
                  %359 = ascvec.reinterpret %322 : <f32> -> <ui32>
                  %360 = ascvec.shr %359, 24, %290 : <ui32> -> <ui32>
                  %361 = ascvec.duplicate_scalar %c128_i32, %290 : i32 -> <ui32>
                  %362 = ascvec.and %360, %361, %290 : <ui32>, <ui32> -> <ui32>
                  %363 = ascvec.and %359, %304, %290 : <ui32>, <ui32> -> <ui32>
                  %364 = ascvec.shr %363, 5, %290 : <ui32> -> <ui32>
                  %365 = ascvec.duplicate_scalar %c29360128_i32, %290 : i32 -> <ui32>
                  %366 = ascvec.sub %364, %365, %290 : <ui32>
                  %367 = ascvec.reinterpret %363 : <ui32> -> <f32>
                  %368 = ascvec.mul_scalar %367, %cst, %290 : <f32>, f32
                  %369 = ascvec.cast %368, %290 {rounding = 4 : i32} : <f32> -> <i32>
                  %370 = ascvec.reinterpret %369 : <i32> -> <ui32>
                  %371 = ascvec.cmp_ge_scalar %363, %c947912704_i32, %290 : <ui32>, i32 -> !ascvec.mask
                  %372 = ascvec.select %366, %370, %371 : <ui32>
                  %373 = ascvec.and %358, %356, %290 : <ui32>, <ui32> -> <ui32>
                  %374 = ascvec.add %372, %373, %290 : <ui32>
                  %375 = ascvec.shr %374, 16, %290 : <ui32> -> <ui32>
                  %376 = ascvec.min_scalar %375, %c123_i32, %290 : <ui32>, i32
                  %377 = ascvec.or %376, %362, %290 : <ui32>, <ui32> -> <ui32>
                  %378 = ascvec.duplicate_scalar %c127_i32, %290 : i32 -> <ui32>
                  %379 = ascvec.cmp_gt_scalar %363, %c2139095040_i32, %290 : <ui32>, i32 -> !ascvec.mask
                  %380 = ascvec.select %378, %377, %379 : <ui32>
                  ascvec.storealign_pack %arg12[%298], %380, %mask {mode = "b32_to_b8"} : memref<32768xui8, 1>, index, <ui32>
                  %381 = arith.muli %arg19, %c2_i32_11 : i32
                  %382 = arith.index_cast %381 : i32 to index
                  ascvec.storealign_1st %arg13[%382], %315 : memref<1024xui32, 1>, index, <ui32>
                  %383 = ascvec.duplicate_scalar %c32_i32, %290 : i32 -> <ui32>
                  %384 = ascvec.gather_reg %315, %383 : <ui32>, <ui32> -> <ui32>
                  %385 = arith.addi %381, %c1_i32_13 : i32
                  %386 = arith.index_cast %385 : i32 to index
                  ascvec.storealign_1st %arg13[%386], %384 : memref<1024xui32, 1>, index, <ui32>
                }
                ascvec.mem_bar vst_vld
                %294 = scf.if %arg17 -> (i32) {
                  %295 = arith.remsi %arg15, %c32_i32 : i32
                  %296 = arith.cmpi ne, %295, %c0_i32_16 : i32
                  %297 = arith.cmpi slt, %295, %c0_i32_16 : i32
                  %298 = arith.andi %296, %297 : i1
                  %299 = arith.divsi %arg15, %c32_i32 : i32
                  %300 = arith.extui %298 : i1 to i32
                  %301 = arith.subi %299, %300 : i32
                  %302 = arith.muli %301, %c2_i32_11 : i32
                  scf.yield %302 : i32
                } else {
                  scf.yield %arg16 : i32
                }
                scf.for %arg19 = %c0_i32_16 to %294 step %c64_i32  : i32 {
                  %295 = arith.subi %294, %arg19 : i32
                  %mask, %remaining_out = ascvec.update_mask %295 : !ascvec.mask
                  %296 = ascvec.add_scalar %291, %arg19, %290 : <ui32>, i32
                  scf.if %arg17 {
                    %297 = ascvec.shr %296, 1, %290 : <ui32> -> <ui32>
                    %298 = ascvec.gather %arg13[%c0_15], %297, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %299 = ascvec.duplicate_scalar %c1_i32_13, %290 : i32 -> <ui32>
                    %300 = ascvec.and %291, %299, %290 : <ui32>, <ui32> -> <ui32>
                    %301 = ascvec.cmp_eq_scalar %300, %c0_i32_16, %290 : <ui32>, i32 -> !ascvec.mask
                    %302 = ascvec.duplicate_scalar %c0_i32_16, %290 : i32 -> <ui32>
                    %303 = ascvec.select %298, %302, %301 : <ui32>
                    %304 = arith.index_cast %arg19 : i32 to index
                    %c0_17 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg20 = %c0_17 to %c1 step %c1 {
                      ascvec.storealign_pack %arg18[%304], %303, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  } else {
                    %297 = ascvec.gather %arg13[%c0_15], %296, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %298 = arith.index_cast %arg19 : i32 to index
                    %c0_17 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg20 = %c0_17 to %c1 step %c1 {
                      ascvec.storealign_pack %arg18[%298], %297, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  }
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %73, 0 {cannir.buffer.generated_sync}
            }
          }
          %114 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %115 = cannir.extract_shape %114 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
          %116 = cannir.extract_dim %115[0] : !cannir.shape<(-1, -1)>
          %117 = arith.minsi %116, %c0_i64 : i64
          %118 = arith.subi %116, %117 : i64
          %119 = arith.minsi %118, %c1_i64 : i64
          %120 = arith.cmpi sge, %116, %c0_i64 : i64
          %121 = arith.select %120, %119, %c0_i64 : i64
          %122 = arith.cmpi sgt, %121, %c1_i64 : i64
          %123 = arith.select %122, %c1_i64, %121 : i64
          %124 = cannir.extract_dim %115[1] : !cannir.shape<(-1, -1)>
          %125 = arith.divsi %124, %85 : i64
          %126 = arith.minsi %36, %125 : i64
          %127 = arith.muli %126, %35 : i64
          %128 = arith.subi %124, %127 : i64
          %129 = arith.minsi %35, %128 : i64
          %130 = arith.cmpi sle, %36, %125 : i64
          %131 = arith.select %130, %129, %c0_i64 : i64
          %132 = arith.divsi %131, %c32768_i64 : i64
          %133 = arith.minsi %arg7, %132 : i64
          %134 = arith.muli %133, %c32768_i64 : i64
          %135 = arith.subi %131, %134 : i64
          %136 = arith.minsi %135, %c32768_i64 : i64
          %137 = arith.cmpi sle, %arg7, %132 : i64
          %138 = arith.select %137, %136, %c0_i64 : i64
          %139 = arith.cmpi sgt, %138, %c32768_i64 : i64
          %140 = arith.select %139, %c32768_i64, %138 : i64
          %141 = arith.muli %109, %c8_i64 : i64
          %142 = ascscalar.memref_address %alloc_0 : memref<1xui8>
          %143 = arith.divui %141, %c8_i64 : i64
          %144 = arith.addi %142, %143 : i64
          %145 = ascscalar.bind_gm_ptr %144 : memref<1xui8>
          %146 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<32768xui8, 1> -> memref<?xui8, 1>
          %147 = cannir.extract_stride %114 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %148 = cannir.extract_dim %147[0] : !cannir.stride<(-1, 1)>
          %149 = arith.cmpi ugt, %123, %c0_i64 : i64
          %150 = arith.cmpi ugt, %140, %c0_i64 : i64
          %151 = arith.andi %149, %150 : i1
          scf.if %151 {
            ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
            %193 = arith.trunci %123 : i64 to i32
            %194 = arith.trunci %140 : i64 to i32
            ascvec.copy_ub2gm %145[%c0], %146[%c0], %193, %194, %148, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
          }
          %152 = arith.muli %49, %c512_i64 overflow<nsw> : i64
          %153 = cannir.extract_dim %39[0] : !cannir.shape<(-1, -1)>
          %154 = arith.minsi %153, %c0_i64 : i64
          %155 = arith.subi %153, %154 : i64
          %156 = arith.minsi %155, %c1_i64 : i64
          %157 = arith.cmpi sge, %153, %c0_i64 : i64
          %158 = arith.select %157, %156, %c0_i64 : i64
          %159 = arith.cmpi sgt, %158, %c1_i64 : i64
          %160 = arith.select %159, %c1_i64, %158 : i64
          %161 = arith.maxsi %51, %c1_i64 : i64
          %162 = arith.divsi %40, %161 : i64
          %163 = arith.minsi %36, %162 : i64
          %164 = arith.muli %163, %51 : i64
          %165 = arith.subi %40, %164 : i64
          %166 = arith.minsi %51, %165 : i64
          %167 = arith.cmpi sle, %36, %162 : i64
          %168 = arith.select %167, %166, %c0_i64 : i64
          %169 = arith.maxsi %152, %c1_i64 : i64
          %170 = arith.divsi %168, %169 : i64
          %171 = arith.minsi %arg7, %170 : i64
          %172 = arith.muli %171, %152 : i64
          %173 = arith.subi %168, %172 : i64
          %174 = arith.minsi %152, %173 : i64
          %175 = arith.cmpi sle, %arg7, %170 : i64
          %176 = arith.select %175, %174, %c0_i64 : i64
          %177 = arith.cmpi sgt, %176, %c2048_i64 : i64
          %178 = arith.select %177, %c2048_i64, %176 : i64
          %179 = arith.muli %arg7, %152 : i64
          %180 = arith.muli %36, %51 : i64
          %181 = arith.addi %179, %180 : i64
          %182 = arith.muli %181, %c8_i64 : i64
          %183 = ascscalar.memref_address %alloc_1 : memref<1xui8>
          %184 = arith.divui %182, %c8_i64 : i64
          %185 = arith.addi %183, %184 : i64
          %186 = ascscalar.bind_gm_ptr %185 : memref<1xui8>
          %187 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<2048xui8, 1> -> memref<?xui8, 1>
          %188 = cannir.extract_stride %38 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %189 = cannir.extract_dim %188[0] : !cannir.stride<(-1, 1)>
          %190 = arith.cmpi ugt, %160, %c0_i64 : i64
          %191 = arith.cmpi ugt, %178, %c0_i64 : i64
          %192 = arith.andi %190, %191 : i1
          scf.if %192 {
            ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
            %193 = arith.trunci %160 : i64 to i32
            %194 = arith.trunci %178 : i64 to i32
            ascvec.copy_ub2gm %186[%c0], %187[%c0], %193, %194, %189, %c2048_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %75 : i64
        }
      }
    } else {
      %10 = arith.cmpi sgt, %7, %c1_i64 : i64
      %11 = scf.if %10 -> (i1) {
        %12 = cannir.extract_dim %6[1] : !cannir.shape<(-1, -1)>
        %13 = arith.cmpi slt, %12, %c128_i64 : i64
        scf.yield %13 : i1
      } else {
        scf.yield %false : i1
      }
      scf.if %11 {
        %12 = cannir.extract_dim %6[1] : !cannir.shape<(-1, -1)>
        %13 = arith.cmpi sgt, %12, %c64_i64 : i64
        %14 = arith.select %13, %c7_i64, %c6_i64 : i64
        %15 = arith.select %13, %c256_i64, %c512_i64 : i64
        %16 = ascscalar.get_block_num
        %17 = arith.remsi %7, %16 : i64
        %18 = arith.cmpi ne, %17, %c0_i64 : i64
        %19 = arith.cmpi slt, %17, %c0_i64 : i64
        %20 = arith.cmpi slt, %16, %c0_i64 : i64
        %21 = arith.xori %19, %20 : i1
        %22 = arith.andi %18, %21 : i1
        %23 = arith.divsi %7, %16 : i64
        %24 = arith.extui %22 : i1 to i64
        %25 = arith.subi %23, %24 : i64
        %26 = arith.cmpi slt, %25, %c1_i64 : i64
        %27 = arith.select %26, %c1_i64, %25 : i64
        %28 = arith.cmpi sgt, %27, %15 : i64
        %29 = arith.select %28, %15, %27 : i64
        %30 = arith.addi %7, %29 : i64
        %31 = arith.subi %30, %c1_i64 : i64
        %32 = arith.remsi %31, %29 : i64
        %33 = arith.cmpi ne, %32, %c0_i64 : i64
        %34 = arith.cmpi slt, %32, %c0_i64 : i64
        %35 = arith.andi %33, %34 : i1
        %36 = arith.divsi %31, %29 : i64
        %37 = arith.extui %35 : i1 to i64
        %38 = arith.subi %36, %37 : i64
        %39 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
        %40 = cannir.extract_shape %39 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
        %41 = cannir.extract_dim %40[1] : !cannir.shape<(-1, -1)>
        %42 = arith.muli %7, %12 : i64
        %43 = arith.muli %7, %41 : i64
        %44 = ascscalar.get_block_idx
        %45 = scf.for %arg7 = %44 to %38 step %16 iter_args(%arg8 = %c0_i64) -> (i64)  : i64 {
          %46 = arith.muli %arg7, %29 : i64
          %47 = arith.subi %7, %46 : i64
          %48 = arith.cmpi sgt, %47, %29 : i64
          %49 = arith.select %48, %29, %47 : i64
          %50 = arith.cmpi eq, %arg8, %c0_i64 : i64
          %51 = ascscalar.memref_address %alloc_4 : memref<32768xbf16, 1>
          %52 = ascscalar.memref_address %alloc_5 : memref<32768xbf16, 1>
          %53 = arith.select %50, %51, %52 : i64
          %54 = arith.cmpi ne, %arg8, %c0_i64 : i64
          %55 = arith.extui %54 : i1 to i64
          %56 = arith.addi %arg8, %c1_i64 overflow<nsw> : i64
          %57 = arith.remui %56, %c2_i64 : i64
          %58 = arith.muli %29, %12 : i64
          %59 = arith.maxsi %58, %c1_i64 : i64
          %60 = arith.divsi %42, %59 : i64
          %61 = arith.minsi %arg7, %60 : i64
          %62 = arith.muli %61, %58 : i64
          %63 = arith.subi %42, %62 : i64
          %64 = arith.minsi %58, %63 : i64
          %65 = arith.cmpi sle, %arg7, %60 : i64
          %66 = arith.select %65, %64, %c0_i64 : i64
          %67 = arith.muli %arg7, %58 : i64
          %68 = arith.muli %67, %c2_i64 : i64
          %69 = arith.muli %66, %c2_i64 : i64
          %70 = arith.cmpi eq, %69, %c0_i64 : i64
          %71 = arith.cmpi ne, %69, %c0_i64 : i64
          %72 = arith.extui %71 : i1 to i64
          %73 = arith.select %70, %c2_i64, %69 : i64
          %74 = ascscalar.bind_local_ptr %53 : memref<1xbf16, 1>
          ascvec.get_buf mte2, %55, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%74, %alloc, %c0_i32, %c0_i32, %c0_i64, %68, %72, %73, %c0_i64, %73, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %55, 0 {cannir.buffer.generated_sync}
          %75 = ascscalar.bind_local_ptr %53 : memref<32768xbf16, 1>
          %76 = arith.muli %46, %12 : i64
          %77 = arith.cmpi eq, %14, %c7_i64 : i64
          %78 = arith.select %77, %c128_i64, %c64_i64 : i64
          %79 = arith.muli %49, %78 : i64
          %80 = arith.addi %79, %c255_i64 : i64
          %81 = arith.remsi %80, %c256_i64 : i64
          %82 = arith.cmpi ne, %81, %c0_i64 : i64
          %83 = arith.cmpi slt, %81, %c0_i64 : i64
          %84 = arith.andi %82, %83 : i1
          %85 = arith.divsi %80, %c256_i64 : i64
          %86 = arith.extui %84 : i1 to i64
          %87 = arith.subi %85, %86 : i64
          %88 = arith.divsi %78, %c32_i64 : i64
          %89 = arith.muli %49, %88 : i64
          %90 = arith.trunci %14 : i64 to i32
          %91 = arith.divsi %78, %c4_i64 : i64
          %92 = arith.subi %91, %c1_i64 overflow<nsw> : i64
          %93 = arith.trunci %92 : i64 to i32
          %94 = arith.remsi %12, %c4_i64 : i64
          %95 = arith.cmpi ne, %94, %c0_i64 : i64
          %96 = arith.cmpi slt, %94, %c0_i64 : i64
          %97 = arith.andi %95, %96 : i1
          %98 = arith.divsi %12, %c4_i64 : i64
          %99 = arith.extui %97 : i1 to i64
          %100 = arith.subi %98, %99 : i64
          %101 = arith.trunci %100 : i64 to i32
          %102 = arith.trunci %49 : i64 to i32
          %103 = arith.muli %49, %12 : i64
          %104 = arith.addi %103, %c1023_i64 : i64
          %105 = arith.remsi %104, %c1024_i64 : i64
          %106 = arith.cmpi ne, %105, %c0_i64 : i64
          %107 = arith.cmpi slt, %105, %c0_i64 : i64
          %108 = arith.andi %106, %107 : i1
          %109 = arith.divsi %104, %c1024_i64 : i64
          %110 = arith.extui %108 : i1 to i64
          %111 = arith.subi %109, %110 : i64
          %112 = arith.trunci %1 : i64 to i32
          %113 = arith.extui %112 : i32 to i64
          %114 = arith.subi %1, %113 : i64
          %115 = arith.remsi %114, %c4294967296_i64 : i64
          %116 = arith.cmpi ne, %115, %c0_i64 : i64
          %117 = arith.cmpi slt, %115, %c0_i64 : i64
          %118 = arith.andi %116, %117 : i1
          %119 = arith.divsi %114, %c4294967296_i64 : i64
          %120 = arith.extui %118 : i1 to i64
          %121 = arith.subi %119, %120 : i64
          %122 = arith.trunci %121 : i64 to i32
          %123 = arith.remsi %76, %c16_i64 : i64
          %124 = arith.cmpi ne, %123, %c0_i64 : i64
          %125 = arith.cmpi slt, %123, %c0_i64 : i64
          %126 = arith.andi %124, %125 : i1
          %127 = arith.divsi %76, %c16_i64 : i64
          %128 = arith.extui %126 : i1 to i64
          %129 = arith.subi %127, %128 : i64
          %130 = arith.remsi %12, %c16_i64 : i64
          %131 = arith.cmpi ne, %130, %c0_i64 : i64
          %132 = arith.cmpi slt, %130, %c0_i64 : i64
          %133 = arith.andi %131, %132 : i1
          %134 = arith.divsi %12, %c16_i64 : i64
          %135 = arith.extui %133 : i1 to i64
          %136 = arith.subi %134, %135 : i64
          %137 = arith.remsi %129, %136 : i64
          %138 = arith.cmpi ne, %137, %c0_i64 : i64
          %139 = arith.cmpi slt, %137, %c0_i64 : i64
          %140 = arith.cmpi slt, %136, %c0_i64 : i64
          %141 = arith.xori %139, %140 : i1
          %142 = arith.andi %138, %141 : i1
          %143 = arith.divsi %129, %136 : i64
          %144 = arith.extui %142 : i1 to i64
          %145 = arith.subi %143, %144 : i64
          %146 = arith.select %142, %136, %c0_i64 : i64
          %147 = arith.addi %137, %146 : i64
          %148 = arith.trunci %145 : i64 to i32
          %149 = arith.extui %148 : i32 to i64
          %150 = arith.subi %145, %149 : i64
          %151 = arith.remsi %150, %c4294967296_i64 : i64
          %152 = arith.cmpi ne, %151, %c0_i64 : i64
          %153 = arith.cmpi slt, %151, %c0_i64 : i64
          %154 = arith.andi %152, %153 : i1
          %155 = arith.divsi %150, %c4294967296_i64 : i64
          %156 = arith.extui %154 : i1 to i64
          %157 = arith.subi %155, %156 : i64
          %158 = arith.trunci %157 : i64 to i32
          %159 = arith.trunci %147 : i64 to i32
          %160 = arith.extui %159 : i32 to i64
          %161 = arith.subi %147, %160 : i64
          %162 = arith.remsi %161, %c4294967296_i64 : i64
          %163 = arith.cmpi ne, %162, %c0_i64 : i64
          %164 = arith.cmpi slt, %162, %c0_i64 : i64
          %165 = arith.andi %163, %164 : i1
          %166 = arith.divsi %161, %c4294967296_i64 : i64
          %167 = arith.extui %165 : i1 to i64
          %168 = arith.subi %166, %167 : i64
          %169 = arith.trunci %168 : i64 to i32
          %170 = arith.remsi %c64_i64, %136 : i64
          %171 = arith.cmpi ne, %170, %c0_i64 : i64
          %172 = arith.cmpi slt, %170, %c0_i64 : i64
          %173 = arith.xori %172, %140 : i1
          %174 = arith.andi %171, %173 : i1
          %175 = arith.divsi %c64_i64, %136 : i64
          %176 = arith.extui %174 : i1 to i64
          %177 = arith.subi %175, %176 : i64
          %178 = arith.select %174, %136, %c0_i64 : i64
          %179 = arith.addi %170, %178 : i64
          %180 = arith.trunci %111 : i64 to i32
          ascvec.vf(%136, %4, %112, %122, %alloc_9, %179, %177, %180, %148, %158, %159, %169 : i64, i64, i32, i32, memref<8192xi32, 1>, i64, i64, i32, i32, i32, i32, i32) {
          ^bb0(%arg9: i64, %arg10: i64, %arg11: i32, %arg12: i32, %arg13: memref<8192xi32, 1>, %arg14: i64, %arg15: i64, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32):
            %c1_i32_11 = arith.constant 1 : i32
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
            %213:4 = scf.for %arg21 = %c0_i32_19 to %arg16 step %c1_i32_11 iter_args(%arg22 = %arg17, %arg23 = %arg18, %arg24 = %arg19, %arg25 = %arg20) -> (i32, i32, i32, i32)  : i32 {
              %214 = arith.extui %arg22 : i32 to i64
              %215 = arith.extui %arg23 : i32 to i64
              %216 = arith.muli %215, %c4294967296_i64_20 : i64
              %217 = arith.addi %214, %216 : i64
              %218 = arith.extui %arg24 : i32 to i64
              %219 = arith.extui %arg25 : i32 to i64
              %220 = arith.muli %219, %c4294967296_i64_20 : i64
              %221 = arith.addi %218, %220 : i64
              %222 = arith.muli %arg21, %c256_i32 : i32
              %223 = ascvec.create_mask : !ascvec.mask
              %224 = ascvec.arange %c0_i32_19 : i32 -> <ui32>
              %225 = arith.muli %217, %arg9 : i64
              %226 = arith.addi %225, %221 : i64
              %227 = arith.trunci %226 : i64 to i32
              %228 = arith.extui %227 : i32 to i64
              %229 = arith.subi %226, %228 : i64
              %230 = arith.remsi %229, %c4294967296_i64_20 : i64
              %231 = arith.cmpi ne, %230, %c0_i64_18 : i64
              %232 = arith.cmpi slt, %230, %c0_i64_18 : i64
              %233 = arith.andi %231, %232 : i1
              %234 = arith.divsi %229, %c4294967296_i64_20 : i64
              %235 = arith.extui %233 : i1 to i64
              %236 = arith.subi %234, %235 : i64
              %237 = arith.trunci %236 : i64 to i32
              %238 = ascvec.duplicate_scalar %227, %223 : i32 -> <ui32>
              %239 = ascvec.duplicate_scalar %237, %223 : i32 -> <ui32>
              %240 = ascvec.duplicate_scalar %c0_i32_19, %223 : i32 -> <ui32>
              %carry, %dst = ascvec.addc %238, %224, %223 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
              %carry_21, %dst_22 = ascvec.addc %239, %240, %carry, %223 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
              %241 = arith.trunci %arg10 : i64 to i32
              %242 = arith.extui %241 : i32 to i64
              %243 = arith.subi %arg10, %242 : i64
              %244 = arith.remsi %243, %c4294967296_i64_20 : i64
              %245 = arith.cmpi ne, %244, %c0_i64_18 : i64
              %246 = arith.cmpi slt, %244, %c0_i64_18 : i64
              %247 = arith.andi %245, %246 : i1
              %248 = arith.divsi %243, %c4294967296_i64_20 : i64
              %249 = arith.extui %247 : i1 to i64
              %250 = arith.subi %248, %249 : i64
              %251 = arith.trunci %250 : i64 to i32
              %252 = arith.divui %241, %c4_i32_17 : i32
              %253 = arith.remui %251, %c4_i32_17 : i32
              %254 = arith.muli %253, %c1073741824_i32_16 : i32
              %255 = arith.addi %252, %254 : i32
              %256 = arith.divui %251, %c4_i32_17 : i32
              %257 = ascvec.shr %dst, 14, %223 : <ui32> -> <ui32>
              %258 = ascvec.shl %dst_22, 18, %223 : <ui32> -> <ui32>
              %259 = ascvec.or %257, %258, %223 : <ui32>, <ui32> -> <ui32>
              %260 = ascvec.shr %dst_22, 14, %223 : <ui32> -> <ui32>
              %261 = ascvec.duplicate_scalar %255, %223 : i32 -> <ui32>
              %262 = ascvec.duplicate_scalar %256, %223 : i32 -> <ui32>
              %carry_23, %dst_24 = ascvec.addc %261, %259, %223 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
              %carry_25, %dst_26 = ascvec.addc %262, %260, %carry_23, %223 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
              %263 = ascvec.duplicate_scalar %c16383_i32, %223 : i32 -> <ui32>
              %264 = ascvec.and %dst, %263, %223 : <ui32>, <ui32> -> <ui32>
              %265 = ascvec.duplicate_scalar %c-766435501_i32, %223 : i32 -> <ui32>
              %266 = ascvec.duplicate_scalar %c-845247145_i32, %223 : i32 -> <ui32>
              %dst0, %dst1 = ascvec.mull %240, %265, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_27, %dst1_28 = ascvec.mull %dst_24, %266, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %267 = ascvec.xor %dst1_28, %264, %223 : <ui32>, <ui32> -> <ui32>
              %268 = ascvec.xor %dst1, %dst_26, %223 : <ui32>, <ui32> -> <ui32>
              %269 = ascvec.duplicate_scalar %arg11, %223 : i32 -> <ui32>
              %270 = ascvec.duplicate_scalar %arg12, %223 : i32 -> <ui32>
              %271 = ascvec.xor %267, %269, %223 : <ui32>, <ui32> -> <ui32>
              %272 = ascvec.xor %268, %270, %223 : <ui32>, <ui32> -> <ui32>
              %dst0_29, %dst1_30 = ascvec.mull %271, %265, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_31, %dst1_32 = ascvec.mull %272, %266, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %273 = ascvec.xor %dst1_32, %dst0_27, %223 : <ui32>, <ui32> -> <ui32>
              %274 = ascvec.xor %dst1_30, %dst0, %223 : <ui32>, <ui32> -> <ui32>
              %275 = arith.addi %arg11, %c-1640531527_i32_15 : i32
              %276 = ascvec.duplicate_scalar %275, %223 : i32 -> <ui32>
              %277 = arith.addi %arg12, %c-1150833019_i32_14 : i32
              %278 = ascvec.duplicate_scalar %277, %223 : i32 -> <ui32>
              %279 = ascvec.xor %273, %276, %223 : <ui32>, <ui32> -> <ui32>
              %280 = ascvec.xor %274, %278, %223 : <ui32>, <ui32> -> <ui32>
              %dst0_33, %dst1_34 = ascvec.mull %279, %265, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_35, %dst1_36 = ascvec.mull %280, %266, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %281 = ascvec.xor %dst1_36, %dst0_31, %223 : <ui32>, <ui32> -> <ui32>
              %282 = ascvec.xor %dst1_34, %dst0_29, %223 : <ui32>, <ui32> -> <ui32>
              %283 = arith.addi %arg11, %c1013904242_i32_13 : i32
              %284 = ascvec.duplicate_scalar %283, %223 : i32 -> <ui32>
              %285 = arith.addi %arg12, %c1993301258_i32_12 : i32
              %286 = ascvec.duplicate_scalar %285, %223 : i32 -> <ui32>
              %287 = ascvec.xor %281, %284, %223 : <ui32>, <ui32> -> <ui32>
              %288 = ascvec.xor %282, %286, %223 : <ui32>, <ui32> -> <ui32>
              %dst0_37, %dst1_38 = ascvec.mull %287, %265, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_39, %dst1_40 = ascvec.mull %288, %266, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %289 = ascvec.xor %dst1_40, %dst0_35, %223 : <ui32>, <ui32> -> <ui32>
              %290 = ascvec.xor %dst1_38, %dst0_33, %223 : <ui32>, <ui32> -> <ui32>
              %291 = arith.addi %arg11, %c-626627285_i32 : i32
              %292 = ascvec.duplicate_scalar %291, %223 : i32 -> <ui32>
              %293 = arith.addi %arg12, %c842468239_i32 : i32
              %294 = ascvec.duplicate_scalar %293, %223 : i32 -> <ui32>
              %295 = ascvec.xor %289, %292, %223 : <ui32>, <ui32> -> <ui32>
              %296 = ascvec.xor %290, %294, %223 : <ui32>, <ui32> -> <ui32>
              %dst0_41, %dst1_42 = ascvec.mull %295, %265, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_43, %dst1_44 = ascvec.mull %296, %266, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %297 = ascvec.xor %dst1_44, %dst0_39, %223 : <ui32>, <ui32> -> <ui32>
              %298 = ascvec.xor %dst1_42, %dst0_37, %223 : <ui32>, <ui32> -> <ui32>
              %299 = arith.addi %arg11, %c2027808484_i32 : i32
              %300 = ascvec.duplicate_scalar %299, %223 : i32 -> <ui32>
              %301 = arith.addi %arg12, %c-308364780_i32 : i32
              %302 = ascvec.duplicate_scalar %301, %223 : i32 -> <ui32>
              %303 = ascvec.xor %297, %300, %223 : <ui32>, <ui32> -> <ui32>
              %304 = ascvec.xor %298, %302, %223 : <ui32>, <ui32> -> <ui32>
              %dst0_45, %dst1_46 = ascvec.mull %303, %265, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_47, %dst1_48 = ascvec.mull %304, %266, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %305 = ascvec.xor %dst1_48, %dst0_43, %223 : <ui32>, <ui32> -> <ui32>
              %306 = ascvec.xor %dst1_46, %dst0_41, %223 : <ui32>, <ui32> -> <ui32>
              %307 = arith.addi %arg11, %c387276957_i32 : i32
              %308 = ascvec.duplicate_scalar %307, %223 : i32 -> <ui32>
              %309 = arith.addi %arg12, %c-1459197799_i32 : i32
              %310 = ascvec.duplicate_scalar %309, %223 : i32 -> <ui32>
              %311 = ascvec.xor %305, %308, %223 : <ui32>, <ui32> -> <ui32>
              %312 = ascvec.xor %306, %310, %223 : <ui32>, <ui32> -> <ui32>
              %dst0_49, %dst1_50 = ascvec.mull %311, %265, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_51, %dst1_52 = ascvec.mull %312, %266, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %313 = ascvec.xor %dst1_52, %dst0_47, %223 : <ui32>, <ui32> -> <ui32>
              %314 = ascvec.xor %dst1_50, %dst0_45, %223 : <ui32>, <ui32> -> <ui32>
              %315 = arith.addi %arg11, %c-1253254570_i32 : i32
              %316 = ascvec.duplicate_scalar %315, %223 : i32 -> <ui32>
              %317 = arith.addi %arg12, %c1684936478_i32 : i32
              %318 = ascvec.duplicate_scalar %317, %223 : i32 -> <ui32>
              %319 = ascvec.xor %313, %316, %223 : <ui32>, <ui32> -> <ui32>
              %320 = ascvec.xor %314, %318, %223 : <ui32>, <ui32> -> <ui32>
              %dst0_53, %dst1_54 = ascvec.mull %319, %265, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_55, %dst1_56 = ascvec.mull %320, %266, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %321 = ascvec.xor %dst1_56, %dst0_51, %223 : <ui32>, <ui32> -> <ui32>
              %322 = ascvec.xor %dst1_54, %dst0_49, %223 : <ui32>, <ui32> -> <ui32>
              %323 = arith.addi %arg11, %c1401181199_i32 : i32
              %324 = ascvec.duplicate_scalar %323, %223 : i32 -> <ui32>
              %325 = arith.addi %arg12, %c534103459_i32 : i32
              %326 = ascvec.duplicate_scalar %325, %223 : i32 -> <ui32>
              %327 = ascvec.xor %321, %324, %223 : <ui32>, <ui32> -> <ui32>
              %328 = ascvec.xor %322, %326, %223 : <ui32>, <ui32> -> <ui32>
              %dst0_57, %dst1_58 = ascvec.mull %327, %265, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_59, %dst1_60 = ascvec.mull %328, %266, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %329 = ascvec.xor %dst1_60, %dst0_55, %223 : <ui32>, <ui32> -> <ui32>
              %330 = ascvec.xor %dst1_58, %dst0_53, %223 : <ui32>, <ui32> -> <ui32>
              %331 = arith.addi %arg11, %c-239350328_i32 : i32
              %332 = ascvec.duplicate_scalar %331, %223 : i32 -> <ui32>
              %333 = arith.addi %arg12, %c-616729560_i32 : i32
              %334 = ascvec.duplicate_scalar %333, %223 : i32 -> <ui32>
              %335 = ascvec.xor %329, %332, %223 : <ui32>, <ui32> -> <ui32>
              %336 = ascvec.xor %330, %334, %223 : <ui32>, <ui32> -> <ui32>
              %dst0_61, %dst1_62 = ascvec.mull %335, %265, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %dst0_63, %dst1_64 = ascvec.mull %336, %266, %223 : <ui32>, <ui32> -> <ui32>, <ui32>
              %337 = ascvec.xor %dst1_64, %dst0_59, %223 : <ui32>, <ui32> -> <ui32>
              %338 = ascvec.xor %dst1_62, %dst0_57, %223 : <ui32>, <ui32> -> <ui32>
              %339 = arith.addi %arg11, %c-1879881855_i32 : i32
              %340 = ascvec.duplicate_scalar %339, %223 : i32 -> <ui32>
              %341 = arith.addi %arg12, %c-1767562579_i32 : i32
              %342 = ascvec.duplicate_scalar %341, %223 : i32 -> <ui32>
              %343 = ascvec.xor %337, %340, %223 : <ui32>, <ui32> -> <ui32>
              %344 = ascvec.xor %338, %342, %223 : <ui32>, <ui32> -> <ui32>
              %low, %high = ascvec.intlv %343, %344 : <ui32>, <ui32> -> <ui32>, <ui32>
              %low_65, %high_66 = ascvec.intlv %dst0_63, %dst0_61 : <ui32>, <ui32> -> <ui32>, <ui32>
              %345 = arith.index_cast %222 : i32 to index
              ascvec.storealign_intlv %arg13[%345], %low, %low_65 : memref<8192xi32, 1>, <ui32>, <ui32>
              %346 = arith.addi %222, %c128_i32 : i32
              %347 = arith.index_cast %346 : i32 to index
              ascvec.storealign_intlv %arg13[%347], %high, %high_66 : memref<8192xi32, 1>, <ui32>, <ui32>
              %348 = arith.addi %221, %arg14 : i64
              %349 = arith.cmpi sge, %348, %arg9 : i64
              %350 = arith.extui %349 : i1 to i64
              %351 = arith.addi %217, %arg15 : i64
              %352 = arith.addi %351, %350 : i64
              %353 = arith.muli %350, %arg9 : i64
              %354 = arith.subi %348, %353 : i64
              %355 = arith.trunci %352 : i64 to i32
              %356 = arith.extui %355 : i32 to i64
              %357 = arith.subi %352, %356 : i64
              %358 = arith.remsi %357, %c4294967296_i64_20 : i64
              %359 = arith.cmpi ne, %358, %c0_i64_18 : i64
              %360 = arith.cmpi slt, %358, %c0_i64_18 : i64
              %361 = arith.andi %359, %360 : i1
              %362 = arith.divsi %357, %c4294967296_i64_20 : i64
              %363 = arith.extui %361 : i1 to i64
              %364 = arith.subi %362, %363 : i64
              %365 = arith.trunci %364 : i64 to i32
              %366 = arith.trunci %354 : i64 to i32
              %367 = arith.extui %366 : i32 to i64
              %368 = arith.subi %354, %367 : i64
              %369 = arith.remsi %368, %c4294967296_i64_20 : i64
              %370 = arith.cmpi ne, %369, %c0_i64_18 : i64
              %371 = arith.cmpi slt, %369, %c0_i64_18 : i64
              %372 = arith.andi %370, %371 : i1
              %373 = arith.divsi %368, %c4294967296_i64_20 : i64
              %374 = arith.extui %372 : i1 to i64
              %375 = arith.subi %373, %374 : i64
              %376 = arith.trunci %375 : i64 to i32
              scf.yield %355, %365, %366, %376 : i32, i32, i32, i32
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          %181 = arith.trunci %87 : i64 to i32
          %182 = arith.trunci %89 : i64 to i32
          ascvec.get_buf v, %55, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
          %183 = ascscalar.bitcast_memref %75 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui32, 1>
          %184 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<32768xui8, 1> -> memref<?xui32, 1>
          ascvec.vf(%90, %93, %101, %102, %183, %alloc_9, %184, %alloc_8, %181, %182, %alloc_7 : i32, i32, i32, i32, memref<?xui32, 1>, memref<8192xi32, 1>, memref<?xui32, 1>, memref<1024xui32, 1>, i32, i32, memref<2048xui8, 1>) {
          ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: memref<?xui32, 1>, %arg14: memref<8192xi32, 1>, %arg15: memref<?xui32, 1>, %arg16: memref<1024xui32, 1>, %arg17: i32, %arg18: i32, %arg19: memref<2048xui8, 1>):
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
            %c1_i32_11 = arith.constant 1 : i32
            %c0_12 = arith.constant 0 : index
            %c64_i32 = arith.constant 64 : i32
            %c7_i32 = arith.constant 7 : i32
            %c8_i32 = arith.constant 8 : i32
            %c0_i32_13 = arith.constant 0 : i32
            %213 = ascvec.create_mask : !ascvec.mask
            %214 = ascvec.arange %c0_i32_13 : i32 -> <ui32>
            %mask, %remaining_out = ascvec.update_mask %c8_i32 : !ascvec.mask
            %215 = ascvec.shr %214, 3, %213 : <ui32> -> <ui32>
            %216 = ascvec.duplicate_scalar %c0_i32_13, %213 : i32 -> <ui32>
            %217 = ascvec.duplicate_scalar %arg9, %213 : i32 -> <ui32>
            %218 = ascvec.cmp_eq_scalar %217, %c7_i32, %213 : <ui32>, i32 -> !ascvec.mask
            scf.for %arg20 = %c0_i32_13 to %arg17 step %c1_i32_11  : i32 {
              %219 = arith.muli %arg20, %c64_i32 : i32
              %220 = ascvec.add_scalar %214, %219, %213 : <ui32>, i32
              %221 = ascvec.shr %220, 5, %213 : <ui32> -> <ui32>
              %222 = ascvec.shr %220, 4, %213 : <ui32> -> <ui32>
              %223 = ascvec.select %221, %222, %218 : <ui32>
              %224 = ascvec.duplicate_scalar %arg10, %213 : i32 -> <ui32>
              %225 = ascvec.and %220, %224, %213 : <ui32>, <ui32> -> <ui32>
              %226 = ascvec.mul_scalar %223, %arg11, %213 : <ui32>, i32
              %227 = ascvec.add %226, %225, %213 : <ui32>
              %228 = ascvec.cmp_lt_scalar %223, %arg12, %213 : <ui32>, i32 -> !ascvec.mask
              %229 = ascvec.cmp_lt_scalar %225, %arg11, %213 : <ui32>, i32 -> !ascvec.mask
              %230 = ascvec.mask_and %228, %229, %213
              %231 = ascvec.shl %227, 1, %213 : <ui32> -> <ui32>
              %232 = ascvec.gather %arg13[%c0_12], %231, %230 : memref<?xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
              %233 = ascvec.add_scalar %231, %c1_i32_11, %213 : <ui32>, i32
              %234 = ascvec.gather %arg13[%c0_12], %233, %230 : memref<?xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
              %235 = ascvec.select %232, %216, %230 : <ui32>
              %236 = ascvec.select %234, %216, %230 : <ui32>
              %237 = ascvec.shl %235, 16, %213 : <ui32> -> <ui32>
              %238 = ascvec.reinterpret %237 : <ui32> -> <f32>
              %239 = ascvec.duplicate_scalar %c-65536_i32, %213 : i32 -> <ui32>
              %240 = ascvec.and %235, %239, %213 : <ui32>, <ui32> -> <ui32>
              %241 = ascvec.reinterpret %240 : <ui32> -> <f32>
              %242 = ascvec.shl %236, 16, %213 : <ui32> -> <ui32>
              %243 = ascvec.reinterpret %242 : <ui32> -> <f32>
              %244 = ascvec.and %236, %239, %213 : <ui32>, <ui32> -> <ui32>
              %245 = ascvec.reinterpret %244 : <ui32> -> <f32>
              %246 = ascvec.reinterpret %238 : <f32> -> <ui32>
              %247 = ascvec.duplicate_scalar %c2147483647_i32, %213 : i32 -> <ui32>
              %248 = ascvec.and %246, %247, %213 : <ui32>, <ui32> -> <ui32>
              %249 = ascvec.reinterpret %241 : <f32> -> <ui32>
              %250 = ascvec.and %249, %247, %213 : <ui32>, <ui32> -> <ui32>
              %251 = ascvec.reinterpret %243 : <f32> -> <ui32>
              %252 = ascvec.and %251, %247, %213 : <ui32>, <ui32> -> <ui32>
              %253 = ascvec.reinterpret %245 : <f32> -> <ui32>
              %254 = ascvec.and %253, %247, %213 : <ui32>, <ui32> -> <ui32>
              %255 = ascvec.max %248, %250, %213 : <ui32>
              %256 = ascvec.max %252, %254, %213 : <ui32>
              %257 = ascvec.max %255, %256, %213 : <ui32>
              %258 = ascvec.reduce_max_datablock %257, %213 : <ui32>
              %259 = ascvec.shr %258, 23, %213 : <ui32> -> <ui32>
              %260 = ascvec.cmp_ge_scalar %259, %c255_i32, %213 : <ui32>, i32 -> !ascvec.mask
              %261 = ascvec.max_scalar %259, %c15_i32, %213 : <ui32>, i32
              %262 = ascvec.add_scalar %261, %c-15_i32, %213 : <ui32>, i32
              %263 = ascvec.duplicate_scalar %c255_i32, %260 merge %262 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
              %264 = ascvec.duplicate_scalar %c254_i32, %213 : i32 -> <ui32>
              %265 = ascvec.sub %264, %263, %213 : <ui32>
              %266 = ascvec.cmp_ne_scalar %263, %c0_i32_13, %213 : <ui32>, i32 -> !ascvec.mask
              %267 = ascvec.shl %265, 23, %266 : <ui32> -> <ui32>
              %268 = ascvec.duplicate_scalar %c2139160576_i32, %260 merge %267 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
              %269 = ascvec.reinterpret %268 : <ui32> -> <f32>
              %270 = ascvec.gather_reg %269, %215 : <f32>, <ui32> -> <f32>
              %271 = ascvec.mul %238, %270, %213 : <f32>
              %272 = ascvec.mul %241, %270, %213 : <f32>
              %273 = ascvec.mul %243, %270, %213 : <f32>
              %274 = ascvec.mul %245, %270, %213 : <f32>
              %275 = ascvec.gather %arg14[%c0_12], %227, %230 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
              %276 = ascvec.reinterpret %275 : <i32> -> <ui32>
              %277 = ascvec.shl %276, 8, %230 : <ui32> -> <ui32>
              %278 = ascvec.shr %276, 24, %230 : <ui32> -> <ui32>
              %279 = ascvec.or %277, %278, %230 : <ui32>, <ui32> -> <ui32>
              %280 = ascvec.shr %276, 16, %213 : <ui32> -> <ui32>
              %281 = ascvec.duplicate_scalar %c65535_i32, %213 : i32 -> <ui32>
              %282 = ascvec.and %276, %281, %213 : <ui32>, <ui32> -> <ui32>
              %283 = ascvec.reinterpret %271 : <f32> -> <ui32>
              %284 = ascvec.shr %283, 24, %213 : <ui32> -> <ui32>
              %285 = ascvec.duplicate_scalar %c128_i32, %213 : i32 -> <ui32>
              %286 = ascvec.and %284, %285, %213 : <ui32>, <ui32> -> <ui32>
              %287 = ascvec.and %283, %247, %213 : <ui32>, <ui32> -> <ui32>
              %288 = ascvec.shr %287, 5, %213 : <ui32> -> <ui32>
              %289 = ascvec.duplicate_scalar %c29360128_i32, %213 : i32 -> <ui32>
              %290 = ascvec.sub %288, %289, %213 : <ui32>
              %291 = ascvec.reinterpret %287 : <ui32> -> <f32>
              %292 = ascvec.mul_scalar %291, %cst, %213 : <f32>, f32
              %293 = ascvec.cast %292, %213 {rounding = 4 : i32} : <f32> -> <i32>
              %294 = ascvec.reinterpret %293 : <i32> -> <ui32>
              %295 = ascvec.cmp_ge_scalar %287, %c947912704_i32, %213 : <ui32>, i32 -> !ascvec.mask
              %296 = ascvec.select %290, %294, %295 : <ui32>
              %297 = ascvec.and %280, %281, %213 : <ui32>, <ui32> -> <ui32>
              %298 = ascvec.add %296, %297, %213 : <ui32>
              %299 = ascvec.shr %298, 16, %213 : <ui32> -> <ui32>
              %300 = ascvec.min_scalar %299, %c123_i32, %213 : <ui32>, i32
              %301 = ascvec.or %300, %286, %213 : <ui32>, <ui32> -> <ui32>
              %302 = ascvec.duplicate_scalar %c127_i32, %213 : i32 -> <ui32>
              %303 = ascvec.cmp_gt_scalar %287, %c2139095040_i32, %213 : <ui32>, i32 -> !ascvec.mask
              %304 = ascvec.select %302, %301, %303 : <ui32>
              %305 = ascvec.and %279, %281, %213 : <ui32>, <ui32> -> <ui32>
              %306 = ascvec.reinterpret %272 : <f32> -> <ui32>
              %307 = ascvec.shr %306, 24, %213 : <ui32> -> <ui32>
              %308 = ascvec.and %307, %285, %213 : <ui32>, <ui32> -> <ui32>
              %309 = ascvec.and %306, %247, %213 : <ui32>, <ui32> -> <ui32>
              %310 = ascvec.shr %309, 5, %213 : <ui32> -> <ui32>
              %311 = ascvec.sub %310, %289, %213 : <ui32>
              %312 = ascvec.reinterpret %309 : <ui32> -> <f32>
              %313 = ascvec.mul_scalar %312, %cst, %213 : <f32>, f32
              %314 = ascvec.cast %313, %213 {rounding = 4 : i32} : <f32> -> <i32>
              %315 = ascvec.reinterpret %314 : <i32> -> <ui32>
              %316 = ascvec.cmp_ge_scalar %309, %c947912704_i32, %213 : <ui32>, i32 -> !ascvec.mask
              %317 = ascvec.select %311, %315, %316 : <ui32>
              %318 = ascvec.and %305, %281, %213 : <ui32>, <ui32> -> <ui32>
              %319 = ascvec.add %317, %318, %213 : <ui32>
              %320 = ascvec.shr %319, 16, %213 : <ui32> -> <ui32>
              %321 = ascvec.min_scalar %320, %c123_i32, %213 : <ui32>, i32
              %322 = ascvec.or %321, %308, %213 : <ui32>, <ui32> -> <ui32>
              %323 = ascvec.cmp_gt_scalar %309, %c2139095040_i32, %213 : <ui32>, i32 -> !ascvec.mask
              %324 = ascvec.select %302, %322, %323 : <ui32>
              %325 = ascvec.reinterpret %273 : <f32> -> <ui32>
              %326 = ascvec.shr %325, 24, %213 : <ui32> -> <ui32>
              %327 = ascvec.and %326, %285, %213 : <ui32>, <ui32> -> <ui32>
              %328 = ascvec.and %325, %247, %213 : <ui32>, <ui32> -> <ui32>
              %329 = ascvec.shr %328, 5, %213 : <ui32> -> <ui32>
              %330 = ascvec.sub %329, %289, %213 : <ui32>
              %331 = ascvec.reinterpret %328 : <ui32> -> <f32>
              %332 = ascvec.mul_scalar %331, %cst, %213 : <f32>, f32
              %333 = ascvec.cast %332, %213 {rounding = 4 : i32} : <f32> -> <i32>
              %334 = ascvec.reinterpret %333 : <i32> -> <ui32>
              %335 = ascvec.cmp_ge_scalar %328, %c947912704_i32, %213 : <ui32>, i32 -> !ascvec.mask
              %336 = ascvec.select %330, %334, %335 : <ui32>
              %337 = ascvec.and %282, %281, %213 : <ui32>, <ui32> -> <ui32>
              %338 = ascvec.add %336, %337, %213 : <ui32>
              %339 = ascvec.shr %338, 16, %213 : <ui32> -> <ui32>
              %340 = ascvec.min_scalar %339, %c123_i32, %213 : <ui32>, i32
              %341 = ascvec.or %340, %327, %213 : <ui32>, <ui32> -> <ui32>
              %342 = ascvec.cmp_gt_scalar %328, %c2139095040_i32, %213 : <ui32>, i32 -> !ascvec.mask
              %343 = ascvec.select %302, %341, %342 : <ui32>
              %344 = ascvec.shr %279, 16, %213 : <ui32> -> <ui32>
              %345 = ascvec.reinterpret %274 : <f32> -> <ui32>
              %346 = ascvec.shr %345, 24, %213 : <ui32> -> <ui32>
              %347 = ascvec.and %346, %285, %213 : <ui32>, <ui32> -> <ui32>
              %348 = ascvec.and %345, %247, %213 : <ui32>, <ui32> -> <ui32>
              %349 = ascvec.shr %348, 5, %213 : <ui32> -> <ui32>
              %350 = ascvec.sub %349, %289, %213 : <ui32>
              %351 = ascvec.reinterpret %348 : <ui32> -> <f32>
              %352 = ascvec.mul_scalar %351, %cst, %213 : <f32>, f32
              %353 = ascvec.cast %352, %213 {rounding = 4 : i32} : <f32> -> <i32>
              %354 = ascvec.reinterpret %353 : <i32> -> <ui32>
              %355 = ascvec.cmp_ge_scalar %348, %c947912704_i32, %213 : <ui32>, i32 -> !ascvec.mask
              %356 = ascvec.select %350, %354, %355 : <ui32>
              %357 = ascvec.and %344, %281, %213 : <ui32>, <ui32> -> <ui32>
              %358 = ascvec.add %356, %357, %213 : <ui32>
              %359 = ascvec.shr %358, 16, %213 : <ui32> -> <ui32>
              %360 = ascvec.min_scalar %359, %c123_i32, %213 : <ui32>, i32
              %361 = ascvec.or %360, %347, %213 : <ui32>, <ui32> -> <ui32>
              %362 = ascvec.cmp_gt_scalar %348, %c2139095040_i32, %213 : <ui32>, i32 -> !ascvec.mask
              %363 = ascvec.select %302, %361, %362 : <ui32>
              %364 = ascvec.shl %324, 8, %213 : <ui32> -> <ui32>
              %365 = ascvec.or %304, %364, %213 : <ui32>, <ui32> -> <ui32>
              %366 = ascvec.shl %343, 16, %213 : <ui32> -> <ui32>
              %367 = ascvec.shl %363, 24, %213 : <ui32> -> <ui32>
              %368 = ascvec.or %366, %367, %213 : <ui32>, <ui32> -> <ui32>
              %369 = ascvec.or %365, %368, %213 : <ui32>, <ui32> -> <ui32>
              ascvec.scatter %arg15[%c0_12], %369, %227, %230 : memref<?xui32, 1>, <ui32>, <ui32>, !ascvec.mask
              %370 = arith.muli %arg20, %c8_i32 : i32
              %371 = arith.index_cast %370 : i32 to index
              ascvec.storealign %arg16[%371], %263, %mask : memref<1024xui32, 1>, index, <ui32>
            }
            ascvec.mem_bar vst_vld
            scf.for %arg20 = %c0_i32_13 to %arg18 step %c64_i32  : i32 {
              %219 = arith.index_cast %arg20 : i32 to index
              %220 = ascvec.loadalign %arg16[%219] : memref<1024xui32, 1>, index -> <ui32>
              %221 = arith.subi %arg18, %arg20 : i32
              %mask_14, %remaining_out_15 = ascvec.update_mask %221 : !ascvec.mask
              ascvec.storealign_pack %arg19[%219], %220, %mask_14 {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
            }
            ascvec.mem_bar vst_vld
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %55, 0 {cannir.buffer.generated_sync}
          %185 = arith.cmpi sgt, %66, %c32768_i64 : i64
          %186 = arith.select %185, %c32768_i64, %66 : i64
          %187 = arith.muli %67, %c8_i64 : i64
          %188 = ascscalar.memref_address %alloc_0 : memref<1xui8>
          %189 = arith.divui %187, %c8_i64 : i64
          %190 = arith.addi %188, %189 : i64
          %191 = ascscalar.bind_gm_ptr %190 : memref<1xui8>
          %192 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<32768xui8, 1> -> memref<?xui8, 1>
          %193 = arith.cmpi ugt, %186, %c0_i64 : i64
          scf.if %193 {
            ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
            %213 = arith.trunci %186 : i64 to i32
            ascvec.copy_ub2gm %191[%c0], %192[%c0], %c1_i32, %213, %42, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
          }
          %194 = arith.muli %29, %41 : i64
          %195 = arith.maxsi %194, %c1_i64 : i64
          %196 = arith.divsi %43, %195 : i64
          %197 = arith.minsi %arg7, %196 : i64
          %198 = arith.muli %197, %194 : i64
          %199 = arith.subi %43, %198 : i64
          %200 = arith.minsi %194, %199 : i64
          %201 = arith.cmpi sle, %arg7, %196 : i64
          %202 = arith.select %201, %200, %c0_i64 : i64
          %203 = arith.cmpi sgt, %202, %c2048_i64 : i64
          %204 = arith.select %203, %c2048_i64, %202 : i64
          %205 = arith.muli %arg7, %194 : i64
          %206 = arith.muli %205, %c8_i64 : i64
          %207 = ascscalar.memref_address %alloc_1 : memref<1xui8>
          %208 = arith.divui %206, %c8_i64 : i64
          %209 = arith.addi %207, %208 : i64
          %210 = ascscalar.bind_gm_ptr %209 : memref<1xui8>
          %211 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<2048xui8, 1> -> memref<?xui8, 1>
          %212 = arith.cmpi ugt, %204, %c0_i64 : i64
          scf.if %212 {
            ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
            %213 = arith.trunci %204 : i64 to i32
            ascvec.copy_ub2gm %210[%c0], %211[%c0], %c1_i32, %213, %43, %c2048_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %57 : i64
        }
      } else {
        %12 = cannir.extract_dim %6[1] : !cannir.shape<(-1, -1)>
        %13 = arith.addi %12, %c63_i64 : i64
        %14 = arith.remsi %13, %c64_i64 : i64
        %15 = arith.cmpi ne, %14, %c0_i64 : i64
        %16 = arith.cmpi slt, %14, %c0_i64 : i64
        %17 = arith.andi %15, %16 : i1
        %18 = arith.divsi %13, %c64_i64 : i64
        %19 = arith.extui %17 : i1 to i64
        %20 = arith.subi %18, %19 : i64
        %21 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
        %22 = cannir.extract_shape %21 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
        %23 = cannir.extract_dim %22[1] : !cannir.shape<(-1, -1)>
        %24 = arith.remsi %12, %c16_i64 : i64
        %25 = arith.cmpi ne, %24, %c0_i64 : i64
        %26 = arith.cmpi slt, %24, %c0_i64 : i64
        %27 = arith.andi %25, %26 : i1
        %28 = arith.divsi %12, %c16_i64 : i64
        %29 = arith.extui %27 : i1 to i64
        %30 = arith.subi %28, %29 : i64
        %31 = arith.cmpi eq, %23, %30 : i64
        %32 = arith.select %31, %c4_i64, %c2_i64 : i64
        %33 = arith.muli %7, %20 : i64
        %34 = ascscalar.get_block_num
        %35 = arith.addi %33, %34 : i64
        %36 = arith.subi %35, %c1_i64 : i64
        %37 = arith.remsi %36, %34 : i64
        %38 = arith.cmpi ne, %37, %c0_i64 : i64
        %39 = arith.cmpi slt, %37, %c0_i64 : i64
        %40 = arith.cmpi slt, %34, %c0_i64 : i64
        %41 = arith.xori %39, %40 : i1
        %42 = arith.andi %38, %41 : i1
        %43 = arith.divsi %36, %34 : i64
        %44 = arith.extui %42 : i1 to i64
        %45 = arith.subi %43, %44 : i64
        %46 = arith.addi %45, %c511_i64 : i64
        %47 = arith.remsi %46, %c512_i64 : i64
        %48 = arith.cmpi ne, %47, %c0_i64 : i64
        %49 = arith.cmpi slt, %47, %c0_i64 : i64
        %50 = arith.andi %48, %49 : i1
        %51 = arith.divsi %46, %c512_i64 : i64
        %52 = arith.extui %50 : i1 to i64
        %53 = arith.subi %51, %52 : i64
        %54 = arith.addi %45, %53 : i64
        %55 = arith.subi %54, %c1_i64 : i64
        %56 = arith.remsi %55, %53 : i64
        %57 = arith.cmpi ne, %56, %c0_i64 : i64
        %58 = arith.cmpi slt, %56, %c0_i64 : i64
        %59 = arith.cmpi slt, %53, %c0_i64 : i64
        %60 = arith.xori %58, %59 : i1
        %61 = arith.andi %57, %60 : i1
        %62 = arith.divsi %55, %53 : i64
        %63 = arith.extui %61 : i1 to i64
        %64 = arith.subi %62, %63 : i64
        %65 = arith.cmpi sge, %64, %c4_i64 : i64
        %66 = scf.if %65 -> (i64) {
          %85 = arith.addi %64, %c3_i64 : i64
          %86 = arith.remsi %85, %c4_i64 : i64
          %87 = arith.cmpi ne, %86, %c0_i64 : i64
          %88 = arith.cmpi slt, %86, %c0_i64 : i64
          %89 = arith.andi %87, %88 : i1
          %90 = arith.divsi %85, %c4_i64 : i64
          %91 = arith.extui %89 : i1 to i64
          %92 = arith.subi %90, %91 : i64
          %93 = arith.muli %92, %c4_i64 : i64
          scf.yield %93 : i64
        } else {
          scf.yield %64 : i64
        }
        %67 = arith.muli %34, %c64_i64 : i64
        %68 = arith.cmpi sge, %33, %67 : i64
        %69 = scf.if %68 -> (i64) {
          %85 = arith.addi %66, %c15_i64 : i64
          %86 = arith.remsi %85, %c16_i64 : i64
          %87 = arith.cmpi ne, %86, %c0_i64 : i64
          %88 = arith.cmpi slt, %86, %c0_i64 : i64
          %89 = arith.andi %87, %88 : i1
          %90 = arith.divsi %85, %c16_i64 : i64
          %91 = arith.extui %89 : i1 to i64
          %92 = arith.subi %90, %91 : i64
          %93 = arith.muli %92, %c16_i64 : i64
          scf.yield %93 : i64
        } else {
          scf.yield %66 : i64
        }
        %70 = arith.muli %69, %c64_i64 : i64
        %71 = arith.addi %20, %69 : i64
        %72 = arith.subi %71, %c1_i64 : i64
        %73 = arith.remsi %72, %69 : i64
        %74 = arith.cmpi ne, %73, %c0_i64 : i64
        %75 = arith.cmpi slt, %73, %c0_i64 : i64
        %76 = arith.cmpi slt, %69, %c0_i64 : i64
        %77 = arith.xori %75, %76 : i1
        %78 = arith.andi %74, %77 : i1
        %79 = arith.divsi %72, %69 : i64
        %80 = arith.extui %78 : i1 to i64
        %81 = arith.subi %79, %80 : i64
        %82 = arith.muli %7, %81 : i64
        %83 = ascscalar.get_block_idx
        %84 = scf.for %arg7 = %83 to %82 step %34 iter_args(%arg8 = %c0_i64) -> (i64)  : i64 {
          %85 = arith.remsi %arg7, %81 : i64
          %86 = arith.cmpi ne, %85, %c0_i64 : i64
          %87 = arith.cmpi slt, %85, %c0_i64 : i64
          %88 = arith.cmpi slt, %81, %c0_i64 : i64
          %89 = arith.xori %87, %88 : i1
          %90 = arith.andi %86, %89 : i1
          %91 = arith.divsi %arg7, %81 : i64
          %92 = arith.extui %90 : i1 to i64
          %93 = arith.subi %91, %92 : i64
          %94 = arith.select %90, %81, %c0_i64 : i64
          %95 = arith.addi %85, %94 : i64
          %96 = arith.muli %95, %70 : i64
          %97 = arith.subi %12, %96 : i64
          %98 = arith.cmpi sgt, %97, %70 : i64
          %99 = arith.select %98, %70, %97 : i64
          %100 = arith.cmpi eq, %arg8, %c0_i64 : i64
          %101 = ascscalar.memref_address %alloc_4 : memref<32768xbf16, 1>
          %102 = ascscalar.memref_address %alloc_5 : memref<32768xbf16, 1>
          %103 = arith.select %100, %101, %102 : i64
          %104 = arith.cmpi ne, %arg8, %c0_i64 : i64
          %105 = arith.extui %104 : i1 to i64
          %106 = arith.addi %arg8, %c1_i64 overflow<nsw> : i64
          %107 = arith.remui %106, %c2_i64 : i64
          %108 = arith.minsi %93, %7 : i64
          %109 = arith.subi %7, %108 : i64
          %110 = arith.minsi %109, %c1_i64 : i64
          %111 = arith.cmpi sle, %93, %7 : i64
          %112 = arith.select %111, %110, %c0_i64 : i64
          %113 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %114 = cannir.extract_stride %113 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %115 = cannir.extract_dim %114[0] : !cannir.stride<(-1, 1)>
          %116 = arith.muli %115, %c2_i64 : i64
          %117 = arith.maxsi %70, %c1_i64 : i64
          %118 = arith.divsi %12, %117 : i64
          %119 = arith.minsi %95, %118 : i64
          %120 = arith.muli %119, %70 : i64
          %121 = arith.subi %12, %120 : i64
          %122 = arith.minsi %70, %121 : i64
          %123 = arith.cmpi sle, %95, %118 : i64
          %124 = arith.select %123, %122, %c0_i64 : i64
          %125 = arith.muli %93, %115 : i64
          %126 = arith.muli %125, %c2_i64 : i64
          %127 = arith.muli %96, %c2_i64 : i64
          %128 = arith.addi %126, %127 : i64
          %129 = arith.muli %124, %c2_i64 : i64
          %130 = arith.cmpi eq, %129, %c0_i64 : i64
          %131 = arith.select %130, %c0_i64, %112 : i64
          %132 = arith.select %130, %c2_i64, %129 : i64
          %133 = ascscalar.bind_local_ptr %103 : memref<1xbf16, 1>
          ascvec.get_buf mte2, %105, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%133, %alloc, %c0_i32, %c0_i32, %c0_i64, %128, %131, %132, %116, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %105, 0 {cannir.buffer.generated_sync}
          %134 = ascscalar.bind_local_ptr %103 : memref<32768xbf16, 1>
          %135 = arith.cmpi sge, %99, %c256_i64 : i64
          %136 = arith.remsi %99, %c256_i64 : i64
          %137 = arith.cmpi eq, %136, %c0_i64 : i64
          %138 = arith.select %135, %137, %false : i1
          scf.if %138 {
            %211 = arith.muli %93, %12 : i64
            %212 = arith.addi %211, %96 : i64
            %213 = arith.remsi %arg3, %c256_i64 : i64
            %214 = arith.cmpi ne, %213, %c0_i64 : i64
            %215 = arith.cmpi slt, %213, %c0_i64 : i64
            %216 = arith.andi %214, %215 : i1
            %217 = arith.select %216, %c256_i64, %c0_i64 : i64
            %218 = arith.addi %213, %217 overflow<nsw> : i64
            %219 = arith.cmpi eq, %218, %c0_i64 : i64
            scf.if %219 {
              %220 = arith.cmpi sge, %99, %c1024_i64 : i64
              %221 = arith.remsi %99, %c1024_i64 : i64
              %222 = arith.cmpi eq, %221, %c0_i64 : i64
              %223 = arith.select %220, %222, %false : i1
              scf.if %223 {
                %224 = arith.divsi %99, %c256_i64 : i64
                %225 = arith.addi %99, %c1023_i64 : i64
                %226 = arith.remsi %225, %c1024_i64 : i64
                %227 = arith.cmpi ne, %226, %c0_i64 : i64
                %228 = arith.cmpi slt, %226, %c0_i64 : i64
                %229 = arith.andi %227, %228 : i1
                %230 = arith.divsi %225, %c1024_i64 : i64
                %231 = arith.extui %229 : i1 to i64
                %232 = arith.subi %230, %231 : i64
                %233 = arith.trunci %1 : i64 to i32
                %234 = arith.extui %233 : i32 to i64
                %235 = arith.subi %1, %234 : i64
                %236 = arith.remsi %235, %c4294967296_i64 : i64
                %237 = arith.cmpi ne, %236, %c0_i64 : i64
                %238 = arith.cmpi slt, %236, %c0_i64 : i64
                %239 = arith.andi %237, %238 : i1
                %240 = arith.divsi %235, %c4294967296_i64 : i64
                %241 = arith.extui %239 : i1 to i64
                %242 = arith.subi %240, %241 : i64
                %243 = arith.trunci %242 : i64 to i32
                %244 = arith.divui %212, %c32_i64 : i64
                %245 = arith.trunci %244 : i64 to i32
                %246 = arith.extui %245 : i32 to i64
                %247 = arith.subi %244, %246 : i64
                %248 = arith.remsi %247, %c4294967296_i64 : i64
                %249 = arith.cmpi ne, %248, %c0_i64 : i64
                %250 = arith.cmpi slt, %248, %c0_i64 : i64
                %251 = arith.andi %249, %250 : i1
                %252 = arith.divsi %247, %c4294967296_i64 : i64
                %253 = arith.extui %251 : i1 to i64
                %254 = arith.subi %252, %253 : i64
                %255 = arith.trunci %254 : i64 to i32
                %256 = arith.trunci %4 : i64 to i32
                %257 = arith.extui %256 : i32 to i64
                %258 = arith.subi %4, %257 : i64
                %259 = arith.remsi %258, %c4294967296_i64 : i64
                %260 = arith.cmpi ne, %259, %c0_i64 : i64
                %261 = arith.cmpi slt, %259, %c0_i64 : i64
                %262 = arith.andi %260, %261 : i1
                %263 = arith.divsi %258, %c4294967296_i64 : i64
                %264 = arith.extui %262 : i1 to i64
                %265 = arith.subi %263, %264 : i64
                %266 = arith.trunci %265 : i64 to i32
                %267 = arith.divui %256, %c4_i32 : i32
                %268 = arith.remui %266, %c4_i32 : i32
                %269 = arith.muli %268, %c1073741824_i32 : i32
                %270 = arith.addi %267, %269 : i32
                %271 = arith.divui %266, %c4_i32 : i32
                %272 = arith.muli %232, %c32_i64 : i64
                %273 = arith.divui %244, %c16384_i64 : i64
                %274 = arith.addi %244, %272 : i64
                %275 = arith.subi %274, %c1_i64 : i64
                %276 = arith.divui %275, %c16384_i64 : i64
                %277 = arith.cmpi eq, %273, %276 : i64
                scf.if %277 {
                  %287 = arith.trunci %273 : i64 to i32
                  %288 = arith.extui %287 : i32 to i64
                  %289 = arith.subi %273, %288 : i64
                  %290 = arith.remsi %289, %c4294967296_i64 : i64
                  %291 = arith.cmpi ne, %290, %c0_i64 : i64
                  %292 = arith.cmpi slt, %290, %c0_i64 : i64
                  %293 = arith.andi %291, %292 : i1
                  %294 = arith.divsi %289, %c4294967296_i64 : i64
                  %295 = arith.extui %293 : i1 to i64
                  %296 = arith.subi %294, %295 : i64
                  %297 = arith.trunci %296 : i64 to i32
                  %298 = arith.addi %270, %287 : i32
                  %299 = arith.addi %271, %297 : i32
                  %300 = arith.cmpi ult, %298, %270 : i32
                  %301 = arith.extui %300 : i1 to i32
                  %302 = arith.addi %299, %301 : i32
                  %303 = arith.extui %298 : i32 to i64
                  %304 = arith.muli %303, %c3449720151_i64 : i64
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
                  %316 = arith.xori %302, %243 : i32
                  %317 = arith.extui %316 : i32 to i64
                  %318 = arith.muli %317, %c3449720151_i64 : i64
                  %319 = arith.trunci %318 : i64 to i32
                  %320 = arith.extui %319 : i32 to i64
                  %321 = arith.subi %318, %320 : i64
                  %322 = arith.remsi %321, %c4294967296_i64 : i64
                  %323 = arith.cmpi ne, %322, %c0_i64 : i64
                  %324 = arith.cmpi slt, %322, %c0_i64 : i64
                  %325 = arith.andi %323, %324 : i1
                  %326 = arith.divsi %321, %c4294967296_i64 : i64
                  %327 = arith.extui %325 : i1 to i64
                  %328 = arith.subi %326, %327 : i64
                  %329 = arith.trunci %328 : i64 to i32
                  %330 = arith.addi %233, %c-1640531527_i32 : i32
                  %331 = arith.addi %243, %c-1150833019_i32 : i32
                  %332 = arith.xori %329, %305 : i32
                  %333 = arith.xori %332, %330 : i32
                  %334 = arith.extui %333 : i32 to i64
                  %335 = arith.muli %334, %c3528531795_i64 : i64
                  %336 = arith.trunci %335 : i64 to i32
                  %337 = arith.extui %336 : i32 to i64
                  %338 = arith.subi %335, %337 : i64
                  %339 = arith.remsi %338, %c4294967296_i64 : i64
                  %340 = arith.cmpi ne, %339, %c0_i64 : i64
                  %341 = arith.cmpi slt, %339, %c0_i64 : i64
                  %342 = arith.andi %340, %341 : i1
                  %343 = arith.divsi %338, %c4294967296_i64 : i64
                  %344 = arith.extui %342 : i1 to i64
                  %345 = arith.subi %343, %344 : i64
                  %346 = arith.trunci %345 : i64 to i32
                  %347 = arith.xori %315, %233 : i32
                  %348 = arith.addi %233, %c1013904242_i32 : i32
                  %349 = arith.xori %319, %348 : i32
                  %350 = arith.addi %243, %c1993301258_i32 : i32
                  %351 = arith.xori %346, %350 : i32
                  ascsimt.vf(%245, %347, %331, %349, %351, %233, %243, %336, %alloc_9, %272 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: memref<8192xi32, 1>, %arg18: i64):
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
                    %c16384_i32 = arith.constant 16384 : i32
                    %352 = ascsimt.thread_idx {ascsimt.scope = 31 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %353 = arith.extsi %352 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg19 = %353 to %arg18 step %c256_i64_11  : i64 {
                      %354 = arith.trunci %arg19 : i64 to i32
                      %355 = arith.addi %arg9, %354 : i32
                      %356 = arith.remui %355, %c16384_i32 : i32
                      %357 = arith.xori %356, %arg10 : i32
                      %358 = arith.muli %357, %c-766435501_i32 : i32
                      %359 = "ascsimt.bitcast"(%357) : (i32) -> ui32
                      %360 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %361 = ascsimt.math_binary %359, %360, "mulhi" : ui32, ui32 -> ui32
                      %362 = "ascsimt.bitcast"(%361) : (ui32) -> i32
                      %363 = arith.xori %362, %arg11 : i32
                      %364 = arith.muli %363, %c-845247145_i32 : i32
                      %365 = "ascsimt.bitcast"(%363) : (i32) -> ui32
                      %366 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %367 = ascsimt.math_binary %365, %366, "mulhi" : ui32, ui32 -> ui32
                      %368 = "ascsimt.bitcast"(%367) : (ui32) -> i32
                      %369 = arith.xori %368, %arg12 : i32
                      %370 = arith.xori %358, %arg13 : i32
                      %371:4 = scf.for %arg20 = %c3_i32 to %c10_i32 step %c1_i32_16 iter_args(%arg21 = %369, %arg22 = %364, %arg23 = %370, %arg24 = %arg16) -> (i32, i32, i32, i32)  : i32 {
                        %380 = arith.muli %arg21, %c-766435501_i32 : i32
                        %381 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %382 = ascsimt.math_binary %381, %360, "mulhi" : ui32, ui32 -> ui32
                        %383 = "ascsimt.bitcast"(%382) : (ui32) -> i32
                        %384 = arith.muli %arg23, %c-845247145_i32 : i32
                        %385 = "ascsimt.bitcast"(%arg23) : (i32) -> ui32
                        %386 = ascsimt.math_binary %385, %366, "mulhi" : ui32, ui32 -> ui32
                        %387 = "ascsimt.bitcast"(%386) : (ui32) -> i32
                        %388 = arith.muli %arg20, %c-1640531527_i32_18 : i32
                        %389 = arith.addi %arg14, %388 : i32
                        %390 = arith.muli %arg20, %c-1150833019_i32_17 : i32
                        %391 = arith.addi %arg15, %390 : i32
                        %392 = arith.xori %387, %arg22 : i32
                        %393 = arith.xori %392, %389 : i32
                        %394 = arith.xori %383, %arg24 : i32
                        %395 = arith.xori %394, %391 : i32
                        scf.yield %393, %384, %395, %380 : i32, i32, i32, i32
                      }
                      %372 = arith.muli %arg19, %c4_i64_15 : i64
                      %373 = arith.index_cast %372 : i64 to index
                      ascsimt.store %arg17[%373], %371#0 : memref<8192xi32, 1>, i32
                      %374 = arith.addi %372, %c1_i64_14 : i64
                      %375 = arith.index_cast %374 : i64 to index
                      ascsimt.store %arg17[%375], %371#1 : memref<8192xi32, 1>, i32
                      %376 = arith.addi %372, %c2_i64_13 : i64
                      %377 = arith.index_cast %376 : i64 to index
                      ascsimt.store %arg17[%377], %371#2 : memref<8192xi32, 1>, i32
                      %378 = arith.addi %372, %c3_i64_12 : i64
                      %379 = arith.index_cast %378 : i64 to index
                      ascsimt.store %arg17[%379], %371#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%245, %255, %270, %271, %233, %243, %alloc_9, %272 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
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
                    %c16384_i32 = arith.constant 16384 : i32
                    %287 = ascsimt.thread_idx {ascsimt.scope = 32 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %288 = arith.extsi %287 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg17 = %288 to %arg16 step %c256_i64_11  : i64 {
                      %289 = arith.trunci %arg17 : i64 to i32
                      %290 = arith.addi %arg9, %289 : i32
                      %291 = arith.cmpi ult, %290, %arg9 : i32
                      %292 = arith.extui %291 : i1 to i32
                      %293 = arith.addi %arg10, %292 : i32
                      %294 = arith.divui %290, %c16384_i32 : i32
                      %295 = arith.remui %293, %c16384_i32 : i32
                      %296 = arith.muli %295, %c262144_i32 : i32
                      %297 = arith.addi %294, %296 : i32
                      %298 = arith.divui %293, %c16384_i32 : i32
                      %299 = arith.addi %arg11, %297 : i32
                      %300 = arith.addi %arg12, %298 : i32
                      %301 = arith.cmpi ult, %299, %arg11 : i32
                      %302 = arith.extui %301 : i1 to i32
                      %303 = arith.addi %300, %302 : i32
                      %304 = arith.remui %290, %c16384_i32 : i32
                      %305:4 = scf.for %arg18 = %c0_i32_17 to %c10_i32 step %c1_i32_16 iter_args(%arg19 = %c0_i32_17, %arg20 = %304, %arg21 = %299, %arg22 = %303) -> (i32, i32, i32, i32)  : i32 {
                        %314 = arith.muli %arg19, %c-766435501_i32 : i32
                        %315 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %316 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %317 = ascsimt.math_binary %315, %316, "mulhi" : ui32, ui32 -> ui32
                        %318 = "ascsimt.bitcast"(%317) : (ui32) -> i32
                        %319 = arith.muli %arg21, %c-845247145_i32 : i32
                        %320 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %321 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %322 = ascsimt.math_binary %320, %321, "mulhi" : ui32, ui32 -> ui32
                        %323 = "ascsimt.bitcast"(%322) : (ui32) -> i32
                        %324 = arith.muli %arg18, %c-1640531527_i32_19 : i32
                        %325 = arith.addi %arg13, %324 : i32
                        %326 = arith.muli %arg18, %c-1150833019_i32_18 : i32
                        %327 = arith.addi %arg14, %326 : i32
                        %328 = arith.xori %323, %arg20 : i32
                        %329 = arith.xori %328, %325 : i32
                        %330 = arith.xori %318, %arg22 : i32
                        %331 = arith.xori %330, %327 : i32
                        scf.yield %329, %319, %331, %314 : i32, i32, i32, i32
                      }
                      %306 = arith.muli %arg17, %c4_i64_15 : i64
                      %307 = arith.index_cast %306 : i64 to index
                      ascsimt.store %arg15[%307], %305#0 : memref<8192xi32, 1>, i32
                      %308 = arith.addi %306, %c1_i64_14 : i64
                      %309 = arith.index_cast %308 : i64 to index
                      ascsimt.store %arg15[%309], %305#1 : memref<8192xi32, 1>, i32
                      %310 = arith.addi %306, %c2_i64_13 : i64
                      %311 = arith.index_cast %310 : i64 to index
                      ascsimt.store %arg15[%311], %305#2 : memref<8192xi32, 1>, i32
                      %312 = arith.addi %306, %c3_i64_12 : i64
                      %313 = arith.index_cast %312 : i64 to index
                      ascsimt.store %arg15[%313], %305#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %278 = arith.trunci %32 : i64 to i32
                %279 = arith.divsi %278, %c2_i32 : i32
                %280 = arith.trunci %279 : i32 to i8
                %281 = arith.trunci %224 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %282 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%280, %alloc_7, %134, %282, %alloc_10, %281 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, memref<2048xui16, 1>, i32) {
                ^bb0(%arg9: i8, %arg10: memref<2048xui8, 1>, %arg11: memref<32768xbf16, 1>, %arg12: memref<?xui16, 1>, %arg13: memref<2048xui16, 1>, %arg14: i32):
                  %c16_11 = arith.constant 16 : index
                  %c4_i32_12 = arith.constant 4 : i32
                  %c0_i32_13 = arith.constant 0 : i32
                  %c3_i32 = arith.constant 3 : i32
                  %c2_i32_14 = arith.constant 2 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c1_i32_15 = arith.constant 1 : i32
                  %c0_16 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %287 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %288 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %289 = ascvec.duplicate_scalar %c32767_i16, %287 : i16 -> <ui16>
                  %290 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %291 = ascvec.shr %290, 4, %287 : <ui16> -> <ui16>
                  %292 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %293 = ascvec.duplicate_scalar %c1_i8, %288 : i8 -> <ui8>
                  %294 = ascvec.and %292, %293, %288 : <ui8>, <ui8> -> <ui8>
                  %295 = ascvec.cmp_lt_scalar %292, %c16_i8, %288 : <ui8>, i8 -> !ascvec.mask
                  %296 = ascvec.cmp_lt_scalar %294, %arg9, %288 : <ui8>, i8 -> !ascvec.mask
                  %297 = ascvec.mask_and %295, %296, %288
                  ascvec.store_unalign_begin %arg10 : memref<2048xui8, 1>
                  %mask_17, %remaining_out_18 = ascvec.update_mask %c1_i32_15 elem_bits 16 : !ascvec.mask
                  %298 = ascvec.duplicate_scalar %c-1_i16, %mask_17 : i16 -> <ui16>
                  %299 = scf.for %arg15 = %c0_i32_13 to %arg14 step %c4_i32_12 iter_args(%arg16 = %298) -> (!ascvec.vreg<ui16>)  : i32 {
                    %300 = ascvec.duplicate_scalar %c-1_i16, %287 : i16 -> <ui16>
                    %301 = ascvec.duplicate_scalar %c0_i16, %287 : i16 -> <ui16>
                    %302 = arith.muli %arg15, %c256_i32 : i32
                    %303 = arith.index_cast %302 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg11[%303] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %304 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %305 = ascvec.and %304, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %306 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %307 = ascvec.and %306, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %308 = ascvec.max %305, %307, %287 : <ui16>
                    %309 = ascvec.reduce_max_datablock %308, %287 : <ui16>
                    %310 = ascvec.shr %309, 7, %287 : <ui16> -> <ui16>
                    %311 = ascvec.cmp_ge_scalar %310, %c255_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %312 = ascvec.max_scalar %310, %c15_i16, %287 : <ui16>, i16
                    %313 = ascvec.add_scalar %312, %c-15_i16, %287 : <ui16>, i16
                    %314 = ascvec.duplicate_scalar %c255_i16, %311 merge %313 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %315 = ascvec.duplicate_scalar %c254_i16, %287 : i16 -> <ui16>
                    %316 = ascvec.sub %315, %314, %287 : <ui16>
                    %317 = ascvec.cmp_ne_scalar %314, %c0_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %318 = ascvec.shl %316, 7, %317 : <ui16> -> <ui16>
                    %319 = ascvec.max %301, %314, %mask : <ui16>
                    %320 = ascvec.gather_reg %318, %291 : <ui16>, <ui16> -> <ui16>
                    %321 = ascvec.reinterpret %320 : <ui16> -> <bf16>
                    %322 = ascvec.mul %result0, %321, %287 : <bf16>
                    %323 = ascvec.mul %result1, %321, %287 : <bf16>
                    ascvec.storealign_intlv %arg11[%303], %322, %323 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %324 = ascvec.reinterpret %322 : <bf16> -> <ui16>
                    %325 = ascvec.and %324, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %326 = ascvec.reinterpret %323 : <bf16> -> <ui16>
                    %327 = ascvec.and %326, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %328 = ascvec.min %325, %327, %287 : <ui16>
                    %329 = ascvec.min %300, %328, %287 : <ui16>
                    %330 = arith.muli %arg15, %c16_i32 : i32
                    %331 = arith.index_cast %330 : i32 to index
                    ascvec.storealign %arg12[%331], %314, %mask : memref<?xui16, 1>, index, <ui16>
                    %332 = ascvec.reinterpret %314 : <ui16> -> <ui8>
                    %333 = ascvec.squeeze_store_reg %332, %297 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_16], %333 : memref<2048xui8, 1>, <ui8>
                    %334 = arith.addi %arg15, %c1_i32_15 : i32
                    %335 = arith.muli %334, %c256_i32 : i32
                    %336 = arith.index_cast %335 : i32 to index
                    %result0_20, %result1_21 = ascvec.loadalign_deintlv %arg11[%336] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %337 = ascvec.reinterpret %result0_20 : <bf16> -> <ui16>
                    %338 = ascvec.and %337, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %339 = ascvec.reinterpret %result1_21 : <bf16> -> <ui16>
                    %340 = ascvec.and %339, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %341 = ascvec.max %338, %340, %287 : <ui16>
                    %342 = ascvec.reduce_max_datablock %341, %287 : <ui16>
                    %343 = ascvec.shr %342, 7, %287 : <ui16> -> <ui16>
                    %344 = ascvec.cmp_ge_scalar %343, %c255_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %345 = ascvec.max_scalar %343, %c15_i16, %287 : <ui16>, i16
                    %346 = ascvec.add_scalar %345, %c-15_i16, %287 : <ui16>, i16
                    %347 = ascvec.duplicate_scalar %c255_i16, %344 merge %346 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %348 = ascvec.sub %315, %347, %287 : <ui16>
                    %349 = ascvec.cmp_ne_scalar %347, %c0_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %350 = ascvec.shl %348, 7, %349 : <ui16> -> <ui16>
                    %351 = ascvec.max %319, %347, %mask : <ui16>
                    %352 = ascvec.gather_reg %350, %291 : <ui16>, <ui16> -> <ui16>
                    %353 = ascvec.reinterpret %352 : <ui16> -> <bf16>
                    %354 = ascvec.mul %result0_20, %353, %287 : <bf16>
                    %355 = ascvec.mul %result1_21, %353, %287 : <bf16>
                    ascvec.storealign_intlv %arg11[%336], %354, %355 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %356 = ascvec.reinterpret %354 : <bf16> -> <ui16>
                    %357 = ascvec.and %356, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %358 = ascvec.reinterpret %355 : <bf16> -> <ui16>
                    %359 = ascvec.and %358, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %360 = ascvec.min %357, %359, %287 : <ui16>
                    %361 = ascvec.min %329, %360, %287 : <ui16>
                    %362 = arith.muli %334, %c16_i32 : i32
                    %363 = arith.index_cast %362 : i32 to index
                    ascvec.storealign %arg12[%363], %347, %mask : memref<?xui16, 1>, index, <ui16>
                    %364 = ascvec.reinterpret %347 : <ui16> -> <ui8>
                    %365 = ascvec.squeeze_store_reg %364, %297 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_16], %365 : memref<2048xui8, 1>, <ui8>
                    %366 = arith.addi %arg15, %c2_i32_14 : i32
                    %367 = arith.muli %366, %c256_i32 : i32
                    %368 = arith.index_cast %367 : i32 to index
                    %result0_22, %result1_23 = ascvec.loadalign_deintlv %arg11[%368] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %369 = ascvec.reinterpret %result0_22 : <bf16> -> <ui16>
                    %370 = ascvec.and %369, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %371 = ascvec.reinterpret %result1_23 : <bf16> -> <ui16>
                    %372 = ascvec.and %371, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %373 = ascvec.max %370, %372, %287 : <ui16>
                    %374 = ascvec.reduce_max_datablock %373, %287 : <ui16>
                    %375 = ascvec.shr %374, 7, %287 : <ui16> -> <ui16>
                    %376 = ascvec.cmp_ge_scalar %375, %c255_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %377 = ascvec.max_scalar %375, %c15_i16, %287 : <ui16>, i16
                    %378 = ascvec.add_scalar %377, %c-15_i16, %287 : <ui16>, i16
                    %379 = ascvec.duplicate_scalar %c255_i16, %376 merge %378 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %380 = ascvec.sub %315, %379, %287 : <ui16>
                    %381 = ascvec.cmp_ne_scalar %379, %c0_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %382 = ascvec.shl %380, 7, %381 : <ui16> -> <ui16>
                    %383 = ascvec.max %351, %379, %mask : <ui16>
                    %384 = ascvec.gather_reg %382, %291 : <ui16>, <ui16> -> <ui16>
                    %385 = ascvec.reinterpret %384 : <ui16> -> <bf16>
                    %386 = ascvec.mul %result0_22, %385, %287 : <bf16>
                    %387 = ascvec.mul %result1_23, %385, %287 : <bf16>
                    ascvec.storealign_intlv %arg11[%368], %386, %387 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %388 = ascvec.reinterpret %386 : <bf16> -> <ui16>
                    %389 = ascvec.and %388, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %390 = ascvec.reinterpret %387 : <bf16> -> <ui16>
                    %391 = ascvec.and %390, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %392 = ascvec.min %389, %391, %287 : <ui16>
                    %393 = ascvec.min %361, %392, %287 : <ui16>
                    %394 = arith.muli %366, %c16_i32 : i32
                    %395 = arith.index_cast %394 : i32 to index
                    ascvec.storealign %arg12[%395], %379, %mask : memref<?xui16, 1>, index, <ui16>
                    %396 = ascvec.reinterpret %379 : <ui16> -> <ui8>
                    %397 = ascvec.squeeze_store_reg %396, %297 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_16], %397 : memref<2048xui8, 1>, <ui8>
                    %398 = arith.addi %arg15, %c3_i32 : i32
                    %399 = arith.muli %398, %c256_i32 : i32
                    %400 = arith.index_cast %399 : i32 to index
                    %result0_24, %result1_25 = ascvec.loadalign_deintlv %arg11[%400] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %401 = ascvec.reinterpret %result0_24 : <bf16> -> <ui16>
                    %402 = ascvec.and %401, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %403 = ascvec.reinterpret %result1_25 : <bf16> -> <ui16>
                    %404 = ascvec.and %403, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %405 = ascvec.max %402, %404, %287 : <ui16>
                    %406 = ascvec.reduce_max_datablock %405, %287 : <ui16>
                    %407 = ascvec.shr %406, 7, %287 : <ui16> -> <ui16>
                    %408 = ascvec.cmp_ge_scalar %407, %c255_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %409 = ascvec.max_scalar %407, %c15_i16, %287 : <ui16>, i16
                    %410 = ascvec.add_scalar %409, %c-15_i16, %287 : <ui16>, i16
                    %411 = ascvec.duplicate_scalar %c255_i16, %408 merge %410 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %412 = ascvec.sub %315, %411, %287 : <ui16>
                    %413 = ascvec.cmp_ne_scalar %411, %c0_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %414 = ascvec.shl %412, 7, %413 : <ui16> -> <ui16>
                    %415 = ascvec.max %383, %411, %mask : <ui16>
                    %416 = ascvec.gather_reg %414, %291 : <ui16>, <ui16> -> <ui16>
                    %417 = ascvec.reinterpret %416 : <ui16> -> <bf16>
                    %418 = ascvec.mul %result0_24, %417, %287 : <bf16>
                    %419 = ascvec.mul %result1_25, %417, %287 : <bf16>
                    ascvec.storealign_intlv %arg11[%400], %418, %419 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %420 = ascvec.reinterpret %418 : <bf16> -> <ui16>
                    %421 = ascvec.and %420, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %422 = ascvec.reinterpret %419 : <bf16> -> <ui16>
                    %423 = ascvec.and %422, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %424 = ascvec.min %421, %423, %287 : <ui16>
                    %425 = ascvec.min %393, %424, %287 : <ui16>
                    %426 = arith.muli %398, %c16_i32 : i32
                    %427 = arith.index_cast %426 : i32 to index
                    ascvec.storealign %arg12[%427], %411, %mask : memref<?xui16, 1>, index, <ui16>
                    %428 = ascvec.reinterpret %411 : <ui16> -> <ui8>
                    %429 = ascvec.squeeze_store_reg %428, %297 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_16], %429 : memref<2048xui8, 1>, <ui8>
                    %430 = ascvec.reduce_min %425, %287 : <ui16>
                    %431 = ascvec.reduce_max %415, %mask : <ui16>
                    %432 = ascvec.cmp_eq_scalar %431, %c255_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %433 = ascvec.duplicate_scalar %c0_i16, %432 merge %430 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.storealign_1st %arg13[%331], %433 : memref<2048xui16, 1>, index, <ui16>
                    %434 = ascvec.min %arg16, %433, %mask_17 : <ui16>
                    scf.yield %434 : !ascvec.vreg<ui16>
                  }
                  %c0_19 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg15 = %c0_19 to %c1 step %c1 {
                    ascvec.store_unalign_post %arg10[%c0_16] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg13[%c16_11], %299 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %283 = memref.load %alloc_10[%c16] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %284 = cannir.scalar_storage_cast %283 : ui16 -> i16
                %285 = arith.extui %284 : i16 to i64
                %286 = arith.cmpi sge, %285, %c14464_i64 : i64
                scf.if %286 {
                  ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %287 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                  %288 = ascscalar.bitcast_memref %134 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%134, %287, %288, %alloc_6, %281 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_11 = arith.constant 1 : i32
                    %c0_i32_12 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c128_i32 = arith.constant 128 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c31_i16 = arith.constant 31 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %289 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %290 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %291 = ascvec.duplicate_scalar %c255_i16, %289 : i16 -> <ui16>
                    %292 = ascvec.duplicate_scalar %c-128_i8, %290 : i8 -> <ui8>
                    %293 = ascvec.duplicate_scalar %c31_i16, %289 : i16 -> <ui16>
                    scf.for %arg14 = %c0_i32_12 to %arg13 step %c1_i32_11  : i32 {
                      %294 = arith.muli %arg14, %c256_i32 : i32
                      %295 = arith.index_cast %294 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg9[%295] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %296 = arith.muli %arg14, %c128_i32 : i32
                      %297 = arith.index_cast %296 : i32 to index
                      %298 = ascvec.loadalign %arg10[%297] : memref<?xui8, 1>, index -> <ui8>
                      %299 = ascvec.unpack %298 {part = "lower"} : <ui8> -> <ui16>
                      %300 = ascvec.and %299, %293, %289 : <ui16>, <ui16> -> <ui16>
                      %301 = ascvec.shr %299, 3, %289 : <ui16> -> <ui16>
                      %302 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %303 = ascvec.add %302, %300, %289 : <ui16>
                      %304 = ascvec.shr %303, 5, %289 : <ui16> -> <ui16>
                      %305 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %306 = ascvec.add %305, %301, %289 : <ui16>
                      %307 = ascvec.shr %306, 5, %289 : <ui16> -> <ui16>
                      %308 = ascvec.and %304, %291, %289 : <ui16>, <ui16> -> <ui16>
                      %309 = ascvec.shl %307, 8, %289 : <ui16> -> <ui16>
                      %310 = ascvec.or %308, %309, %289 : <ui16>, <ui16> -> <ui16>
                      %311 = ascvec.reinterpret %310 : <ui16> -> <ui8>
                      %312 = ascvec.add_scalar %311, %c64_i8, %290 : <ui8>, i8
                      %313 = ascvec.min_scalar %312, %c123_i8, %290 : <ui8>, i8
                      %314 = arith.muli %arg14, %c512_i32 : i32
                      %315 = arith.index_cast %314 : i32 to index
                      %result0_13, %result1_14 = ascvec.loadalign_deintlv %arg11[%315] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %316 = ascvec.and %result1_14, %292, %290 : <ui8>, <ui8> -> <ui8>
                      %317 = ascvec.or %313, %316, %290 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg12[%295], %317, %290 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
                } else {
                  scf.for %arg9 = %c0_i64 to %224 step %c16_i64  : i64 {
                    %287 = arith.addi %arg9, %c16_i64 : i64
                    %288 = arith.cmpi sgt, %287, %224 : i64
                    %289 = arith.select %288, %224, %287 : i64
                    %290 = arith.muli %arg9, %c16_i64 : i64
                    ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %291 = arith.index_cast %290 : i64 to index
                    %292 = memref.load %alloc_10[%291] : memref<2048xui16, 1>
                    ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %293 = cannir.scalar_storage_cast %292 : ui16 -> i16
                    %294 = arith.extui %293 : i16 to i64
                    %295 = arith.addi %arg9, %c4_i64 : i64
                    %296 = scf.for %arg10 = %295 to %289 step %c4_i64 iter_args(%arg11 = %294) -> (i64)  : i64 {
                      %298 = arith.muli %arg10, %c16_i64 : i64
                      ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %299 = arith.index_cast %298 : i64 to index
                      %300 = memref.load %alloc_10[%299] : memref<2048xui16, 1>
                      ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %301 = cannir.scalar_storage_cast %300 : ui16 -> i16
                      %302 = arith.extui %301 : i16 to i64
                      %303 = arith.cmpi slt, %302, %arg11 : i64
                      %304 = arith.select %303, %302, %arg11 : i64
                      scf.yield %304 : i64
                    }
                    %297 = arith.cmpi sge, %296, %c14464_i64 : i64
                    scf.if %297 {
                      %298 = arith.trunci %arg9 : i64 to i32
                      %299 = arith.trunci %289 : i64 to i32
                      ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %300 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                      %301 = ascscalar.bitcast_memref %134 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                      ascvec.vf(%134, %300, %301, %alloc_6, %298, %299 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                      ^bb0(%arg10: memref<32768xbf16, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<?xui8, 1>, %arg13: memref<32768xui8, 1>, %arg14: i32, %arg15: i32):
                        %c512_i32 = arith.constant 512 : i32
                        %c1_i32_11 = arith.constant 1 : i32
                        %c123_i8 = arith.constant 123 : i8
                        %c64_i8 = arith.constant 64 : i8
                        %c128_i32 = arith.constant 128 : i32
                        %c256_i32 = arith.constant 256 : i32
                        %c31_i16 = arith.constant 31 : i16
                        %c-128_i8 = arith.constant -128 : i8
                        %c255_i16 = arith.constant 255 : i16
                        %302 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                        %303 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                        %304 = ascvec.duplicate_scalar %c255_i16, %302 : i16 -> <ui16>
                        %305 = ascvec.duplicate_scalar %c-128_i8, %303 : i8 -> <ui8>
                        %306 = ascvec.duplicate_scalar %c31_i16, %302 : i16 -> <ui16>
                        scf.for %arg16 = %arg14 to %arg15 step %c1_i32_11  : i32 {
                          %307 = arith.muli %arg16, %c256_i32 : i32
                          %308 = arith.index_cast %307 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg10[%308] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                          %309 = arith.muli %arg16, %c128_i32 : i32
                          %310 = arith.index_cast %309 : i32 to index
                          %311 = ascvec.loadalign %arg11[%310] : memref<?xui8, 1>, index -> <ui8>
                          %312 = ascvec.unpack %311 {part = "lower"} : <ui8> -> <ui16>
                          %313 = ascvec.and %312, %306, %302 : <ui16>, <ui16> -> <ui16>
                          %314 = ascvec.shr %312, 3, %302 : <ui16> -> <ui16>
                          %315 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                          %316 = ascvec.add %315, %313, %302 : <ui16>
                          %317 = ascvec.shr %316, 5, %302 : <ui16> -> <ui16>
                          %318 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                          %319 = ascvec.add %318, %314, %302 : <ui16>
                          %320 = ascvec.shr %319, 5, %302 : <ui16> -> <ui16>
                          %321 = ascvec.and %317, %304, %302 : <ui16>, <ui16> -> <ui16>
                          %322 = ascvec.shl %320, 8, %302 : <ui16> -> <ui16>
                          %323 = ascvec.or %321, %322, %302 : <ui16>, <ui16> -> <ui16>
                          %324 = ascvec.reinterpret %323 : <ui16> -> <ui8>
                          %325 = ascvec.add_scalar %324, %c64_i8, %303 : <ui8>, i8
                          %326 = ascvec.min_scalar %325, %c123_i8, %303 : <ui8>, i8
                          %327 = arith.muli %arg16, %c512_i32 : i32
                          %328 = arith.index_cast %327 : i32 to index
                          %result0_12, %result1_13 = ascvec.loadalign_deintlv %arg12[%328] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                          %329 = ascvec.and %result1_13, %305, %303 : <ui8>, <ui8> -> <ui8>
                          %330 = ascvec.or %326, %329, %303 : <ui8>, <ui8> -> <ui8>
                          ascvec.storealign %arg13[%308], %330, %303 : memref<32768xui8, 1>, index, <ui8>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
                    } else {
                      %298 = arith.cmpi sge, %296, %c13568_i64 : i64
                      scf.if %298 {
                        %299 = arith.trunci %arg9 : i64 to i32
                        %300 = arith.trunci %289 : i64 to i32
                        ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %301 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                        %302 = ascscalar.bitcast_memref %134 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%134, %301, %302, %alloc_6, %299, %300 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg10: memref<32768xbf16, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<?xui8, 1>, %arg13: memref<32768xui8, 1>, %arg14: i32, %arg15: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_11 = arith.constant 1 : i32
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
                          %303 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %304 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %305 = ascvec.duplicate_scalar %c255_i16, %303 : i16 -> <ui16>
                          %306 = ascvec.duplicate_scalar %c-128_i8, %304 : i8 -> <ui8>
                          %307 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %308 = ascvec.shr %307, 1, %304 : <ui8> -> <ui8>
                          %309 = ascvec.shr %308, 2, %304 : <ui8> -> <ui8>
                          %310 = ascvec.shl %309, 2, %304 : <ui8> -> <ui8>
                          %311 = ascvec.duplicate_scalar %c1_i8, %304 : i8 -> <ui8>
                          %312 = ascvec.and %307, %311, %304 : <ui8>, <ui8> -> <ui8>
                          %313 = ascvec.add_scalar %308, %c3_i8, %304 : <ui8>, i8
                          %314 = ascvec.add %313, %312, %304 : <ui8>
                          %315 = ascvec.duplicate_scalar %c3_i8, %304 : i8 -> <ui8>
                          %316 = ascvec.and %314, %315, %304 : <ui8>, <ui8> -> <ui8>
                          %317 = ascvec.or %310, %316, %304 : <ui8>, <ui8> -> <ui8>
                          %318 = ascvec.duplicate_scalar %c4095_i16, %303 : i16 -> <ui16>
                          %319 = ascvec.duplicate_scalar %c32767_i16, %303 : i16 -> <ui16>
                          %320 = ascvec.duplicate_scalar %cst, %303 : bf16 -> <bf16>
                          scf.for %arg16 = %arg14 to %arg15 step %c1_i32_11  : i32 {
                            %321 = arith.muli %arg16, %c256_i32 : i32
                            %322 = arith.index_cast %321 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg10[%322] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %323 = arith.muli %arg16, %c128_i32 : i32
                            %324 = arith.index_cast %323 : i32 to index
                            %325 = ascvec.loadalign %arg11[%324] : memref<?xui8, 1>, index -> <ui8>
                            %326 = ascvec.gather_reg %325, %317 : <ui8>, <ui8> -> <ui8>
                            %327 = ascvec.reinterpret %326 : <ui8> -> <ui16>
                            %328 = ascvec.shr %327, 1, %303 : <ui16> -> <ui16>
                            %329 = ascvec.and %328, %318, %303 : <ui16>, <ui16> -> <ui16>
                            %330 = ascvec.shr %327, 4, %303 : <ui16> -> <ui16>
                            %331 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %332 = ascvec.and %331, %319, %303 : <ui16>, <ui16> -> <ui16>
                            %333 = ascvec.cmp_lt_scalar %332, %c14464_i16, %303 : <ui16>, i16 -> !ascvec.mask
                            %334 = ascvec.shr %329, 7, %303 : <ui16> -> <ui16>
                            %335 = ascvec.add %332, %334, %303 : <ui16>
                            %336 = ascvec.shr %335, 5, %303 : <ui16> -> <ui16>
                            %337 = ascvec.reinterpret %332 : <ui16> -> <bf16>
                            %338 = ascvec.mul %337, %320, %333 : <bf16>
                            %339 = ascvec.cast %338, %303 {rounding = 4 : i32} : <bf16> -> <f16>
                            %340 = ascvec.cast %339, %303 {rounding = 4 : i32} : <f16> -> <i16>
                            %341 = ascvec.reinterpret %340 : <i16> -> <ui16>
                            %342 = ascvec.add %341, %329, %303 : <ui16>
                            %343 = ascvec.shr %342, 12, %303 : <ui16> -> <ui16>
                            %344 = ascvec.add_scalar %343, %c448_i16, %303 : <ui16>, i16
                            %345 = ascvec.select %344, %336, %333 : <ui16>
                            %346 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %347 = ascvec.and %346, %319, %303 : <ui16>, <ui16> -> <ui16>
                            %348 = ascvec.cmp_lt_scalar %347, %c14464_i16, %303 : <ui16>, i16 -> !ascvec.mask
                            %349 = ascvec.shr %330, 7, %303 : <ui16> -> <ui16>
                            %350 = ascvec.add %347, %349, %303 : <ui16>
                            %351 = ascvec.shr %350, 5, %303 : <ui16> -> <ui16>
                            %352 = ascvec.reinterpret %347 : <ui16> -> <bf16>
                            %353 = ascvec.mul %352, %320, %348 : <bf16>
                            %354 = ascvec.cast %353, %303 {rounding = 4 : i32} : <bf16> -> <f16>
                            %355 = ascvec.cast %354, %303 {rounding = 4 : i32} : <f16> -> <i16>
                            %356 = ascvec.reinterpret %355 : <i16> -> <ui16>
                            %357 = ascvec.add %356, %330, %303 : <ui16>
                            %358 = ascvec.shr %357, 12, %303 : <ui16> -> <ui16>
                            %359 = ascvec.add_scalar %358, %c448_i16, %303 : <ui16>, i16
                            %360 = ascvec.select %359, %351, %348 : <ui16>
                            %361 = ascvec.and %345, %305, %303 : <ui16>, <ui16> -> <ui16>
                            %362 = ascvec.shl %360, 8, %303 : <ui16> -> <ui16>
                            %363 = ascvec.or %361, %362, %303 : <ui16>, <ui16> -> <ui16>
                            %364 = ascvec.reinterpret %363 : <ui16> -> <ui8>
                            %365 = ascvec.add_scalar %364, %c64_i8, %304 : <ui8>, i8
                            %366 = ascvec.min_scalar %365, %c123_i8, %304 : <ui8>, i8
                            %367 = arith.muli %arg16, %c512_i32 : i32
                            %368 = arith.index_cast %367 : i32 to index
                            %result0_12, %result1_13 = ascvec.loadalign_deintlv %arg12[%368] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %369 = ascvec.and %result1_13, %306, %304 : <ui8>, <ui8> -> <ui8>
                            %370 = ascvec.or %366, %369, %304 : <ui8>, <ui8> -> <ui8>
                            ascvec.storealign %arg13[%322], %370, %304 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
                      } else {
                        %299 = arith.trunci %arg9 : i64 to i32
                        %300 = arith.trunci %289 : i64 to i32
                        ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %301 = ascscalar.bitcast_memref %134 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%134, %alloc_9, %301, %282, %alloc_6, %299, %300 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg10: memref<32768xbf16, 1>, %arg11: memref<8192xi32, 1>, %arg12: memref<?xui8, 1>, %arg13: memref<?xui16, 1>, %arg14: memref<32768xui8, 1>, %arg15: i32, %arg16: i32):
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
                          %c4_i32_11 = arith.constant 4 : i32
                          %c2_i32_12 = arith.constant 2 : i32
                          %c1_i32_13 = arith.constant 1 : i32
                          %c7_i32 = arith.constant 7 : i32
                          %c32_i32 = arith.constant 32 : i32
                          %c0_i32_14 = arith.constant 0 : i32
                          %c32767_i16 = arith.constant 32767 : i16
                          %c0_15 = arith.constant 0 : index
                          %c256_i32 = arith.constant 256 : i32
                          %c3_i8 = arith.constant 3 : i8
                          %c-128_i8 = arith.constant -128 : i8
                          %c0_i8 = arith.constant 0 : i8
                          %c8_i32 = arith.constant 8 : i32
                          %302 = ascvec.create_mask : !ascvec.mask
                          %303 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %304 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                          %305 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %306 = ascvec.duplicate_scalar %c-128_i8, %304 : i8 -> <ui8>
                          %307 = ascvec.shr %305, 5, %304 : <ui8> -> <ui8>
                          %308 = ascvec.duplicate_scalar %c3_i8, %304 : i8 -> <ui8>
                          %309 = ascvec.and %305, %308, %304 : <ui8>, <ui8> -> <ui8>
                          %310 = ascvec.shl %309, 6, %304 : <ui8> -> <ui8>
                          %311 = ascvec.shr %305, 2, %304 : <ui8> -> <ui8>
                          %312 = ascvec.or %310, %311, %304 : <ui8>, <ui8> -> <ui8>
                          scf.for %arg17 = %arg15 to %arg16 step %c1_i32_13  : i32 {
                            %313 = arith.muli %arg17, %c256_i32 : i32
                            %314 = arith.index_cast %313 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg10[%314] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %315 = ascvec.duplicate_scalar %c32767_i16, %303 : i16 -> <ui16>
                            %316 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %317 = ascvec.and %316, %315, %303 : <ui16>, <ui16> -> <ui16>
                            %318 = ascvec.reinterpret %317 : <ui16> -> <bf16>
                            %319 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %320 = ascvec.and %319, %315, %303 : <ui16>, <ui16> -> <ui16>
                            %321 = ascvec.reinterpret %320 : <ui16> -> <bf16>
                            %322 = ascvec.cast %318, %303 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %323 = ascvec.cast %318, %303 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %324 = ascvec.cast %321, %303 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %325 = ascvec.cast %321, %303 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %326 = ascvec.arange %c0_i32_14 : i32 -> <ui32>
                            %327 = ascvec.shl %326, 2, %302 : <ui32> -> <ui32>
                            %328 = ascvec.add_scalar %327, %c0_i32_14, %302 : <ui32>, i32
                            %329 = ascvec.shr %328, 3, %302 : <ui32> -> <ui32>
                            %330 = arith.muli %arg17, %c32_i32 : i32
                            %331 = ascvec.add_scalar %329, %330, %302 : <ui32>, i32
                            %332 = ascvec.gather %arg11[%c0_15], %331, %302 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %333 = ascvec.reinterpret %332 : <i32> -> <ui32>
                            %334 = ascvec.duplicate_scalar %c7_i32, %302 : i32 -> <ui32>
                            %335 = ascvec.and %328, %334, %302 : <ui32>, <ui32> -> <ui32>
                            %336 = ascvec.shl %333, 11, %302 : <ui32> -> <ui32>
                            %337 = ascvec.shr %333, 21, %302 : <ui32> -> <ui32>
                            %338 = ascvec.or %336, %337, %302 : <ui32>, <ui32> -> <ui32>
                            %339 = ascvec.shr %338, 3, %302 : <ui32> -> <ui32>
                            %340 = ascvec.shl %338, 29, %302 : <ui32> -> <ui32>
                            %341 = ascvec.or %339, %340, %302 : <ui32>, <ui32> -> <ui32>
                            %342 = ascvec.duplicate_scalar %c1_i32_13, %302 : i32 -> <ui32>
                            %343 = ascvec.and %335, %342, %302 : <ui32>, <ui32> -> <ui32>
                            %344 = ascvec.cmp_ne_scalar %343, %c0_i32_14, %302 : <ui32>, i32 -> !ascvec.mask
                            %345 = ascvec.select %341, %338, %344 : <ui32>
                            %346 = ascvec.shr %345, 8, %302 : <ui32> -> <ui32>
                            %347 = ascvec.shl %345, 24, %302 : <ui32> -> <ui32>
                            %348 = ascvec.or %346, %347, %302 : <ui32>, <ui32> -> <ui32>
                            %349 = ascvec.duplicate_scalar %c2_i32_12, %302 : i32 -> <ui32>
                            %350 = ascvec.and %335, %349, %302 : <ui32>, <ui32> -> <ui32>
                            %351 = ascvec.cmp_ne_scalar %350, %c0_i32_14, %302 : <ui32>, i32 -> !ascvec.mask
                            %352 = ascvec.select %348, %345, %351 : <ui32>
                            %353 = ascvec.shr %352, 16, %302 : <ui32> -> <ui32>
                            %354 = ascvec.shl %352, 16, %302 : <ui32> -> <ui32>
                            %355 = ascvec.or %353, %354, %302 : <ui32>, <ui32> -> <ui32>
                            %356 = ascvec.duplicate_scalar %c4_i32_11, %302 : i32 -> <ui32>
                            %357 = ascvec.and %335, %356, %302 : <ui32>, <ui32> -> <ui32>
                            %358 = ascvec.cmp_ne_scalar %357, %c0_i32_14, %302 : <ui32>, i32 -> !ascvec.mask
                            %359 = ascvec.select %355, %352, %358 : <ui32>
                            %360 = ascvec.duplicate_scalar %c65535_i32, %302 : i32 -> <ui32>
                            %361 = ascvec.and %359, %360, %302 : <ui32>, <ui32> -> <ui32>
                            %362 = ascvec.add_scalar %327, %c1_i32_13, %302 : <ui32>, i32
                            %363 = ascvec.shr %362, 3, %302 : <ui32> -> <ui32>
                            %364 = ascvec.add_scalar %363, %330, %302 : <ui32>, i32
                            %365 = ascvec.gather %arg11[%c0_15], %364, %302 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %366 = ascvec.reinterpret %365 : <i32> -> <ui32>
                            %367 = ascvec.and %362, %334, %302 : <ui32>, <ui32> -> <ui32>
                            %368 = ascvec.shl %366, 11, %302 : <ui32> -> <ui32>
                            %369 = ascvec.shr %366, 21, %302 : <ui32> -> <ui32>
                            %370 = ascvec.or %368, %369, %302 : <ui32>, <ui32> -> <ui32>
                            %371 = ascvec.shr %370, 3, %302 : <ui32> -> <ui32>
                            %372 = ascvec.shl %370, 29, %302 : <ui32> -> <ui32>
                            %373 = ascvec.or %371, %372, %302 : <ui32>, <ui32> -> <ui32>
                            %374 = ascvec.and %367, %342, %302 : <ui32>, <ui32> -> <ui32>
                            %375 = ascvec.cmp_ne_scalar %374, %c0_i32_14, %302 : <ui32>, i32 -> !ascvec.mask
                            %376 = ascvec.select %373, %370, %375 : <ui32>
                            %377 = ascvec.shr %376, 8, %302 : <ui32> -> <ui32>
                            %378 = ascvec.shl %376, 24, %302 : <ui32> -> <ui32>
                            %379 = ascvec.or %377, %378, %302 : <ui32>, <ui32> -> <ui32>
                            %380 = ascvec.and %367, %349, %302 : <ui32>, <ui32> -> <ui32>
                            %381 = ascvec.cmp_ne_scalar %380, %c0_i32_14, %302 : <ui32>, i32 -> !ascvec.mask
                            %382 = ascvec.select %379, %376, %381 : <ui32>
                            %383 = ascvec.shr %382, 16, %302 : <ui32> -> <ui32>
                            %384 = ascvec.shl %382, 16, %302 : <ui32> -> <ui32>
                            %385 = ascvec.or %383, %384, %302 : <ui32>, <ui32> -> <ui32>
                            %386 = ascvec.and %367, %356, %302 : <ui32>, <ui32> -> <ui32>
                            %387 = ascvec.cmp_ne_scalar %386, %c0_i32_14, %302 : <ui32>, i32 -> !ascvec.mask
                            %388 = ascvec.select %385, %382, %387 : <ui32>
                            %389 = ascvec.and %388, %360, %302 : <ui32>, <ui32> -> <ui32>
                            %390 = ascvec.add_scalar %327, %c2_i32_12, %302 : <ui32>, i32
                            %391 = ascvec.shr %390, 3, %302 : <ui32> -> <ui32>
                            %392 = ascvec.add_scalar %391, %330, %302 : <ui32>, i32
                            %393 = ascvec.gather %arg11[%c0_15], %392, %302 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %394 = ascvec.reinterpret %393 : <i32> -> <ui32>
                            %395 = ascvec.and %390, %334, %302 : <ui32>, <ui32> -> <ui32>
                            %396 = ascvec.shl %394, 11, %302 : <ui32> -> <ui32>
                            %397 = ascvec.shr %394, 21, %302 : <ui32> -> <ui32>
                            %398 = ascvec.or %396, %397, %302 : <ui32>, <ui32> -> <ui32>
                            %399 = ascvec.shr %398, 3, %302 : <ui32> -> <ui32>
                            %400 = ascvec.shl %398, 29, %302 : <ui32> -> <ui32>
                            %401 = ascvec.or %399, %400, %302 : <ui32>, <ui32> -> <ui32>
                            %402 = ascvec.and %395, %342, %302 : <ui32>, <ui32> -> <ui32>
                            %403 = ascvec.cmp_ne_scalar %402, %c0_i32_14, %302 : <ui32>, i32 -> !ascvec.mask
                            %404 = ascvec.select %401, %398, %403 : <ui32>
                            %405 = ascvec.shr %404, 8, %302 : <ui32> -> <ui32>
                            %406 = ascvec.shl %404, 24, %302 : <ui32> -> <ui32>
                            %407 = ascvec.or %405, %406, %302 : <ui32>, <ui32> -> <ui32>
                            %408 = ascvec.and %395, %349, %302 : <ui32>, <ui32> -> <ui32>
                            %409 = ascvec.cmp_ne_scalar %408, %c0_i32_14, %302 : <ui32>, i32 -> !ascvec.mask
                            %410 = ascvec.select %407, %404, %409 : <ui32>
                            %411 = ascvec.shr %410, 16, %302 : <ui32> -> <ui32>
                            %412 = ascvec.shl %410, 16, %302 : <ui32> -> <ui32>
                            %413 = ascvec.or %411, %412, %302 : <ui32>, <ui32> -> <ui32>
                            %414 = ascvec.and %395, %356, %302 : <ui32>, <ui32> -> <ui32>
                            %415 = ascvec.cmp_ne_scalar %414, %c0_i32_14, %302 : <ui32>, i32 -> !ascvec.mask
                            %416 = ascvec.select %413, %410, %415 : <ui32>
                            %417 = ascvec.and %416, %360, %302 : <ui32>, <ui32> -> <ui32>
                            %418 = ascvec.add_scalar %327, %c3_i32, %302 : <ui32>, i32
                            %419 = ascvec.shr %418, 3, %302 : <ui32> -> <ui32>
                            %420 = ascvec.add_scalar %419, %330, %302 : <ui32>, i32
                            %421 = ascvec.gather %arg11[%c0_15], %420, %302 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                            %422 = ascvec.reinterpret %421 : <i32> -> <ui32>
                            %423 = ascvec.and %418, %334, %302 : <ui32>, <ui32> -> <ui32>
                            %424 = ascvec.shl %422, 11, %302 : <ui32> -> <ui32>
                            %425 = ascvec.shr %422, 21, %302 : <ui32> -> <ui32>
                            %426 = ascvec.or %424, %425, %302 : <ui32>, <ui32> -> <ui32>
                            %427 = ascvec.shr %426, 3, %302 : <ui32> -> <ui32>
                            %428 = ascvec.shl %426, 29, %302 : <ui32> -> <ui32>
                            %429 = ascvec.or %427, %428, %302 : <ui32>, <ui32> -> <ui32>
                            %430 = ascvec.and %423, %342, %302 : <ui32>, <ui32> -> <ui32>
                            %431 = ascvec.cmp_ne_scalar %430, %c0_i32_14, %302 : <ui32>, i32 -> !ascvec.mask
                            %432 = ascvec.select %429, %426, %431 : <ui32>
                            %433 = ascvec.shr %432, 8, %302 : <ui32> -> <ui32>
                            %434 = ascvec.shl %432, 24, %302 : <ui32> -> <ui32>
                            %435 = ascvec.or %433, %434, %302 : <ui32>, <ui32> -> <ui32>
                            %436 = ascvec.and %423, %349, %302 : <ui32>, <ui32> -> <ui32>
                            %437 = ascvec.cmp_ne_scalar %436, %c0_i32_14, %302 : <ui32>, i32 -> !ascvec.mask
                            %438 = ascvec.select %435, %432, %437 : <ui32>
                            %439 = ascvec.shr %438, 16, %302 : <ui32> -> <ui32>
                            %440 = ascvec.shl %438, 16, %302 : <ui32> -> <ui32>
                            %441 = ascvec.or %439, %440, %302 : <ui32>, <ui32> -> <ui32>
                            %442 = ascvec.and %423, %356, %302 : <ui32>, <ui32> -> <ui32>
                            %443 = ascvec.cmp_ne_scalar %442, %c0_i32_14, %302 : <ui32>, i32 -> !ascvec.mask
                            %444 = ascvec.select %441, %438, %443 : <ui32>
                            %445 = ascvec.and %444, %360, %302 : <ui32>, <ui32> -> <ui32>
                            %446 = ascvec.reinterpret %322 : <f32> -> <ui32>
                            %447 = ascvec.reinterpret %446 : <ui32> -> <f32>
                            %448 = ascvec.max_scalar %447, %cst, %302 : <f32>, f32
                            %449 = ascvec.add %447, %448, %302 : <f32>
                            %450 = ascvec.reinterpret %449 : <f32> -> <ui32>
                            %451 = ascvec.shr %450, 5, %302 : <ui32> -> <ui32>
                            %452 = ascvec.add %451, %361, %302 : <ui32>
                            %453 = ascvec.reinterpret %324 : <f32> -> <ui32>
                            %454 = ascvec.reinterpret %453 : <ui32> -> <f32>
                            %455 = ascvec.max_scalar %454, %cst, %302 : <f32>, f32
                            %456 = ascvec.add %454, %455, %302 : <f32>
                            %457 = ascvec.reinterpret %456 : <f32> -> <ui32>
                            %458 = ascvec.shr %457, 5, %302 : <ui32> -> <ui32>
                            %459 = ascvec.add %458, %389, %302 : <ui32>
                            %460 = ascvec.reinterpret %323 : <f32> -> <ui32>
                            %461 = ascvec.reinterpret %460 : <ui32> -> <f32>
                            %462 = ascvec.max_scalar %461, %cst, %302 : <f32>, f32
                            %463 = ascvec.add %461, %462, %302 : <f32>
                            %464 = ascvec.reinterpret %463 : <f32> -> <ui32>
                            %465 = ascvec.shr %464, 5, %302 : <ui32> -> <ui32>
                            %466 = ascvec.add %465, %417, %302 : <ui32>
                            %467 = ascvec.reinterpret %325 : <f32> -> <ui32>
                            %468 = ascvec.reinterpret %467 : <ui32> -> <f32>
                            %469 = ascvec.max_scalar %468, %cst, %302 : <f32>, f32
                            %470 = ascvec.add %468, %469, %302 : <f32>
                            %471 = ascvec.reinterpret %470 : <f32> -> <ui32>
                            %472 = ascvec.shr %471, 5, %302 : <ui32> -> <ui32>
                            %473 = ascvec.add %472, %445, %302 : <ui32>
                            %474 = ascvec.reinterpret %452 : <ui32> -> <ui16>
                            %475 = ascvec.reinterpret %459 : <ui32> -> <ui16>
                            %result0_16, %result1_17 = ascvec.deintlv %474, %475 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %476 = ascvec.reinterpret %466 : <ui32> -> <ui16>
                            %477 = ascvec.reinterpret %473 : <ui32> -> <ui16>
                            %result0_18, %result1_19 = ascvec.deintlv %476, %477 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %478 = ascvec.reinterpret %result1_17 : <ui16> -> <ui8>
                            %479 = ascvec.reinterpret %result1_19 : <ui16> -> <ui8>
                            %result0_20, %result1_21 = ascvec.deintlv %478, %479 : <ui8>, <ui8> -> <ui8>, <ui8>
                            %480 = ascvec.gather_reg %result0_20, %312 : <ui8>, <ui8> -> <ui8>
                            %481 = ascvec.add_scalar %480, %c60_i8, %304 : <ui8>, i8
                            %482 = ascvec.cmp_ne_scalar %481, %c-1_i8, %304 : <ui8>, i8 -> !ascvec.mask
                            %483 = ascvec.min_scalar %481, %c123_i8, %482 : <ui8>, i8
                            %484 = arith.muli %arg17, %c512_i32 : i32
                            %485 = arith.index_cast %484 : i32 to index
                            %result0_22, %result1_23 = ascvec.loadalign_deintlv %arg12[%485] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %486 = ascvec.and %result1_23, %306, %304 : <ui8>, <ui8> -> <ui8>
                            %487 = ascvec.or %483, %486, %304 : <ui8>, <ui8> -> <ui8>
                            %488 = ascvec.arange %c0_i16 : i16 -> <ui16>
                            %489 = arith.muli %arg17, %c16_i32 : i32
                            %490 = arith.trunci %489 : i32 to i16
                            %491 = ascvec.add_scalar %488, %490, %303 : <ui16>, i16
                            %492 = ascvec.gather %arg13[%c0_15], %491, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                            %493 = ascvec.pack %492 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                            %494 = ascvec.gather_reg %493, %307 : <ui8>, <ui8> -> <ui8>
                            %495 = ascvec.cmp_eq_scalar %494, %c-1_i8, %304 : <ui8>, i8 -> !ascvec.mask
                            %496 = ascvec.duplicate_scalar %c127_i8, %495 merge %487 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                            ascvec.storealign %arg14[%314], %496, %304 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
                      }
                    }
                  }
                }
              } else {
                %224 = arith.divsi %99, %c256_i64 : i64
                %225 = arith.addi %99, %c1023_i64 : i64
                %226 = arith.remsi %225, %c1024_i64 : i64
                %227 = arith.cmpi ne, %226, %c0_i64 : i64
                %228 = arith.cmpi slt, %226, %c0_i64 : i64
                %229 = arith.andi %227, %228 : i1
                %230 = arith.divsi %225, %c1024_i64 : i64
                %231 = arith.extui %229 : i1 to i64
                %232 = arith.subi %230, %231 : i64
                %233 = arith.trunci %1 : i64 to i32
                %234 = arith.extui %233 : i32 to i64
                %235 = arith.subi %1, %234 : i64
                %236 = arith.remsi %235, %c4294967296_i64 : i64
                %237 = arith.cmpi ne, %236, %c0_i64 : i64
                %238 = arith.cmpi slt, %236, %c0_i64 : i64
                %239 = arith.andi %237, %238 : i1
                %240 = arith.divsi %235, %c4294967296_i64 : i64
                %241 = arith.extui %239 : i1 to i64
                %242 = arith.subi %240, %241 : i64
                %243 = arith.trunci %242 : i64 to i32
                %244 = arith.divui %212, %c32_i64 : i64
                %245 = arith.trunci %244 : i64 to i32
                %246 = arith.extui %245 : i32 to i64
                %247 = arith.subi %244, %246 : i64
                %248 = arith.remsi %247, %c4294967296_i64 : i64
                %249 = arith.cmpi ne, %248, %c0_i64 : i64
                %250 = arith.cmpi slt, %248, %c0_i64 : i64
                %251 = arith.andi %249, %250 : i1
                %252 = arith.divsi %247, %c4294967296_i64 : i64
                %253 = arith.extui %251 : i1 to i64
                %254 = arith.subi %252, %253 : i64
                %255 = arith.trunci %254 : i64 to i32
                %256 = arith.trunci %4 : i64 to i32
                %257 = arith.extui %256 : i32 to i64
                %258 = arith.subi %4, %257 : i64
                %259 = arith.remsi %258, %c4294967296_i64 : i64
                %260 = arith.cmpi ne, %259, %c0_i64 : i64
                %261 = arith.cmpi slt, %259, %c0_i64 : i64
                %262 = arith.andi %260, %261 : i1
                %263 = arith.divsi %258, %c4294967296_i64 : i64
                %264 = arith.extui %262 : i1 to i64
                %265 = arith.subi %263, %264 : i64
                %266 = arith.trunci %265 : i64 to i32
                %267 = arith.divui %256, %c4_i32 : i32
                %268 = arith.remui %266, %c4_i32 : i32
                %269 = arith.muli %268, %c1073741824_i32 : i32
                %270 = arith.addi %267, %269 : i32
                %271 = arith.divui %266, %c4_i32 : i32
                %272 = arith.muli %232, %c32_i64 : i64
                %273 = arith.divui %244, %c16384_i64 : i64
                %274 = arith.addi %244, %272 : i64
                %275 = arith.subi %274, %c1_i64 : i64
                %276 = arith.divui %275, %c16384_i64 : i64
                %277 = arith.cmpi eq, %273, %276 : i64
                scf.if %277 {
                  %287 = arith.trunci %273 : i64 to i32
                  %288 = arith.extui %287 : i32 to i64
                  %289 = arith.subi %273, %288 : i64
                  %290 = arith.remsi %289, %c4294967296_i64 : i64
                  %291 = arith.cmpi ne, %290, %c0_i64 : i64
                  %292 = arith.cmpi slt, %290, %c0_i64 : i64
                  %293 = arith.andi %291, %292 : i1
                  %294 = arith.divsi %289, %c4294967296_i64 : i64
                  %295 = arith.extui %293 : i1 to i64
                  %296 = arith.subi %294, %295 : i64
                  %297 = arith.trunci %296 : i64 to i32
                  %298 = arith.addi %270, %287 : i32
                  %299 = arith.addi %271, %297 : i32
                  %300 = arith.cmpi ult, %298, %270 : i32
                  %301 = arith.extui %300 : i1 to i32
                  %302 = arith.addi %299, %301 : i32
                  %303 = arith.extui %298 : i32 to i64
                  %304 = arith.muli %303, %c3449720151_i64 : i64
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
                  %316 = arith.xori %302, %243 : i32
                  %317 = arith.extui %316 : i32 to i64
                  %318 = arith.muli %317, %c3449720151_i64 : i64
                  %319 = arith.trunci %318 : i64 to i32
                  %320 = arith.extui %319 : i32 to i64
                  %321 = arith.subi %318, %320 : i64
                  %322 = arith.remsi %321, %c4294967296_i64 : i64
                  %323 = arith.cmpi ne, %322, %c0_i64 : i64
                  %324 = arith.cmpi slt, %322, %c0_i64 : i64
                  %325 = arith.andi %323, %324 : i1
                  %326 = arith.divsi %321, %c4294967296_i64 : i64
                  %327 = arith.extui %325 : i1 to i64
                  %328 = arith.subi %326, %327 : i64
                  %329 = arith.trunci %328 : i64 to i32
                  %330 = arith.addi %233, %c-1640531527_i32 : i32
                  %331 = arith.addi %243, %c-1150833019_i32 : i32
                  %332 = arith.xori %329, %305 : i32
                  %333 = arith.xori %332, %330 : i32
                  %334 = arith.extui %333 : i32 to i64
                  %335 = arith.muli %334, %c3528531795_i64 : i64
                  %336 = arith.trunci %335 : i64 to i32
                  %337 = arith.extui %336 : i32 to i64
                  %338 = arith.subi %335, %337 : i64
                  %339 = arith.remsi %338, %c4294967296_i64 : i64
                  %340 = arith.cmpi ne, %339, %c0_i64 : i64
                  %341 = arith.cmpi slt, %339, %c0_i64 : i64
                  %342 = arith.andi %340, %341 : i1
                  %343 = arith.divsi %338, %c4294967296_i64 : i64
                  %344 = arith.extui %342 : i1 to i64
                  %345 = arith.subi %343, %344 : i64
                  %346 = arith.trunci %345 : i64 to i32
                  %347 = arith.xori %315, %233 : i32
                  %348 = arith.addi %233, %c1013904242_i32 : i32
                  %349 = arith.xori %319, %348 : i32
                  %350 = arith.addi %243, %c1993301258_i32 : i32
                  %351 = arith.xori %346, %350 : i32
                  ascsimt.vf(%245, %347, %331, %349, %351, %233, %243, %336, %alloc_9, %272 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: memref<8192xi32, 1>, %arg18: i64):
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
                    %c16384_i32 = arith.constant 16384 : i32
                    %352 = ascsimt.thread_idx {ascsimt.scope = 38 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %353 = arith.extsi %352 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg19 = %353 to %arg18 step %c256_i64_11  : i64 {
                      %354 = arith.trunci %arg19 : i64 to i32
                      %355 = arith.addi %arg9, %354 : i32
                      %356 = arith.remui %355, %c16384_i32 : i32
                      %357 = arith.xori %356, %arg10 : i32
                      %358 = arith.muli %357, %c-766435501_i32 : i32
                      %359 = "ascsimt.bitcast"(%357) : (i32) -> ui32
                      %360 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %361 = ascsimt.math_binary %359, %360, "mulhi" : ui32, ui32 -> ui32
                      %362 = "ascsimt.bitcast"(%361) : (ui32) -> i32
                      %363 = arith.xori %362, %arg11 : i32
                      %364 = arith.muli %363, %c-845247145_i32 : i32
                      %365 = "ascsimt.bitcast"(%363) : (i32) -> ui32
                      %366 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %367 = ascsimt.math_binary %365, %366, "mulhi" : ui32, ui32 -> ui32
                      %368 = "ascsimt.bitcast"(%367) : (ui32) -> i32
                      %369 = arith.xori %368, %arg12 : i32
                      %370 = arith.xori %358, %arg13 : i32
                      %371:4 = scf.for %arg20 = %c3_i32 to %c10_i32 step %c1_i32_16 iter_args(%arg21 = %369, %arg22 = %364, %arg23 = %370, %arg24 = %arg16) -> (i32, i32, i32, i32)  : i32 {
                        %380 = arith.muli %arg21, %c-766435501_i32 : i32
                        %381 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %382 = ascsimt.math_binary %381, %360, "mulhi" : ui32, ui32 -> ui32
                        %383 = "ascsimt.bitcast"(%382) : (ui32) -> i32
                        %384 = arith.muli %arg23, %c-845247145_i32 : i32
                        %385 = "ascsimt.bitcast"(%arg23) : (i32) -> ui32
                        %386 = ascsimt.math_binary %385, %366, "mulhi" : ui32, ui32 -> ui32
                        %387 = "ascsimt.bitcast"(%386) : (ui32) -> i32
                        %388 = arith.muli %arg20, %c-1640531527_i32_18 : i32
                        %389 = arith.addi %arg14, %388 : i32
                        %390 = arith.muli %arg20, %c-1150833019_i32_17 : i32
                        %391 = arith.addi %arg15, %390 : i32
                        %392 = arith.xori %387, %arg22 : i32
                        %393 = arith.xori %392, %389 : i32
                        %394 = arith.xori %383, %arg24 : i32
                        %395 = arith.xori %394, %391 : i32
                        scf.yield %393, %384, %395, %380 : i32, i32, i32, i32
                      }
                      %372 = arith.muli %arg19, %c4_i64_15 : i64
                      %373 = arith.index_cast %372 : i64 to index
                      ascsimt.store %arg17[%373], %371#0 : memref<8192xi32, 1>, i32
                      %374 = arith.addi %372, %c1_i64_14 : i64
                      %375 = arith.index_cast %374 : i64 to index
                      ascsimt.store %arg17[%375], %371#1 : memref<8192xi32, 1>, i32
                      %376 = arith.addi %372, %c2_i64_13 : i64
                      %377 = arith.index_cast %376 : i64 to index
                      ascsimt.store %arg17[%377], %371#2 : memref<8192xi32, 1>, i32
                      %378 = arith.addi %372, %c3_i64_12 : i64
                      %379 = arith.index_cast %378 : i64 to index
                      ascsimt.store %arg17[%379], %371#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%245, %255, %270, %271, %233, %243, %alloc_9, %272 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
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
                    %c16384_i32 = arith.constant 16384 : i32
                    %287 = ascsimt.thread_idx {ascsimt.scope = 39 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %288 = arith.extsi %287 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg17 = %288 to %arg16 step %c256_i64_11  : i64 {
                      %289 = arith.trunci %arg17 : i64 to i32
                      %290 = arith.addi %arg9, %289 : i32
                      %291 = arith.cmpi ult, %290, %arg9 : i32
                      %292 = arith.extui %291 : i1 to i32
                      %293 = arith.addi %arg10, %292 : i32
                      %294 = arith.divui %290, %c16384_i32 : i32
                      %295 = arith.remui %293, %c16384_i32 : i32
                      %296 = arith.muli %295, %c262144_i32 : i32
                      %297 = arith.addi %294, %296 : i32
                      %298 = arith.divui %293, %c16384_i32 : i32
                      %299 = arith.addi %arg11, %297 : i32
                      %300 = arith.addi %arg12, %298 : i32
                      %301 = arith.cmpi ult, %299, %arg11 : i32
                      %302 = arith.extui %301 : i1 to i32
                      %303 = arith.addi %300, %302 : i32
                      %304 = arith.remui %290, %c16384_i32 : i32
                      %305:4 = scf.for %arg18 = %c0_i32_17 to %c10_i32 step %c1_i32_16 iter_args(%arg19 = %c0_i32_17, %arg20 = %304, %arg21 = %299, %arg22 = %303) -> (i32, i32, i32, i32)  : i32 {
                        %314 = arith.muli %arg19, %c-766435501_i32 : i32
                        %315 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %316 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %317 = ascsimt.math_binary %315, %316, "mulhi" : ui32, ui32 -> ui32
                        %318 = "ascsimt.bitcast"(%317) : (ui32) -> i32
                        %319 = arith.muli %arg21, %c-845247145_i32 : i32
                        %320 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %321 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %322 = ascsimt.math_binary %320, %321, "mulhi" : ui32, ui32 -> ui32
                        %323 = "ascsimt.bitcast"(%322) : (ui32) -> i32
                        %324 = arith.muli %arg18, %c-1640531527_i32_19 : i32
                        %325 = arith.addi %arg13, %324 : i32
                        %326 = arith.muli %arg18, %c-1150833019_i32_18 : i32
                        %327 = arith.addi %arg14, %326 : i32
                        %328 = arith.xori %323, %arg20 : i32
                        %329 = arith.xori %328, %325 : i32
                        %330 = arith.xori %318, %arg22 : i32
                        %331 = arith.xori %330, %327 : i32
                        scf.yield %329, %319, %331, %314 : i32, i32, i32, i32
                      }
                      %306 = arith.muli %arg17, %c4_i64_15 : i64
                      %307 = arith.index_cast %306 : i64 to index
                      ascsimt.store %arg15[%307], %305#0 : memref<8192xi32, 1>, i32
                      %308 = arith.addi %306, %c1_i64_14 : i64
                      %309 = arith.index_cast %308 : i64 to index
                      ascsimt.store %arg15[%309], %305#1 : memref<8192xi32, 1>, i32
                      %310 = arith.addi %306, %c2_i64_13 : i64
                      %311 = arith.index_cast %310 : i64 to index
                      ascsimt.store %arg15[%311], %305#2 : memref<8192xi32, 1>, i32
                      %312 = arith.addi %306, %c3_i64_12 : i64
                      %313 = arith.index_cast %312 : i64 to index
                      ascsimt.store %arg15[%313], %305#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %278 = arith.trunci %32 : i64 to i32
                %279 = arith.divsi %278, %c2_i32 : i32
                %280 = arith.trunci %279 : i32 to i8
                %281 = arith.trunci %224 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %282 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%280, %alloc_7, %134, %282, %281, %alloc_10 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, i32, memref<2048xui16, 1>) {
                ^bb0(%arg9: i8, %arg10: memref<2048xui8, 1>, %arg11: memref<32768xbf16, 1>, %arg12: memref<?xui16, 1>, %arg13: i32, %arg14: memref<2048xui16, 1>):
                  %c1_i32_11 = arith.constant 1 : i32
                  %c0_i32_12 = arith.constant 0 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c0_13 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %287 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %288 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %289 = ascvec.duplicate_scalar %c32767_i16, %287 : i16 -> <ui16>
                  %290 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %291 = ascvec.shr %290, 4, %287 : <ui16> -> <ui16>
                  %292 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %293 = ascvec.duplicate_scalar %c1_i8, %288 : i8 -> <ui8>
                  %294 = ascvec.and %292, %293, %288 : <ui8>, <ui8> -> <ui8>
                  %295 = ascvec.cmp_lt_scalar %292, %c16_i8, %288 : <ui8>, i8 -> !ascvec.mask
                  %296 = ascvec.cmp_lt_scalar %294, %arg9, %288 : <ui8>, i8 -> !ascvec.mask
                  %297 = ascvec.mask_and %295, %296, %288
                  ascvec.store_unalign_begin %arg10 : memref<2048xui8, 1>
                  %298 = ascvec.duplicate_scalar %c-1_i16, %287 : i16 -> <ui16>
                  %299 = ascvec.duplicate_scalar %c0_i16, %287 : i16 -> <ui16>
                  %300:2 = scf.for %arg15 = %c0_i32_12 to %arg13 step %c1_i32_11 iter_args(%arg16 = %299, %arg17 = %298) -> (!ascvec.vreg<ui16>, !ascvec.vreg<ui16>)  : i32 {
                    %301 = arith.muli %arg15, %c256_i32 : i32
                    %302 = arith.index_cast %301 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg11[%302] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %303 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %304 = ascvec.and %303, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %305 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %306 = ascvec.and %305, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %307 = ascvec.max %304, %306, %287 : <ui16>
                    %308 = ascvec.reduce_max_datablock %307, %287 : <ui16>
                    %309 = ascvec.shr %308, 7, %287 : <ui16> -> <ui16>
                    %310 = ascvec.cmp_ge_scalar %309, %c255_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %311 = ascvec.max_scalar %309, %c15_i16, %287 : <ui16>, i16
                    %312 = ascvec.add_scalar %311, %c-15_i16, %287 : <ui16>, i16
                    %313 = ascvec.duplicate_scalar %c255_i16, %310 merge %312 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %314 = ascvec.duplicate_scalar %c254_i16, %287 : i16 -> <ui16>
                    %315 = ascvec.sub %314, %313, %287 : <ui16>
                    %316 = ascvec.cmp_ne_scalar %313, %c0_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %317 = ascvec.shl %315, 7, %316 : <ui16> -> <ui16>
                    %318 = ascvec.max %arg16, %313, %mask : <ui16>
                    %319 = ascvec.gather_reg %317, %291 : <ui16>, <ui16> -> <ui16>
                    %320 = ascvec.reinterpret %319 : <ui16> -> <bf16>
                    %321 = ascvec.mul %result0, %320, %287 : <bf16>
                    %322 = ascvec.mul %result1, %320, %287 : <bf16>
                    ascvec.storealign_intlv %arg11[%302], %321, %322 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %323 = ascvec.reinterpret %321 : <bf16> -> <ui16>
                    %324 = ascvec.and %323, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %325 = ascvec.reinterpret %322 : <bf16> -> <ui16>
                    %326 = ascvec.and %325, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %327 = ascvec.min %324, %326, %287 : <ui16>
                    %328 = ascvec.min %arg17, %327, %287 : <ui16>
                    %329 = arith.muli %arg15, %c16_i32 : i32
                    %330 = arith.index_cast %329 : i32 to index
                    ascvec.storealign %arg12[%330], %313, %mask : memref<?xui16, 1>, index, <ui16>
                    %331 = ascvec.reinterpret %313 : <ui16> -> <ui8>
                    %332 = ascvec.squeeze_store_reg %331, %297 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_13], %332 : memref<2048xui8, 1>, <ui8>
                    scf.yield %318, %328 : !ascvec.vreg<ui16>, !ascvec.vreg<ui16>
                  }
                  %c0_14 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg15 = %c0_14 to %c1 step %c1 {
                    %301 = ascvec.reduce_min %300#1, %287 : <ui16>
                    %302 = ascvec.reduce_max %300#0, %mask : <ui16>
                    %303 = ascvec.cmp_eq_scalar %302, %c255_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %304 = ascvec.duplicate_scalar %c0_i16, %303 merge %301 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.store_unalign_post %arg10[%c0_13] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg14[%c0_13], %304 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %283 = memref.load %alloc_10[%c0] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %284 = cannir.scalar_storage_cast %283 : ui16 -> i16
                %285 = arith.extui %284 : i16 to i64
                %286 = arith.cmpi sge, %285, %c14464_i64 : i64
                scf.if %286 {
                  ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %287 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                  %288 = ascscalar.bitcast_memref %134 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%134, %287, %288, %alloc_6, %281 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_11 = arith.constant 1 : i32
                    %c0_i32_12 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c128_i32 = arith.constant 128 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c31_i16 = arith.constant 31 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %289 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %290 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %291 = ascvec.duplicate_scalar %c255_i16, %289 : i16 -> <ui16>
                    %292 = ascvec.duplicate_scalar %c-128_i8, %290 : i8 -> <ui8>
                    %293 = ascvec.duplicate_scalar %c31_i16, %289 : i16 -> <ui16>
                    scf.for %arg14 = %c0_i32_12 to %arg13 step %c1_i32_11  : i32 {
                      %294 = arith.muli %arg14, %c256_i32 : i32
                      %295 = arith.index_cast %294 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg9[%295] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %296 = arith.muli %arg14, %c128_i32 : i32
                      %297 = arith.index_cast %296 : i32 to index
                      %298 = ascvec.loadalign %arg10[%297] : memref<?xui8, 1>, index -> <ui8>
                      %299 = ascvec.unpack %298 {part = "lower"} : <ui8> -> <ui16>
                      %300 = ascvec.and %299, %293, %289 : <ui16>, <ui16> -> <ui16>
                      %301 = ascvec.shr %299, 3, %289 : <ui16> -> <ui16>
                      %302 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %303 = ascvec.add %302, %300, %289 : <ui16>
                      %304 = ascvec.shr %303, 5, %289 : <ui16> -> <ui16>
                      %305 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %306 = ascvec.add %305, %301, %289 : <ui16>
                      %307 = ascvec.shr %306, 5, %289 : <ui16> -> <ui16>
                      %308 = ascvec.and %304, %291, %289 : <ui16>, <ui16> -> <ui16>
                      %309 = ascvec.shl %307, 8, %289 : <ui16> -> <ui16>
                      %310 = ascvec.or %308, %309, %289 : <ui16>, <ui16> -> <ui16>
                      %311 = ascvec.reinterpret %310 : <ui16> -> <ui8>
                      %312 = ascvec.add_scalar %311, %c64_i8, %290 : <ui8>, i8
                      %313 = ascvec.min_scalar %312, %c123_i8, %290 : <ui8>, i8
                      %314 = arith.muli %arg14, %c512_i32 : i32
                      %315 = arith.index_cast %314 : i32 to index
                      %result0_13, %result1_14 = ascvec.loadalign_deintlv %arg11[%315] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %316 = ascvec.and %result1_14, %292, %290 : <ui8>, <ui8> -> <ui8>
                      %317 = ascvec.or %313, %316, %290 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg12[%295], %317, %290 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
                } else {
                  %287 = arith.cmpi sge, %285, %c13568_i64 : i64
                  scf.if %287 {
                    ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %288 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<8192xi32, 1> -> memref<?xui8, 1>
                    %289 = ascscalar.bitcast_memref %134 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                    ascvec.vf(%134, %288, %289, %alloc_6, %281 : memref<32768xbf16, 1>, memref<?xui8, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                    ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<?xui8, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32):
                      %c512_i32 = arith.constant 512 : i32
                      %c1_i32_11 = arith.constant 1 : i32
                      %c0_i32_12 = arith.constant 0 : i32
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
                      %290 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                      %291 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                      %292 = ascvec.duplicate_scalar %c255_i16, %290 : i16 -> <ui16>
                      %293 = ascvec.duplicate_scalar %c-128_i8, %291 : i8 -> <ui8>
                      %294 = ascvec.arange %c0_i8 : i8 -> <ui8>
                      %295 = ascvec.shr %294, 1, %291 : <ui8> -> <ui8>
                      %296 = ascvec.shr %295, 2, %291 : <ui8> -> <ui8>
                      %297 = ascvec.shl %296, 2, %291 : <ui8> -> <ui8>
                      %298 = ascvec.duplicate_scalar %c1_i8, %291 : i8 -> <ui8>
                      %299 = ascvec.and %294, %298, %291 : <ui8>, <ui8> -> <ui8>
                      %300 = ascvec.add_scalar %295, %c3_i8, %291 : <ui8>, i8
                      %301 = ascvec.add %300, %299, %291 : <ui8>
                      %302 = ascvec.duplicate_scalar %c3_i8, %291 : i8 -> <ui8>
                      %303 = ascvec.and %301, %302, %291 : <ui8>, <ui8> -> <ui8>
                      %304 = ascvec.or %297, %303, %291 : <ui8>, <ui8> -> <ui8>
                      %305 = ascvec.duplicate_scalar %c4095_i16, %290 : i16 -> <ui16>
                      %306 = ascvec.duplicate_scalar %c32767_i16, %290 : i16 -> <ui16>
                      %307 = ascvec.duplicate_scalar %cst, %290 : bf16 -> <bf16>
                      scf.for %arg14 = %c0_i32_12 to %arg13 step %c1_i32_11  : i32 {
                        %308 = arith.muli %arg14, %c256_i32 : i32
                        %309 = arith.index_cast %308 : i32 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg9[%309] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                        %310 = arith.muli %arg14, %c128_i32 : i32
                        %311 = arith.index_cast %310 : i32 to index
                        %312 = ascvec.loadalign %arg10[%311] : memref<?xui8, 1>, index -> <ui8>
                        %313 = ascvec.gather_reg %312, %304 : <ui8>, <ui8> -> <ui8>
                        %314 = ascvec.reinterpret %313 : <ui8> -> <ui16>
                        %315 = ascvec.shr %314, 1, %290 : <ui16> -> <ui16>
                        %316 = ascvec.and %315, %305, %290 : <ui16>, <ui16> -> <ui16>
                        %317 = ascvec.shr %314, 4, %290 : <ui16> -> <ui16>
                        %318 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                        %319 = ascvec.and %318, %306, %290 : <ui16>, <ui16> -> <ui16>
                        %320 = ascvec.cmp_lt_scalar %319, %c14464_i16, %290 : <ui16>, i16 -> !ascvec.mask
                        %321 = ascvec.shr %316, 7, %290 : <ui16> -> <ui16>
                        %322 = ascvec.add %319, %321, %290 : <ui16>
                        %323 = ascvec.shr %322, 5, %290 : <ui16> -> <ui16>
                        %324 = ascvec.reinterpret %319 : <ui16> -> <bf16>
                        %325 = ascvec.mul %324, %307, %320 : <bf16>
                        %326 = ascvec.cast %325, %290 {rounding = 4 : i32} : <bf16> -> <f16>
                        %327 = ascvec.cast %326, %290 {rounding = 4 : i32} : <f16> -> <i16>
                        %328 = ascvec.reinterpret %327 : <i16> -> <ui16>
                        %329 = ascvec.add %328, %316, %290 : <ui16>
                        %330 = ascvec.shr %329, 12, %290 : <ui16> -> <ui16>
                        %331 = ascvec.add_scalar %330, %c448_i16, %290 : <ui16>, i16
                        %332 = ascvec.select %331, %323, %320 : <ui16>
                        %333 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                        %334 = ascvec.and %333, %306, %290 : <ui16>, <ui16> -> <ui16>
                        %335 = ascvec.cmp_lt_scalar %334, %c14464_i16, %290 : <ui16>, i16 -> !ascvec.mask
                        %336 = ascvec.shr %317, 7, %290 : <ui16> -> <ui16>
                        %337 = ascvec.add %334, %336, %290 : <ui16>
                        %338 = ascvec.shr %337, 5, %290 : <ui16> -> <ui16>
                        %339 = ascvec.reinterpret %334 : <ui16> -> <bf16>
                        %340 = ascvec.mul %339, %307, %335 : <bf16>
                        %341 = ascvec.cast %340, %290 {rounding = 4 : i32} : <bf16> -> <f16>
                        %342 = ascvec.cast %341, %290 {rounding = 4 : i32} : <f16> -> <i16>
                        %343 = ascvec.reinterpret %342 : <i16> -> <ui16>
                        %344 = ascvec.add %343, %317, %290 : <ui16>
                        %345 = ascvec.shr %344, 12, %290 : <ui16> -> <ui16>
                        %346 = ascvec.add_scalar %345, %c448_i16, %290 : <ui16>, i16
                        %347 = ascvec.select %346, %338, %335 : <ui16>
                        %348 = ascvec.and %332, %292, %290 : <ui16>, <ui16> -> <ui16>
                        %349 = ascvec.shl %347, 8, %290 : <ui16> -> <ui16>
                        %350 = ascvec.or %348, %349, %290 : <ui16>, <ui16> -> <ui16>
                        %351 = ascvec.reinterpret %350 : <ui16> -> <ui8>
                        %352 = ascvec.add_scalar %351, %c64_i8, %291 : <ui8>, i8
                        %353 = ascvec.min_scalar %352, %c123_i8, %291 : <ui8>, i8
                        %354 = arith.muli %arg14, %c512_i32 : i32
                        %355 = arith.index_cast %354 : i32 to index
                        %result0_13, %result1_14 = ascvec.loadalign_deintlv %arg11[%355] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                        %356 = ascvec.and %result1_14, %293, %291 : <ui8>, <ui8> -> <ui8>
                        %357 = ascvec.or %353, %356, %291 : <ui8>, <ui8> -> <ui8>
                        ascvec.storealign %arg12[%309], %357, %291 : memref<32768xui8, 1>, index, <ui8>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
                  } else {
                    ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
                    ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    %288 = ascscalar.bitcast_memref %134 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                    ascvec.vf(%134, %alloc_9, %288, %282, %alloc_6, %281 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32) {
                    ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<8192xi32, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<?xui16, 1>, %arg13: memref<32768xui8, 1>, %arg14: i32):
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
                      %c4_i32_11 = arith.constant 4 : i32
                      %c2_i32_12 = arith.constant 2 : i32
                      %c1_i32_13 = arith.constant 1 : i32
                      %c7_i32 = arith.constant 7 : i32
                      %c32_i32 = arith.constant 32 : i32
                      %c0_i32_14 = arith.constant 0 : i32
                      %c32767_i16 = arith.constant 32767 : i16
                      %c0_15 = arith.constant 0 : index
                      %c256_i32 = arith.constant 256 : i32
                      %c3_i8 = arith.constant 3 : i8
                      %c-128_i8 = arith.constant -128 : i8
                      %c0_i8 = arith.constant 0 : i8
                      %c8_i32 = arith.constant 8 : i32
                      %289 = ascvec.create_mask : !ascvec.mask
                      %290 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                      %291 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                      %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                      %292 = ascvec.arange %c0_i8 : i8 -> <ui8>
                      %293 = ascvec.duplicate_scalar %c-128_i8, %291 : i8 -> <ui8>
                      %294 = ascvec.shr %292, 5, %291 : <ui8> -> <ui8>
                      %295 = ascvec.duplicate_scalar %c3_i8, %291 : i8 -> <ui8>
                      %296 = ascvec.and %292, %295, %291 : <ui8>, <ui8> -> <ui8>
                      %297 = ascvec.shl %296, 6, %291 : <ui8> -> <ui8>
                      %298 = ascvec.shr %292, 2, %291 : <ui8> -> <ui8>
                      %299 = ascvec.or %297, %298, %291 : <ui8>, <ui8> -> <ui8>
                      scf.for %arg15 = %c0_i32_14 to %arg14 step %c1_i32_13  : i32 {
                        %300 = arith.muli %arg15, %c256_i32 : i32
                        %301 = arith.index_cast %300 : i32 to index
                        %result0, %result1 = ascvec.loadalign_deintlv %arg9[%301] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                        %302 = ascvec.duplicate_scalar %c32767_i16, %290 : i16 -> <ui16>
                        %303 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                        %304 = ascvec.and %303, %302, %290 : <ui16>, <ui16> -> <ui16>
                        %305 = ascvec.reinterpret %304 : <ui16> -> <bf16>
                        %306 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                        %307 = ascvec.and %306, %302, %290 : <ui16>, <ui16> -> <ui16>
                        %308 = ascvec.reinterpret %307 : <ui16> -> <bf16>
                        %309 = ascvec.cast %305, %290 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %310 = ascvec.cast %305, %290 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %311 = ascvec.cast %308, %290 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %312 = ascvec.cast %308, %290 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                        %313 = ascvec.arange %c0_i32_14 : i32 -> <ui32>
                        %314 = ascvec.shl %313, 2, %289 : <ui32> -> <ui32>
                        %315 = ascvec.add_scalar %314, %c0_i32_14, %289 : <ui32>, i32
                        %316 = ascvec.shr %315, 3, %289 : <ui32> -> <ui32>
                        %317 = arith.muli %arg15, %c32_i32 : i32
                        %318 = ascvec.add_scalar %316, %317, %289 : <ui32>, i32
                        %319 = ascvec.gather %arg10[%c0_15], %318, %289 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %320 = ascvec.reinterpret %319 : <i32> -> <ui32>
                        %321 = ascvec.duplicate_scalar %c7_i32, %289 : i32 -> <ui32>
                        %322 = ascvec.and %315, %321, %289 : <ui32>, <ui32> -> <ui32>
                        %323 = ascvec.shl %320, 11, %289 : <ui32> -> <ui32>
                        %324 = ascvec.shr %320, 21, %289 : <ui32> -> <ui32>
                        %325 = ascvec.or %323, %324, %289 : <ui32>, <ui32> -> <ui32>
                        %326 = ascvec.shr %325, 3, %289 : <ui32> -> <ui32>
                        %327 = ascvec.shl %325, 29, %289 : <ui32> -> <ui32>
                        %328 = ascvec.or %326, %327, %289 : <ui32>, <ui32> -> <ui32>
                        %329 = ascvec.duplicate_scalar %c1_i32_13, %289 : i32 -> <ui32>
                        %330 = ascvec.and %322, %329, %289 : <ui32>, <ui32> -> <ui32>
                        %331 = ascvec.cmp_ne_scalar %330, %c0_i32_14, %289 : <ui32>, i32 -> !ascvec.mask
                        %332 = ascvec.select %328, %325, %331 : <ui32>
                        %333 = ascvec.shr %332, 8, %289 : <ui32> -> <ui32>
                        %334 = ascvec.shl %332, 24, %289 : <ui32> -> <ui32>
                        %335 = ascvec.or %333, %334, %289 : <ui32>, <ui32> -> <ui32>
                        %336 = ascvec.duplicate_scalar %c2_i32_12, %289 : i32 -> <ui32>
                        %337 = ascvec.and %322, %336, %289 : <ui32>, <ui32> -> <ui32>
                        %338 = ascvec.cmp_ne_scalar %337, %c0_i32_14, %289 : <ui32>, i32 -> !ascvec.mask
                        %339 = ascvec.select %335, %332, %338 : <ui32>
                        %340 = ascvec.shr %339, 16, %289 : <ui32> -> <ui32>
                        %341 = ascvec.shl %339, 16, %289 : <ui32> -> <ui32>
                        %342 = ascvec.or %340, %341, %289 : <ui32>, <ui32> -> <ui32>
                        %343 = ascvec.duplicate_scalar %c4_i32_11, %289 : i32 -> <ui32>
                        %344 = ascvec.and %322, %343, %289 : <ui32>, <ui32> -> <ui32>
                        %345 = ascvec.cmp_ne_scalar %344, %c0_i32_14, %289 : <ui32>, i32 -> !ascvec.mask
                        %346 = ascvec.select %342, %339, %345 : <ui32>
                        %347 = ascvec.duplicate_scalar %c65535_i32, %289 : i32 -> <ui32>
                        %348 = ascvec.and %346, %347, %289 : <ui32>, <ui32> -> <ui32>
                        %349 = ascvec.add_scalar %314, %c1_i32_13, %289 : <ui32>, i32
                        %350 = ascvec.shr %349, 3, %289 : <ui32> -> <ui32>
                        %351 = ascvec.add_scalar %350, %317, %289 : <ui32>, i32
                        %352 = ascvec.gather %arg10[%c0_15], %351, %289 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %353 = ascvec.reinterpret %352 : <i32> -> <ui32>
                        %354 = ascvec.and %349, %321, %289 : <ui32>, <ui32> -> <ui32>
                        %355 = ascvec.shl %353, 11, %289 : <ui32> -> <ui32>
                        %356 = ascvec.shr %353, 21, %289 : <ui32> -> <ui32>
                        %357 = ascvec.or %355, %356, %289 : <ui32>, <ui32> -> <ui32>
                        %358 = ascvec.shr %357, 3, %289 : <ui32> -> <ui32>
                        %359 = ascvec.shl %357, 29, %289 : <ui32> -> <ui32>
                        %360 = ascvec.or %358, %359, %289 : <ui32>, <ui32> -> <ui32>
                        %361 = ascvec.and %354, %329, %289 : <ui32>, <ui32> -> <ui32>
                        %362 = ascvec.cmp_ne_scalar %361, %c0_i32_14, %289 : <ui32>, i32 -> !ascvec.mask
                        %363 = ascvec.select %360, %357, %362 : <ui32>
                        %364 = ascvec.shr %363, 8, %289 : <ui32> -> <ui32>
                        %365 = ascvec.shl %363, 24, %289 : <ui32> -> <ui32>
                        %366 = ascvec.or %364, %365, %289 : <ui32>, <ui32> -> <ui32>
                        %367 = ascvec.and %354, %336, %289 : <ui32>, <ui32> -> <ui32>
                        %368 = ascvec.cmp_ne_scalar %367, %c0_i32_14, %289 : <ui32>, i32 -> !ascvec.mask
                        %369 = ascvec.select %366, %363, %368 : <ui32>
                        %370 = ascvec.shr %369, 16, %289 : <ui32> -> <ui32>
                        %371 = ascvec.shl %369, 16, %289 : <ui32> -> <ui32>
                        %372 = ascvec.or %370, %371, %289 : <ui32>, <ui32> -> <ui32>
                        %373 = ascvec.and %354, %343, %289 : <ui32>, <ui32> -> <ui32>
                        %374 = ascvec.cmp_ne_scalar %373, %c0_i32_14, %289 : <ui32>, i32 -> !ascvec.mask
                        %375 = ascvec.select %372, %369, %374 : <ui32>
                        %376 = ascvec.and %375, %347, %289 : <ui32>, <ui32> -> <ui32>
                        %377 = ascvec.add_scalar %314, %c2_i32_12, %289 : <ui32>, i32
                        %378 = ascvec.shr %377, 3, %289 : <ui32> -> <ui32>
                        %379 = ascvec.add_scalar %378, %317, %289 : <ui32>, i32
                        %380 = ascvec.gather %arg10[%c0_15], %379, %289 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %381 = ascvec.reinterpret %380 : <i32> -> <ui32>
                        %382 = ascvec.and %377, %321, %289 : <ui32>, <ui32> -> <ui32>
                        %383 = ascvec.shl %381, 11, %289 : <ui32> -> <ui32>
                        %384 = ascvec.shr %381, 21, %289 : <ui32> -> <ui32>
                        %385 = ascvec.or %383, %384, %289 : <ui32>, <ui32> -> <ui32>
                        %386 = ascvec.shr %385, 3, %289 : <ui32> -> <ui32>
                        %387 = ascvec.shl %385, 29, %289 : <ui32> -> <ui32>
                        %388 = ascvec.or %386, %387, %289 : <ui32>, <ui32> -> <ui32>
                        %389 = ascvec.and %382, %329, %289 : <ui32>, <ui32> -> <ui32>
                        %390 = ascvec.cmp_ne_scalar %389, %c0_i32_14, %289 : <ui32>, i32 -> !ascvec.mask
                        %391 = ascvec.select %388, %385, %390 : <ui32>
                        %392 = ascvec.shr %391, 8, %289 : <ui32> -> <ui32>
                        %393 = ascvec.shl %391, 24, %289 : <ui32> -> <ui32>
                        %394 = ascvec.or %392, %393, %289 : <ui32>, <ui32> -> <ui32>
                        %395 = ascvec.and %382, %336, %289 : <ui32>, <ui32> -> <ui32>
                        %396 = ascvec.cmp_ne_scalar %395, %c0_i32_14, %289 : <ui32>, i32 -> !ascvec.mask
                        %397 = ascvec.select %394, %391, %396 : <ui32>
                        %398 = ascvec.shr %397, 16, %289 : <ui32> -> <ui32>
                        %399 = ascvec.shl %397, 16, %289 : <ui32> -> <ui32>
                        %400 = ascvec.or %398, %399, %289 : <ui32>, <ui32> -> <ui32>
                        %401 = ascvec.and %382, %343, %289 : <ui32>, <ui32> -> <ui32>
                        %402 = ascvec.cmp_ne_scalar %401, %c0_i32_14, %289 : <ui32>, i32 -> !ascvec.mask
                        %403 = ascvec.select %400, %397, %402 : <ui32>
                        %404 = ascvec.and %403, %347, %289 : <ui32>, <ui32> -> <ui32>
                        %405 = ascvec.add_scalar %314, %c3_i32, %289 : <ui32>, i32
                        %406 = ascvec.shr %405, 3, %289 : <ui32> -> <ui32>
                        %407 = ascvec.add_scalar %406, %317, %289 : <ui32>, i32
                        %408 = ascvec.gather %arg10[%c0_15], %407, %289 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                        %409 = ascvec.reinterpret %408 : <i32> -> <ui32>
                        %410 = ascvec.and %405, %321, %289 : <ui32>, <ui32> -> <ui32>
                        %411 = ascvec.shl %409, 11, %289 : <ui32> -> <ui32>
                        %412 = ascvec.shr %409, 21, %289 : <ui32> -> <ui32>
                        %413 = ascvec.or %411, %412, %289 : <ui32>, <ui32> -> <ui32>
                        %414 = ascvec.shr %413, 3, %289 : <ui32> -> <ui32>
                        %415 = ascvec.shl %413, 29, %289 : <ui32> -> <ui32>
                        %416 = ascvec.or %414, %415, %289 : <ui32>, <ui32> -> <ui32>
                        %417 = ascvec.and %410, %329, %289 : <ui32>, <ui32> -> <ui32>
                        %418 = ascvec.cmp_ne_scalar %417, %c0_i32_14, %289 : <ui32>, i32 -> !ascvec.mask
                        %419 = ascvec.select %416, %413, %418 : <ui32>
                        %420 = ascvec.shr %419, 8, %289 : <ui32> -> <ui32>
                        %421 = ascvec.shl %419, 24, %289 : <ui32> -> <ui32>
                        %422 = ascvec.or %420, %421, %289 : <ui32>, <ui32> -> <ui32>
                        %423 = ascvec.and %410, %336, %289 : <ui32>, <ui32> -> <ui32>
                        %424 = ascvec.cmp_ne_scalar %423, %c0_i32_14, %289 : <ui32>, i32 -> !ascvec.mask
                        %425 = ascvec.select %422, %419, %424 : <ui32>
                        %426 = ascvec.shr %425, 16, %289 : <ui32> -> <ui32>
                        %427 = ascvec.shl %425, 16, %289 : <ui32> -> <ui32>
                        %428 = ascvec.or %426, %427, %289 : <ui32>, <ui32> -> <ui32>
                        %429 = ascvec.and %410, %343, %289 : <ui32>, <ui32> -> <ui32>
                        %430 = ascvec.cmp_ne_scalar %429, %c0_i32_14, %289 : <ui32>, i32 -> !ascvec.mask
                        %431 = ascvec.select %428, %425, %430 : <ui32>
                        %432 = ascvec.and %431, %347, %289 : <ui32>, <ui32> -> <ui32>
                        %433 = ascvec.reinterpret %309 : <f32> -> <ui32>
                        %434 = ascvec.reinterpret %433 : <ui32> -> <f32>
                        %435 = ascvec.max_scalar %434, %cst, %289 : <f32>, f32
                        %436 = ascvec.add %434, %435, %289 : <f32>
                        %437 = ascvec.reinterpret %436 : <f32> -> <ui32>
                        %438 = ascvec.shr %437, 5, %289 : <ui32> -> <ui32>
                        %439 = ascvec.add %438, %348, %289 : <ui32>
                        %440 = ascvec.reinterpret %311 : <f32> -> <ui32>
                        %441 = ascvec.reinterpret %440 : <ui32> -> <f32>
                        %442 = ascvec.max_scalar %441, %cst, %289 : <f32>, f32
                        %443 = ascvec.add %441, %442, %289 : <f32>
                        %444 = ascvec.reinterpret %443 : <f32> -> <ui32>
                        %445 = ascvec.shr %444, 5, %289 : <ui32> -> <ui32>
                        %446 = ascvec.add %445, %376, %289 : <ui32>
                        %447 = ascvec.reinterpret %310 : <f32> -> <ui32>
                        %448 = ascvec.reinterpret %447 : <ui32> -> <f32>
                        %449 = ascvec.max_scalar %448, %cst, %289 : <f32>, f32
                        %450 = ascvec.add %448, %449, %289 : <f32>
                        %451 = ascvec.reinterpret %450 : <f32> -> <ui32>
                        %452 = ascvec.shr %451, 5, %289 : <ui32> -> <ui32>
                        %453 = ascvec.add %452, %404, %289 : <ui32>
                        %454 = ascvec.reinterpret %312 : <f32> -> <ui32>
                        %455 = ascvec.reinterpret %454 : <ui32> -> <f32>
                        %456 = ascvec.max_scalar %455, %cst, %289 : <f32>, f32
                        %457 = ascvec.add %455, %456, %289 : <f32>
                        %458 = ascvec.reinterpret %457 : <f32> -> <ui32>
                        %459 = ascvec.shr %458, 5, %289 : <ui32> -> <ui32>
                        %460 = ascvec.add %459, %432, %289 : <ui32>
                        %461 = ascvec.reinterpret %439 : <ui32> -> <ui16>
                        %462 = ascvec.reinterpret %446 : <ui32> -> <ui16>
                        %result0_16, %result1_17 = ascvec.deintlv %461, %462 : <ui16>, <ui16> -> <ui16>, <ui16>
                        %463 = ascvec.reinterpret %453 : <ui32> -> <ui16>
                        %464 = ascvec.reinterpret %460 : <ui32> -> <ui16>
                        %result0_18, %result1_19 = ascvec.deintlv %463, %464 : <ui16>, <ui16> -> <ui16>, <ui16>
                        %465 = ascvec.reinterpret %result1_17 : <ui16> -> <ui8>
                        %466 = ascvec.reinterpret %result1_19 : <ui16> -> <ui8>
                        %result0_20, %result1_21 = ascvec.deintlv %465, %466 : <ui8>, <ui8> -> <ui8>, <ui8>
                        %467 = ascvec.gather_reg %result0_20, %299 : <ui8>, <ui8> -> <ui8>
                        %468 = ascvec.add_scalar %467, %c60_i8, %291 : <ui8>, i8
                        %469 = ascvec.cmp_ne_scalar %468, %c-1_i8, %291 : <ui8>, i8 -> !ascvec.mask
                        %470 = ascvec.min_scalar %468, %c123_i8, %469 : <ui8>, i8
                        %471 = arith.muli %arg15, %c512_i32 : i32
                        %472 = arith.index_cast %471 : i32 to index
                        %result0_22, %result1_23 = ascvec.loadalign_deintlv %arg11[%472] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                        %473 = ascvec.and %result1_23, %293, %291 : <ui8>, <ui8> -> <ui8>
                        %474 = ascvec.or %470, %473, %291 : <ui8>, <ui8> -> <ui8>
                        %475 = ascvec.arange %c0_i16 : i16 -> <ui16>
                        %476 = arith.muli %arg15, %c16_i32 : i32
                        %477 = arith.trunci %476 : i32 to i16
                        %478 = ascvec.add_scalar %475, %477, %290 : <ui16>, i16
                        %479 = ascvec.gather %arg12[%c0_15], %478, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                        %480 = ascvec.pack %479 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                        %481 = ascvec.gather_reg %480, %294 : <ui8>, <ui8> -> <ui8>
                        %482 = ascvec.cmp_eq_scalar %481, %c-1_i8, %291 : <ui8>, i8 -> !ascvec.mask
                        %483 = ascvec.duplicate_scalar %c127_i8, %482 merge %474 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                        ascvec.storealign %arg13[%301], %483, %291 : memref<32768xui8, 1>, index, <ui8>
                      }
                      ascvec.mem_bar vst_vld
                    } {cannir.vf_mode = "simd"}
                    ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                    ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
                  }
                }
              }
            } else {
              %220 = arith.cmpi sge, %99, %c1024_i64 : i64
              %221 = arith.remsi %99, %c1024_i64 : i64
              %222 = arith.cmpi eq, %221, %c0_i64 : i64
              %223 = arith.select %220, %222, %false : i1
              scf.if %223 {
                %224 = arith.divsi %99, %c256_i64 : i64
                %225 = arith.addi %99, %c1023_i64 : i64
                %226 = arith.remsi %225, %c1024_i64 : i64
                %227 = arith.cmpi ne, %226, %c0_i64 : i64
                %228 = arith.cmpi slt, %226, %c0_i64 : i64
                %229 = arith.andi %227, %228 : i1
                %230 = arith.divsi %225, %c1024_i64 : i64
                %231 = arith.extui %229 : i1 to i64
                %232 = arith.subi %230, %231 : i64
                %233 = arith.trunci %1 : i64 to i32
                %234 = arith.extui %233 : i32 to i64
                %235 = arith.subi %1, %234 : i64
                %236 = arith.remsi %235, %c4294967296_i64 : i64
                %237 = arith.cmpi ne, %236, %c0_i64 : i64
                %238 = arith.cmpi slt, %236, %c0_i64 : i64
                %239 = arith.andi %237, %238 : i1
                %240 = arith.divsi %235, %c4294967296_i64 : i64
                %241 = arith.extui %239 : i1 to i64
                %242 = arith.subi %240, %241 : i64
                %243 = arith.trunci %242 : i64 to i32
                %244 = arith.divui %212, %c16_i64 : i64
                %245 = arith.trunci %244 : i64 to i32
                %246 = arith.extui %245 : i32 to i64
                %247 = arith.subi %244, %246 : i64
                %248 = arith.remsi %247, %c4294967296_i64 : i64
                %249 = arith.cmpi ne, %248, %c0_i64 : i64
                %250 = arith.cmpi slt, %248, %c0_i64 : i64
                %251 = arith.andi %249, %250 : i1
                %252 = arith.divsi %247, %c4294967296_i64 : i64
                %253 = arith.extui %251 : i1 to i64
                %254 = arith.subi %252, %253 : i64
                %255 = arith.trunci %254 : i64 to i32
                %256 = arith.trunci %4 : i64 to i32
                %257 = arith.extui %256 : i32 to i64
                %258 = arith.subi %4, %257 : i64
                %259 = arith.remsi %258, %c4294967296_i64 : i64
                %260 = arith.cmpi ne, %259, %c0_i64 : i64
                %261 = arith.cmpi slt, %259, %c0_i64 : i64
                %262 = arith.andi %260, %261 : i1
                %263 = arith.divsi %258, %c4294967296_i64 : i64
                %264 = arith.extui %262 : i1 to i64
                %265 = arith.subi %263, %264 : i64
                %266 = arith.trunci %265 : i64 to i32
                %267 = arith.divui %256, %c4_i32 : i32
                %268 = arith.remui %266, %c4_i32 : i32
                %269 = arith.muli %268, %c1073741824_i32 : i32
                %270 = arith.addi %267, %269 : i32
                %271 = arith.divui %266, %c4_i32 : i32
                %272 = arith.muli %232, %c64_i64 : i64
                %273 = arith.divui %244, %c16384_i64 : i64
                %274 = arith.addi %244, %272 : i64
                %275 = arith.subi %274, %c1_i64 : i64
                %276 = arith.divui %275, %c16384_i64 : i64
                %277 = arith.cmpi eq, %273, %276 : i64
                scf.if %277 {
                  %287 = arith.trunci %273 : i64 to i32
                  %288 = arith.extui %287 : i32 to i64
                  %289 = arith.subi %273, %288 : i64
                  %290 = arith.remsi %289, %c4294967296_i64 : i64
                  %291 = arith.cmpi ne, %290, %c0_i64 : i64
                  %292 = arith.cmpi slt, %290, %c0_i64 : i64
                  %293 = arith.andi %291, %292 : i1
                  %294 = arith.divsi %289, %c4294967296_i64 : i64
                  %295 = arith.extui %293 : i1 to i64
                  %296 = arith.subi %294, %295 : i64
                  %297 = arith.trunci %296 : i64 to i32
                  %298 = arith.addi %270, %287 : i32
                  %299 = arith.addi %271, %297 : i32
                  %300 = arith.cmpi ult, %298, %270 : i32
                  %301 = arith.extui %300 : i1 to i32
                  %302 = arith.addi %299, %301 : i32
                  %303 = arith.extui %298 : i32 to i64
                  %304 = arith.muli %303, %c3449720151_i64 : i64
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
                  %316 = arith.xori %302, %243 : i32
                  %317 = arith.extui %316 : i32 to i64
                  %318 = arith.muli %317, %c3449720151_i64 : i64
                  %319 = arith.trunci %318 : i64 to i32
                  %320 = arith.extui %319 : i32 to i64
                  %321 = arith.subi %318, %320 : i64
                  %322 = arith.remsi %321, %c4294967296_i64 : i64
                  %323 = arith.cmpi ne, %322, %c0_i64 : i64
                  %324 = arith.cmpi slt, %322, %c0_i64 : i64
                  %325 = arith.andi %323, %324 : i1
                  %326 = arith.divsi %321, %c4294967296_i64 : i64
                  %327 = arith.extui %325 : i1 to i64
                  %328 = arith.subi %326, %327 : i64
                  %329 = arith.trunci %328 : i64 to i32
                  %330 = arith.addi %233, %c-1640531527_i32 : i32
                  %331 = arith.addi %243, %c-1150833019_i32 : i32
                  %332 = arith.xori %329, %305 : i32
                  %333 = arith.xori %332, %330 : i32
                  %334 = arith.extui %333 : i32 to i64
                  %335 = arith.muli %334, %c3528531795_i64 : i64
                  %336 = arith.trunci %335 : i64 to i32
                  %337 = arith.extui %336 : i32 to i64
                  %338 = arith.subi %335, %337 : i64
                  %339 = arith.remsi %338, %c4294967296_i64 : i64
                  %340 = arith.cmpi ne, %339, %c0_i64 : i64
                  %341 = arith.cmpi slt, %339, %c0_i64 : i64
                  %342 = arith.andi %340, %341 : i1
                  %343 = arith.divsi %338, %c4294967296_i64 : i64
                  %344 = arith.extui %342 : i1 to i64
                  %345 = arith.subi %343, %344 : i64
                  %346 = arith.trunci %345 : i64 to i32
                  %347 = arith.xori %315, %233 : i32
                  %348 = arith.addi %233, %c1013904242_i32 : i32
                  %349 = arith.xori %319, %348 : i32
                  %350 = arith.addi %243, %c1993301258_i32 : i32
                  %351 = arith.xori %346, %350 : i32
                  ascsimt.vf(%245, %347, %331, %349, %351, %233, %243, %336, %alloc_9, %272 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: memref<8192xi32, 1>, %arg18: i64):
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
                    %c16384_i32 = arith.constant 16384 : i32
                    %352 = ascsimt.thread_idx {ascsimt.scope = 44 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %353 = arith.extsi %352 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg19 = %353 to %arg18 step %c256_i64_11  : i64 {
                      %354 = arith.trunci %arg19 : i64 to i32
                      %355 = arith.addi %arg9, %354 : i32
                      %356 = arith.remui %355, %c16384_i32 : i32
                      %357 = arith.xori %356, %arg10 : i32
                      %358 = arith.muli %357, %c-766435501_i32 : i32
                      %359 = "ascsimt.bitcast"(%357) : (i32) -> ui32
                      %360 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %361 = ascsimt.math_binary %359, %360, "mulhi" : ui32, ui32 -> ui32
                      %362 = "ascsimt.bitcast"(%361) : (ui32) -> i32
                      %363 = arith.xori %362, %arg11 : i32
                      %364 = arith.muli %363, %c-845247145_i32 : i32
                      %365 = "ascsimt.bitcast"(%363) : (i32) -> ui32
                      %366 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %367 = ascsimt.math_binary %365, %366, "mulhi" : ui32, ui32 -> ui32
                      %368 = "ascsimt.bitcast"(%367) : (ui32) -> i32
                      %369 = arith.xori %368, %arg12 : i32
                      %370 = arith.xori %358, %arg13 : i32
                      %371:4 = scf.for %arg20 = %c3_i32 to %c10_i32 step %c1_i32_16 iter_args(%arg21 = %369, %arg22 = %364, %arg23 = %370, %arg24 = %arg16) -> (i32, i32, i32, i32)  : i32 {
                        %380 = arith.muli %arg21, %c-766435501_i32 : i32
                        %381 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %382 = ascsimt.math_binary %381, %360, "mulhi" : ui32, ui32 -> ui32
                        %383 = "ascsimt.bitcast"(%382) : (ui32) -> i32
                        %384 = arith.muli %arg23, %c-845247145_i32 : i32
                        %385 = "ascsimt.bitcast"(%arg23) : (i32) -> ui32
                        %386 = ascsimt.math_binary %385, %366, "mulhi" : ui32, ui32 -> ui32
                        %387 = "ascsimt.bitcast"(%386) : (ui32) -> i32
                        %388 = arith.muli %arg20, %c-1640531527_i32_18 : i32
                        %389 = arith.addi %arg14, %388 : i32
                        %390 = arith.muli %arg20, %c-1150833019_i32_17 : i32
                        %391 = arith.addi %arg15, %390 : i32
                        %392 = arith.xori %387, %arg22 : i32
                        %393 = arith.xori %392, %389 : i32
                        %394 = arith.xori %383, %arg24 : i32
                        %395 = arith.xori %394, %391 : i32
                        scf.yield %393, %384, %395, %380 : i32, i32, i32, i32
                      }
                      %372 = arith.muli %arg19, %c4_i64_15 : i64
                      %373 = arith.index_cast %372 : i64 to index
                      ascsimt.store %arg17[%373], %371#0 : memref<8192xi32, 1>, i32
                      %374 = arith.addi %372, %c1_i64_14 : i64
                      %375 = arith.index_cast %374 : i64 to index
                      ascsimt.store %arg17[%375], %371#1 : memref<8192xi32, 1>, i32
                      %376 = arith.addi %372, %c2_i64_13 : i64
                      %377 = arith.index_cast %376 : i64 to index
                      ascsimt.store %arg17[%377], %371#2 : memref<8192xi32, 1>, i32
                      %378 = arith.addi %372, %c3_i64_12 : i64
                      %379 = arith.index_cast %378 : i64 to index
                      ascsimt.store %arg17[%379], %371#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%245, %255, %270, %271, %233, %243, %alloc_9, %272 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
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
                    %c16384_i32 = arith.constant 16384 : i32
                    %287 = ascsimt.thread_idx {ascsimt.scope = 45 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %288 = arith.extsi %287 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg17 = %288 to %arg16 step %c256_i64_11  : i64 {
                      %289 = arith.trunci %arg17 : i64 to i32
                      %290 = arith.addi %arg9, %289 : i32
                      %291 = arith.cmpi ult, %290, %arg9 : i32
                      %292 = arith.extui %291 : i1 to i32
                      %293 = arith.addi %arg10, %292 : i32
                      %294 = arith.divui %290, %c16384_i32 : i32
                      %295 = arith.remui %293, %c16384_i32 : i32
                      %296 = arith.muli %295, %c262144_i32 : i32
                      %297 = arith.addi %294, %296 : i32
                      %298 = arith.divui %293, %c16384_i32 : i32
                      %299 = arith.addi %arg11, %297 : i32
                      %300 = arith.addi %arg12, %298 : i32
                      %301 = arith.cmpi ult, %299, %arg11 : i32
                      %302 = arith.extui %301 : i1 to i32
                      %303 = arith.addi %300, %302 : i32
                      %304 = arith.remui %290, %c16384_i32 : i32
                      %305:4 = scf.for %arg18 = %c0_i32_17 to %c10_i32 step %c1_i32_16 iter_args(%arg19 = %c0_i32_17, %arg20 = %304, %arg21 = %299, %arg22 = %303) -> (i32, i32, i32, i32)  : i32 {
                        %314 = arith.muli %arg19, %c-766435501_i32 : i32
                        %315 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %316 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %317 = ascsimt.math_binary %315, %316, "mulhi" : ui32, ui32 -> ui32
                        %318 = "ascsimt.bitcast"(%317) : (ui32) -> i32
                        %319 = arith.muli %arg21, %c-845247145_i32 : i32
                        %320 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %321 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %322 = ascsimt.math_binary %320, %321, "mulhi" : ui32, ui32 -> ui32
                        %323 = "ascsimt.bitcast"(%322) : (ui32) -> i32
                        %324 = arith.muli %arg18, %c-1640531527_i32_19 : i32
                        %325 = arith.addi %arg13, %324 : i32
                        %326 = arith.muli %arg18, %c-1150833019_i32_18 : i32
                        %327 = arith.addi %arg14, %326 : i32
                        %328 = arith.xori %323, %arg20 : i32
                        %329 = arith.xori %328, %325 : i32
                        %330 = arith.xori %318, %arg22 : i32
                        %331 = arith.xori %330, %327 : i32
                        scf.yield %329, %319, %331, %314 : i32, i32, i32, i32
                      }
                      %306 = arith.muli %arg17, %c4_i64_15 : i64
                      %307 = arith.index_cast %306 : i64 to index
                      ascsimt.store %arg15[%307], %305#0 : memref<8192xi32, 1>, i32
                      %308 = arith.addi %306, %c1_i64_14 : i64
                      %309 = arith.index_cast %308 : i64 to index
                      ascsimt.store %arg15[%309], %305#1 : memref<8192xi32, 1>, i32
                      %310 = arith.addi %306, %c2_i64_13 : i64
                      %311 = arith.index_cast %310 : i64 to index
                      ascsimt.store %arg15[%311], %305#2 : memref<8192xi32, 1>, i32
                      %312 = arith.addi %306, %c3_i64_12 : i64
                      %313 = arith.index_cast %312 : i64 to index
                      ascsimt.store %arg15[%313], %305#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %278 = arith.trunci %32 : i64 to i32
                %279 = arith.divsi %278, %c2_i32 : i32
                %280 = arith.trunci %279 : i32 to i8
                %281 = arith.trunci %224 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                %282 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<1024xui32, 1> -> memref<?xui16, 1>
                ascvec.vf(%280, %alloc_7, %134, %282, %alloc_10, %281 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui16, 1>, memref<2048xui16, 1>, i32) {
                ^bb0(%arg9: i8, %arg10: memref<2048xui8, 1>, %arg11: memref<32768xbf16, 1>, %arg12: memref<?xui16, 1>, %arg13: memref<2048xui16, 1>, %arg14: i32):
                  %c16_11 = arith.constant 16 : index
                  %c4_i32_12 = arith.constant 4 : i32
                  %c0_i32_13 = arith.constant 0 : i32
                  %c3_i32 = arith.constant 3 : i32
                  %c2_i32_14 = arith.constant 2 : i32
                  %c16_i32 = arith.constant 16 : i32
                  %c254_i16 = arith.constant 254 : i16
                  %c-15_i16 = arith.constant -15 : i16
                  %c15_i16 = arith.constant 15 : i16
                  %c255_i16 = arith.constant 255 : i16
                  %c256_i32 = arith.constant 256 : i32
                  %c-1_i16 = arith.constant -1 : i16
                  %c1_i32_15 = arith.constant 1 : i32
                  %c0_16 = arith.constant 0 : index
                  %c16_i8 = arith.constant 16 : i8
                  %c1_i8 = arith.constant 1 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c0_i16 = arith.constant 0 : i16
                  %c32767_i16 = arith.constant 32767 : i16
                  %c8_i32 = arith.constant 8 : i32
                  %287 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %288 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                  %289 = ascvec.duplicate_scalar %c32767_i16, %287 : i16 -> <ui16>
                  %290 = ascvec.arange %c0_i16 : i16 -> <ui16>
                  %291 = ascvec.shr %290, 4, %287 : <ui16> -> <ui16>
                  %292 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %293 = ascvec.duplicate_scalar %c1_i8, %288 : i8 -> <ui8>
                  %294 = ascvec.and %292, %293, %288 : <ui8>, <ui8> -> <ui8>
                  %295 = ascvec.cmp_lt_scalar %292, %c16_i8, %288 : <ui8>, i8 -> !ascvec.mask
                  %296 = ascvec.cmp_lt_scalar %294, %arg9, %288 : <ui8>, i8 -> !ascvec.mask
                  %297 = ascvec.mask_and %295, %296, %288
                  ascvec.store_unalign_begin %arg10 : memref<2048xui8, 1>
                  %mask_17, %remaining_out_18 = ascvec.update_mask %c1_i32_15 elem_bits 16 : !ascvec.mask
                  %298 = ascvec.duplicate_scalar %c-1_i16, %mask_17 : i16 -> <ui16>
                  %299 = scf.for %arg15 = %c0_i32_13 to %arg14 step %c4_i32_12 iter_args(%arg16 = %298) -> (!ascvec.vreg<ui16>)  : i32 {
                    %300 = ascvec.duplicate_scalar %c-1_i16, %287 : i16 -> <ui16>
                    %301 = ascvec.duplicate_scalar %c0_i16, %287 : i16 -> <ui16>
                    %302 = arith.muli %arg15, %c256_i32 : i32
                    %303 = arith.index_cast %302 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg11[%303] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %304 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %305 = ascvec.and %304, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %306 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %307 = ascvec.and %306, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %308 = ascvec.max %305, %307, %287 : <ui16>
                    %309 = ascvec.reduce_max_datablock %308, %287 : <ui16>
                    %310 = ascvec.shr %309, 7, %287 : <ui16> -> <ui16>
                    %311 = ascvec.cmp_ge_scalar %310, %c255_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %312 = ascvec.max_scalar %310, %c15_i16, %287 : <ui16>, i16
                    %313 = ascvec.add_scalar %312, %c-15_i16, %287 : <ui16>, i16
                    %314 = ascvec.duplicate_scalar %c255_i16, %311 merge %313 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %315 = ascvec.duplicate_scalar %c254_i16, %287 : i16 -> <ui16>
                    %316 = ascvec.sub %315, %314, %287 : <ui16>
                    %317 = ascvec.cmp_ne_scalar %314, %c0_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %318 = ascvec.shl %316, 7, %317 : <ui16> -> <ui16>
                    %319 = ascvec.max %301, %314, %mask : <ui16>
                    %320 = ascvec.gather_reg %318, %291 : <ui16>, <ui16> -> <ui16>
                    %321 = ascvec.reinterpret %320 : <ui16> -> <bf16>
                    %322 = ascvec.mul %result0, %321, %287 : <bf16>
                    %323 = ascvec.mul %result1, %321, %287 : <bf16>
                    ascvec.storealign_intlv %arg11[%303], %322, %323 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %324 = ascvec.reinterpret %322 : <bf16> -> <ui16>
                    %325 = ascvec.and %324, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %326 = ascvec.reinterpret %323 : <bf16> -> <ui16>
                    %327 = ascvec.and %326, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %328 = ascvec.min %325, %327, %287 : <ui16>
                    %329 = ascvec.min %300, %328, %287 : <ui16>
                    %330 = arith.muli %arg15, %c16_i32 : i32
                    %331 = arith.index_cast %330 : i32 to index
                    ascvec.storealign %arg12[%331], %314, %mask : memref<?xui16, 1>, index, <ui16>
                    %332 = ascvec.reinterpret %314 : <ui16> -> <ui8>
                    %333 = ascvec.squeeze_store_reg %332, %297 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_16], %333 : memref<2048xui8, 1>, <ui8>
                    %334 = arith.addi %arg15, %c1_i32_15 : i32
                    %335 = arith.muli %334, %c256_i32 : i32
                    %336 = arith.index_cast %335 : i32 to index
                    %result0_20, %result1_21 = ascvec.loadalign_deintlv %arg11[%336] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %337 = ascvec.reinterpret %result0_20 : <bf16> -> <ui16>
                    %338 = ascvec.and %337, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %339 = ascvec.reinterpret %result1_21 : <bf16> -> <ui16>
                    %340 = ascvec.and %339, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %341 = ascvec.max %338, %340, %287 : <ui16>
                    %342 = ascvec.reduce_max_datablock %341, %287 : <ui16>
                    %343 = ascvec.shr %342, 7, %287 : <ui16> -> <ui16>
                    %344 = ascvec.cmp_ge_scalar %343, %c255_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %345 = ascvec.max_scalar %343, %c15_i16, %287 : <ui16>, i16
                    %346 = ascvec.add_scalar %345, %c-15_i16, %287 : <ui16>, i16
                    %347 = ascvec.duplicate_scalar %c255_i16, %344 merge %346 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %348 = ascvec.sub %315, %347, %287 : <ui16>
                    %349 = ascvec.cmp_ne_scalar %347, %c0_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %350 = ascvec.shl %348, 7, %349 : <ui16> -> <ui16>
                    %351 = ascvec.max %319, %347, %mask : <ui16>
                    %352 = ascvec.gather_reg %350, %291 : <ui16>, <ui16> -> <ui16>
                    %353 = ascvec.reinterpret %352 : <ui16> -> <bf16>
                    %354 = ascvec.mul %result0_20, %353, %287 : <bf16>
                    %355 = ascvec.mul %result1_21, %353, %287 : <bf16>
                    ascvec.storealign_intlv %arg11[%336], %354, %355 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %356 = ascvec.reinterpret %354 : <bf16> -> <ui16>
                    %357 = ascvec.and %356, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %358 = ascvec.reinterpret %355 : <bf16> -> <ui16>
                    %359 = ascvec.and %358, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %360 = ascvec.min %357, %359, %287 : <ui16>
                    %361 = ascvec.min %329, %360, %287 : <ui16>
                    %362 = arith.muli %334, %c16_i32 : i32
                    %363 = arith.index_cast %362 : i32 to index
                    ascvec.storealign %arg12[%363], %347, %mask : memref<?xui16, 1>, index, <ui16>
                    %364 = ascvec.reinterpret %347 : <ui16> -> <ui8>
                    %365 = ascvec.squeeze_store_reg %364, %297 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_16], %365 : memref<2048xui8, 1>, <ui8>
                    %366 = arith.addi %arg15, %c2_i32_14 : i32
                    %367 = arith.muli %366, %c256_i32 : i32
                    %368 = arith.index_cast %367 : i32 to index
                    %result0_22, %result1_23 = ascvec.loadalign_deintlv %arg11[%368] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %369 = ascvec.reinterpret %result0_22 : <bf16> -> <ui16>
                    %370 = ascvec.and %369, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %371 = ascvec.reinterpret %result1_23 : <bf16> -> <ui16>
                    %372 = ascvec.and %371, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %373 = ascvec.max %370, %372, %287 : <ui16>
                    %374 = ascvec.reduce_max_datablock %373, %287 : <ui16>
                    %375 = ascvec.shr %374, 7, %287 : <ui16> -> <ui16>
                    %376 = ascvec.cmp_ge_scalar %375, %c255_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %377 = ascvec.max_scalar %375, %c15_i16, %287 : <ui16>, i16
                    %378 = ascvec.add_scalar %377, %c-15_i16, %287 : <ui16>, i16
                    %379 = ascvec.duplicate_scalar %c255_i16, %376 merge %378 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %380 = ascvec.sub %315, %379, %287 : <ui16>
                    %381 = ascvec.cmp_ne_scalar %379, %c0_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %382 = ascvec.shl %380, 7, %381 : <ui16> -> <ui16>
                    %383 = ascvec.max %351, %379, %mask : <ui16>
                    %384 = ascvec.gather_reg %382, %291 : <ui16>, <ui16> -> <ui16>
                    %385 = ascvec.reinterpret %384 : <ui16> -> <bf16>
                    %386 = ascvec.mul %result0_22, %385, %287 : <bf16>
                    %387 = ascvec.mul %result1_23, %385, %287 : <bf16>
                    ascvec.storealign_intlv %arg11[%368], %386, %387 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %388 = ascvec.reinterpret %386 : <bf16> -> <ui16>
                    %389 = ascvec.and %388, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %390 = ascvec.reinterpret %387 : <bf16> -> <ui16>
                    %391 = ascvec.and %390, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %392 = ascvec.min %389, %391, %287 : <ui16>
                    %393 = ascvec.min %361, %392, %287 : <ui16>
                    %394 = arith.muli %366, %c16_i32 : i32
                    %395 = arith.index_cast %394 : i32 to index
                    ascvec.storealign %arg12[%395], %379, %mask : memref<?xui16, 1>, index, <ui16>
                    %396 = ascvec.reinterpret %379 : <ui16> -> <ui8>
                    %397 = ascvec.squeeze_store_reg %396, %297 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_16], %397 : memref<2048xui8, 1>, <ui8>
                    %398 = arith.addi %arg15, %c3_i32 : i32
                    %399 = arith.muli %398, %c256_i32 : i32
                    %400 = arith.index_cast %399 : i32 to index
                    %result0_24, %result1_25 = ascvec.loadalign_deintlv %arg11[%400] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %401 = ascvec.reinterpret %result0_24 : <bf16> -> <ui16>
                    %402 = ascvec.and %401, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %403 = ascvec.reinterpret %result1_25 : <bf16> -> <ui16>
                    %404 = ascvec.and %403, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %405 = ascvec.max %402, %404, %287 : <ui16>
                    %406 = ascvec.reduce_max_datablock %405, %287 : <ui16>
                    %407 = ascvec.shr %406, 7, %287 : <ui16> -> <ui16>
                    %408 = ascvec.cmp_ge_scalar %407, %c255_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %409 = ascvec.max_scalar %407, %c15_i16, %287 : <ui16>, i16
                    %410 = ascvec.add_scalar %409, %c-15_i16, %287 : <ui16>, i16
                    %411 = ascvec.duplicate_scalar %c255_i16, %408 merge %410 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    %412 = ascvec.sub %315, %411, %287 : <ui16>
                    %413 = ascvec.cmp_ne_scalar %411, %c0_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %414 = ascvec.shl %412, 7, %413 : <ui16> -> <ui16>
                    %415 = ascvec.max %383, %411, %mask : <ui16>
                    %416 = ascvec.gather_reg %414, %291 : <ui16>, <ui16> -> <ui16>
                    %417 = ascvec.reinterpret %416 : <ui16> -> <bf16>
                    %418 = ascvec.mul %result0_24, %417, %287 : <bf16>
                    %419 = ascvec.mul %result1_25, %417, %287 : <bf16>
                    ascvec.storealign_intlv %arg11[%400], %418, %419 : memref<32768xbf16, 1>, <bf16>, <bf16>
                    %420 = ascvec.reinterpret %418 : <bf16> -> <ui16>
                    %421 = ascvec.and %420, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %422 = ascvec.reinterpret %419 : <bf16> -> <ui16>
                    %423 = ascvec.and %422, %289, %287 : <ui16>, <ui16> -> <ui16>
                    %424 = ascvec.min %421, %423, %287 : <ui16>
                    %425 = ascvec.min %393, %424, %287 : <ui16>
                    %426 = arith.muli %398, %c16_i32 : i32
                    %427 = arith.index_cast %426 : i32 to index
                    ascvec.storealign %arg12[%427], %411, %mask : memref<?xui16, 1>, index, <ui16>
                    %428 = ascvec.reinterpret %411 : <ui16> -> <ui8>
                    %429 = ascvec.squeeze_store_reg %428, %297 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_16], %429 : memref<2048xui8, 1>, <ui8>
                    %430 = ascvec.reduce_min %425, %287 : <ui16>
                    %431 = ascvec.reduce_max %415, %mask : <ui16>
                    %432 = ascvec.cmp_eq_scalar %431, %c255_i16, %287 : <ui16>, i16 -> !ascvec.mask
                    %433 = ascvec.duplicate_scalar %c0_i16, %432 merge %430 : !ascvec.vreg<ui16> {mode = "merging"} : i16 -> <ui16>
                    ascvec.storealign_1st %arg13[%331], %433 : memref<2048xui16, 1>, index, <ui16>
                    %434 = ascvec.min %arg16, %433, %mask_17 : <ui16>
                    scf.yield %434 : !ascvec.vreg<ui16>
                  }
                  %c0_19 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg15 = %c0_19 to %c1 step %c1 {
                    ascvec.store_unalign_post %arg10[%c0_16] : memref<2048xui8, 1>
                    ascvec.storealign_1st %arg13[%c16_11], %299 : memref<2048xui16, 1>, index, <ui16>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %283 = memref.load %alloc_10[%c16] : memref<2048xui16, 1>
                ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                %284 = cannir.scalar_storage_cast %283 : ui16 -> i16
                %285 = arith.extui %284 : i16 to i64
                %286 = arith.cmpi sge, %285, %c14464_i64 : i64
                scf.if %286 {
                  ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
                  ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  %287 = ascscalar.bitcast_memref %134 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                  ascvec.vf(%134, %alloc_9, %287, %alloc_6, %281 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32) {
                  ^bb0(%arg9: memref<32768xbf16, 1>, %arg10: memref<8192xi32, 1>, %arg11: memref<?xui8, 1>, %arg12: memref<32768xui8, 1>, %arg13: i32):
                    %c512_i32 = arith.constant 512 : i32
                    %c1_i32_11 = arith.constant 1 : i32
                    %c0_i32_12 = arith.constant 0 : i32
                    %c123_i8 = arith.constant 123 : i8
                    %c64_i8 = arith.constant 64 : i8
                    %c64_i32 = arith.constant 64 : i32
                    %c256_i32 = arith.constant 256 : i32
                    %c1_i16 = arith.constant 1 : i16
                    %c0_i16 = arith.constant 0 : i16
                    %c-128_i8 = arith.constant -128 : i8
                    %c255_i16 = arith.constant 255 : i16
                    %288 = ascvec.create_mask : !ascvec.mask
                    %289 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                    %290 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                    %291 = ascvec.duplicate_scalar %c255_i16, %289 : i16 -> <ui16>
                    %292 = ascvec.duplicate_scalar %c-128_i8, %290 : i8 -> <ui8>
                    %293 = ascvec.arange %c0_i16 : i16 -> <ui16>
                    %294 = ascvec.duplicate_scalar %c1_i16, %289 : i16 -> <ui16>
                    %295 = ascvec.xor %293, %294, %289 : <ui16>, <ui16> -> <ui16>
                    scf.for %arg14 = %c0_i32_12 to %arg13 step %c1_i32_11  : i32 {
                      %296 = arith.muli %arg14, %c256_i32 : i32
                      %297 = arith.index_cast %296 : i32 to index
                      %result0, %result1 = ascvec.loadalign_deintlv %arg9[%297] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                      %298 = arith.muli %arg14, %c64_i32 : i32
                      %299 = arith.index_cast %298 : i32 to index
                      %300 = ascvec.loadalign %arg10[%299] : memref<8192xi32, 1>, index -> <i32>
                      %301 = ascvec.reinterpret %300 : <i32> -> <ui16>
                      %302 = ascvec.reinterpret %301 : <ui16> -> <ui32>
                      %303 = ascvec.shl %302, 8, %288 : <ui32> -> <ui32>
                      %304 = ascvec.shr %302, 24, %288 : <ui32> -> <ui32>
                      %305 = ascvec.or %303, %304, %288 : <ui32>, <ui32> -> <ui32>
                      %306 = ascvec.reinterpret %305 : <ui32> -> <ui16>
                      %307 = ascvec.gather_reg %301, %295 : <ui16>, <ui16> -> <ui16>
                      %308 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                      %309 = ascvec.shr %307, 11, %289 : <ui16> -> <ui16>
                      %310 = ascvec.add %308, %309, %289 : <ui16>
                      %311 = ascvec.shr %310, 5, %289 : <ui16> -> <ui16>
                      %312 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                      %313 = ascvec.shr %306, 11, %289 : <ui16> -> <ui16>
                      %314 = ascvec.add %312, %313, %289 : <ui16>
                      %315 = ascvec.shr %314, 5, %289 : <ui16> -> <ui16>
                      %316 = ascvec.and %311, %291, %289 : <ui16>, <ui16> -> <ui16>
                      %317 = ascvec.shl %315, 8, %289 : <ui16> -> <ui16>
                      %318 = ascvec.or %316, %317, %289 : <ui16>, <ui16> -> <ui16>
                      %319 = ascvec.reinterpret %318 : <ui16> -> <ui8>
                      %320 = ascvec.add_scalar %319, %c64_i8, %290 : <ui8>, i8
                      %321 = ascvec.min_scalar %320, %c123_i8, %290 : <ui8>, i8
                      %322 = arith.muli %arg14, %c512_i32 : i32
                      %323 = arith.index_cast %322 : i32 to index
                      %result0_13, %result1_14 = ascvec.loadalign_deintlv %arg11[%323] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                      %324 = ascvec.and %result1_14, %292, %290 : <ui8>, <ui8> -> <ui8>
                      %325 = ascvec.or %321, %324, %290 : <ui8>, <ui8> -> <ui8>
                      ascvec.storealign %arg12[%297], %325, %290 : memref<32768xui8, 1>, index, <ui8>
                    }
                    ascvec.mem_bar vst_vld
                  } {cannir.vf_mode = "simd"}
                  ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                  ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
                } else {
                  scf.for %arg9 = %c0_i64 to %224 step %c16_i64  : i64 {
                    %287 = arith.addi %arg9, %c16_i64 : i64
                    %288 = arith.cmpi sgt, %287, %224 : i64
                    %289 = arith.select %288, %224, %287 : i64
                    %290 = arith.muli %arg9, %c16_i64 : i64
                    ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %291 = arith.index_cast %290 : i64 to index
                    %292 = memref.load %alloc_10[%291] : memref<2048xui16, 1>
                    ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                    %293 = cannir.scalar_storage_cast %292 : ui16 -> i16
                    %294 = arith.extui %293 : i16 to i64
                    %295 = arith.addi %arg9, %c4_i64 : i64
                    %296 = scf.for %arg10 = %295 to %289 step %c4_i64 iter_args(%arg11 = %294) -> (i64)  : i64 {
                      %298 = arith.muli %arg10, %c16_i64 : i64
                      ascvec.get_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %299 = arith.index_cast %298 : i64 to index
                      %300 = memref.load %alloc_10[%299] : memref<2048xui16, 1>
                      ascvec.release_buf s, %c4_i64, 0 {cannir.buffer.generated_sync}
                      %301 = cannir.scalar_storage_cast %300 : ui16 -> i16
                      %302 = arith.extui %301 : i16 to i64
                      %303 = arith.cmpi slt, %302, %arg11 : i64
                      %304 = arith.select %303, %302, %arg11 : i64
                      scf.yield %304 : i64
                    }
                    %297 = arith.cmpi sge, %296, %c14464_i64 : i64
                    scf.if %297 {
                      %298 = arith.trunci %arg9 : i64 to i32
                      %299 = arith.trunci %289 : i64 to i32
                      ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
                      ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      %300 = ascscalar.bitcast_memref %134 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                      ascvec.vf(%134, %alloc_9, %300, %alloc_6, %298, %299 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                      ^bb0(%arg10: memref<32768xbf16, 1>, %arg11: memref<8192xi32, 1>, %arg12: memref<?xui8, 1>, %arg13: memref<32768xui8, 1>, %arg14: i32, %arg15: i32):
                        %c512_i32 = arith.constant 512 : i32
                        %c1_i32_11 = arith.constant 1 : i32
                        %c123_i8 = arith.constant 123 : i8
                        %c64_i8 = arith.constant 64 : i8
                        %c64_i32 = arith.constant 64 : i32
                        %c256_i32 = arith.constant 256 : i32
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
                        scf.for %arg16 = %arg14 to %arg15 step %c1_i32_11  : i32 {
                          %309 = arith.muli %arg16, %c256_i32 : i32
                          %310 = arith.index_cast %309 : i32 to index
                          %result0, %result1 = ascvec.loadalign_deintlv %arg10[%310] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                          %311 = arith.muli %arg16, %c64_i32 : i32
                          %312 = arith.index_cast %311 : i32 to index
                          %313 = ascvec.loadalign %arg11[%312] : memref<8192xi32, 1>, index -> <i32>
                          %314 = ascvec.reinterpret %313 : <i32> -> <ui16>
                          %315 = ascvec.reinterpret %314 : <ui16> -> <ui32>
                          %316 = ascvec.shl %315, 8, %301 : <ui32> -> <ui32>
                          %317 = ascvec.shr %315, 24, %301 : <ui32> -> <ui32>
                          %318 = ascvec.or %316, %317, %301 : <ui32>, <ui32> -> <ui32>
                          %319 = ascvec.reinterpret %318 : <ui32> -> <ui16>
                          %320 = ascvec.gather_reg %314, %308 : <ui16>, <ui16> -> <ui16>
                          %321 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                          %322 = ascvec.shr %320, 11, %302 : <ui16> -> <ui16>
                          %323 = ascvec.add %321, %322, %302 : <ui16>
                          %324 = ascvec.shr %323, 5, %302 : <ui16> -> <ui16>
                          %325 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                          %326 = ascvec.shr %319, 11, %302 : <ui16> -> <ui16>
                          %327 = ascvec.add %325, %326, %302 : <ui16>
                          %328 = ascvec.shr %327, 5, %302 : <ui16> -> <ui16>
                          %329 = ascvec.and %324, %304, %302 : <ui16>, <ui16> -> <ui16>
                          %330 = ascvec.shl %328, 8, %302 : <ui16> -> <ui16>
                          %331 = ascvec.or %329, %330, %302 : <ui16>, <ui16> -> <ui16>
                          %332 = ascvec.reinterpret %331 : <ui16> -> <ui8>
                          %333 = ascvec.add_scalar %332, %c64_i8, %303 : <ui8>, i8
                          %334 = ascvec.min_scalar %333, %c123_i8, %303 : <ui8>, i8
                          %335 = arith.muli %arg16, %c512_i32 : i32
                          %336 = arith.index_cast %335 : i32 to index
                          %result0_12, %result1_13 = ascvec.loadalign_deintlv %arg12[%336] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                          %337 = ascvec.and %result1_13, %305, %303 : <ui8>, <ui8> -> <ui8>
                          %338 = ascvec.or %334, %337, %303 : <ui8>, <ui8> -> <ui8>
                          ascvec.storealign %arg13[%310], %338, %303 : memref<32768xui8, 1>, index, <ui8>
                        }
                        ascvec.mem_bar vst_vld
                      } {cannir.vf_mode = "simd"}
                      ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                      ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
                    } else {
                      %298 = arith.cmpi sge, %296, %c13568_i64 : i64
                      scf.if %298 {
                        %299 = arith.trunci %arg9 : i64 to i32
                        %300 = arith.trunci %289 : i64 to i32
                        ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %301 = ascscalar.bitcast_memref %134 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%134, %alloc_9, %301, %alloc_6, %299, %300 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg10: memref<32768xbf16, 1>, %arg11: memref<8192xi32, 1>, %arg12: memref<?xui8, 1>, %arg13: memref<32768xui8, 1>, %arg14: i32, %arg15: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_11 = arith.constant 1 : i32
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
                          %302 = ascvec.create_mask : !ascvec.mask
                          %303 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %304 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %305 = ascvec.duplicate_scalar %c255_i16, %303 : i16 -> <ui16>
                          %306 = ascvec.duplicate_scalar %c-128_i8, %304 : i8 -> <ui8>
                          %307 = ascvec.arange %c0_i16 : i16 -> <ui16>
                          %308 = ascvec.duplicate_scalar %c1_i16, %303 : i16 -> <ui16>
                          %309 = ascvec.xor %307, %308, %303 : <ui16>, <ui16> -> <ui16>
                          %310 = ascvec.duplicate_scalar %c32767_i16, %303 : i16 -> <ui16>
                          %311 = ascvec.duplicate_scalar %cst, %303 : bf16 -> <bf16>
                          scf.for %arg16 = %arg14 to %arg15 step %c1_i32_11  : i32 {
                            %312 = arith.muli %arg16, %c256_i32 : i32
                            %313 = arith.index_cast %312 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg10[%313] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %314 = arith.muli %arg16, %c64_i32 : i32
                            %315 = arith.index_cast %314 : i32 to index
                            %316 = ascvec.loadalign %arg11[%315] : memref<8192xi32, 1>, index -> <i32>
                            %317 = ascvec.reinterpret %316 : <i32> -> <ui16>
                            %318 = ascvec.reinterpret %317 : <ui16> -> <ui32>
                            %319 = ascvec.shl %318, 8, %302 : <ui32> -> <ui32>
                            %320 = ascvec.shr %318, 24, %302 : <ui32> -> <ui32>
                            %321 = ascvec.or %319, %320, %302 : <ui32>, <ui32> -> <ui32>
                            %322 = ascvec.reinterpret %321 : <ui32> -> <ui16>
                            %323 = ascvec.gather_reg %317, %309 : <ui16>, <ui16> -> <ui16>
                            %324 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %325 = ascvec.and %324, %310, %303 : <ui16>, <ui16> -> <ui16>
                            %326 = ascvec.cmp_lt_scalar %325, %c14464_i16, %303 : <ui16>, i16 -> !ascvec.mask
                            %327 = ascvec.shr %323, 11, %303 : <ui16> -> <ui16>
                            %328 = ascvec.add %325, %327, %303 : <ui16>
                            %329 = ascvec.shr %328, 5, %303 : <ui16> -> <ui16>
                            %330 = ascvec.reinterpret %325 : <ui16> -> <bf16>
                            %331 = ascvec.mul %330, %311, %326 : <bf16>
                            %332 = ascvec.cast %331, %303 {rounding = 4 : i32} : <bf16> -> <f16>
                            %333 = ascvec.cast %332, %303 {rounding = 4 : i32} : <f16> -> <i16>
                            %334 = ascvec.reinterpret %333 : <i16> -> <ui16>
                            %335 = ascvec.shr %323, 4, %303 : <ui16> -> <ui16>
                            %336 = ascvec.add %334, %335, %303 : <ui16>
                            %337 = ascvec.shr %336, 12, %303 : <ui16> -> <ui16>
                            %338 = ascvec.add_scalar %337, %c448_i16, %303 : <ui16>, i16
                            %339 = ascvec.select %338, %329, %326 : <ui16>
                            %340 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %341 = ascvec.and %340, %310, %303 : <ui16>, <ui16> -> <ui16>
                            %342 = ascvec.cmp_lt_scalar %341, %c14464_i16, %303 : <ui16>, i16 -> !ascvec.mask
                            %343 = ascvec.shr %322, 11, %303 : <ui16> -> <ui16>
                            %344 = ascvec.add %341, %343, %303 : <ui16>
                            %345 = ascvec.shr %344, 5, %303 : <ui16> -> <ui16>
                            %346 = ascvec.reinterpret %341 : <ui16> -> <bf16>
                            %347 = ascvec.mul %346, %311, %342 : <bf16>
                            %348 = ascvec.cast %347, %303 {rounding = 4 : i32} : <bf16> -> <f16>
                            %349 = ascvec.cast %348, %303 {rounding = 4 : i32} : <f16> -> <i16>
                            %350 = ascvec.reinterpret %349 : <i16> -> <ui16>
                            %351 = ascvec.shr %322, 4, %303 : <ui16> -> <ui16>
                            %352 = ascvec.add %350, %351, %303 : <ui16>
                            %353 = ascvec.shr %352, 12, %303 : <ui16> -> <ui16>
                            %354 = ascvec.add_scalar %353, %c448_i16, %303 : <ui16>, i16
                            %355 = ascvec.select %354, %345, %342 : <ui16>
                            %356 = ascvec.and %339, %305, %303 : <ui16>, <ui16> -> <ui16>
                            %357 = ascvec.shl %355, 8, %303 : <ui16> -> <ui16>
                            %358 = ascvec.or %356, %357, %303 : <ui16>, <ui16> -> <ui16>
                            %359 = ascvec.reinterpret %358 : <ui16> -> <ui8>
                            %360 = ascvec.add_scalar %359, %c64_i8, %304 : <ui8>, i8
                            %361 = ascvec.min_scalar %360, %c123_i8, %304 : <ui8>, i8
                            %362 = arith.muli %arg16, %c512_i32 : i32
                            %363 = arith.index_cast %362 : i32 to index
                            %result0_12, %result1_13 = ascvec.loadalign_deintlv %arg12[%363] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %364 = ascvec.and %result1_13, %306, %304 : <ui8>, <ui8> -> <ui8>
                            %365 = ascvec.or %361, %364, %304 : <ui8>, <ui8> -> <ui8>
                            ascvec.storealign %arg13[%313], %365, %304 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
                      } else {
                        %299 = arith.trunci %arg9 : i64 to i32
                        %300 = arith.trunci %289 : i64 to i32
                        ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
                        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        %301 = ascscalar.bitcast_memref %134 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                        ascvec.vf(%134, %alloc_9, %301, %282, %alloc_6, %299, %300 : memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<?xui8, 1>, memref<?xui16, 1>, memref<32768xui8, 1>, i32, i32) {
                        ^bb0(%arg10: memref<32768xbf16, 1>, %arg11: memref<8192xi32, 1>, %arg12: memref<?xui8, 1>, %arg13: memref<?xui16, 1>, %arg14: memref<32768xui8, 1>, %arg15: i32, %arg16: i32):
                          %c512_i32 = arith.constant 512 : i32
                          %c1_i32_11 = arith.constant 1 : i32
                          %c127_i8 = arith.constant 127 : i8
                          %c16_i32 = arith.constant 16 : i32
                          %c0_i16 = arith.constant 0 : i16
                          %c123_i8 = arith.constant 123 : i8
                          %c-1_i8 = arith.constant -1 : i8
                          %c60_i8 = arith.constant 60 : i8
                          %cst = arith.constant 6.10351526E-5 : f32
                          %c64_i32 = arith.constant 64 : i32
                          %c32767_i16 = arith.constant 32767 : i16
                          %c0_12 = arith.constant 0 : index
                          %c256_i32 = arith.constant 256 : i32
                          %c65535_i32 = arith.constant 65535 : i32
                          %c3_i8 = arith.constant 3 : i8
                          %c-128_i8 = arith.constant -128 : i8
                          %c0_i8 = arith.constant 0 : i8
                          %c8_i32 = arith.constant 8 : i32
                          %302 = ascvec.create_mask : !ascvec.mask
                          %303 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                          %304 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                          %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 16 : !ascvec.mask
                          %305 = ascvec.arange %c0_i8 : i8 -> <ui8>
                          %306 = ascvec.duplicate_scalar %c-128_i8, %304 : i8 -> <ui8>
                          %307 = ascvec.shr %305, 5, %304 : <ui8> -> <ui8>
                          %308 = ascvec.duplicate_scalar %c3_i8, %304 : i8 -> <ui8>
                          %309 = ascvec.and %305, %308, %304 : <ui8>, <ui8> -> <ui8>
                          %310 = ascvec.shl %309, 6, %304 : <ui8> -> <ui8>
                          %311 = ascvec.shr %305, 2, %304 : <ui8> -> <ui8>
                          %312 = ascvec.or %310, %311, %304 : <ui8>, <ui8> -> <ui8>
                          %313 = ascvec.duplicate_scalar %c65535_i32, %302 : i32 -> <ui32>
                          scf.for %arg17 = %arg15 to %arg16 step %c1_i32_11  : i32 {
                            %314 = arith.muli %arg17, %c256_i32 : i32
                            %315 = arith.index_cast %314 : i32 to index
                            %result0, %result1 = ascvec.loadalign_deintlv %arg10[%315] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                            %316 = ascvec.duplicate_scalar %c32767_i16, %303 : i16 -> <ui16>
                            %317 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                            %318 = ascvec.and %317, %316, %303 : <ui16>, <ui16> -> <ui16>
                            %319 = ascvec.reinterpret %318 : <ui16> -> <bf16>
                            %320 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                            %321 = ascvec.and %320, %316, %303 : <ui16>, <ui16> -> <ui16>
                            %322 = ascvec.reinterpret %321 : <ui16> -> <bf16>
                            %323 = ascvec.cast %319, %303 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %324 = ascvec.cast %319, %303 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %325 = ascvec.cast %322, %303 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %326 = ascvec.cast %322, %303 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                            %327 = arith.muli %arg17, %c64_i32 : i32
                            %328 = arith.index_cast %327 : i32 to index
                            %329 = ascvec.loadalign %arg11[%328] : memref<8192xi32, 1>, index -> <i32>
                            %330 = ascvec.reinterpret %329 : <i32> -> <ui32>
                            %331 = ascvec.shl %330, 8, %302 : <ui32> -> <ui32>
                            %332 = ascvec.shr %330, 24, %302 : <ui32> -> <ui32>
                            %333 = ascvec.or %331, %332, %302 : <ui32>, <ui32> -> <ui32>
                            %334 = ascvec.shr %330, 16, %302 : <ui32> -> <ui32>
                            %335 = ascvec.and %333, %313, %302 : <ui32>, <ui32> -> <ui32>
                            %336 = ascvec.and %330, %313, %302 : <ui32>, <ui32> -> <ui32>
                            %337 = ascvec.shr %333, 16, %302 : <ui32> -> <ui32>
                            %338 = ascvec.reinterpret %323 : <f32> -> <ui32>
                            %339 = ascvec.reinterpret %338 : <ui32> -> <f32>
                            %340 = ascvec.max_scalar %339, %cst, %302 : <f32>, f32
                            %341 = ascvec.add %339, %340, %302 : <f32>
                            %342 = ascvec.reinterpret %341 : <f32> -> <ui32>
                            %343 = ascvec.shr %342, 5, %302 : <ui32> -> <ui32>
                            %344 = ascvec.add %343, %334, %302 : <ui32>
                            %345 = ascvec.reinterpret %325 : <f32> -> <ui32>
                            %346 = ascvec.reinterpret %345 : <ui32> -> <f32>
                            %347 = ascvec.max_scalar %346, %cst, %302 : <f32>, f32
                            %348 = ascvec.add %346, %347, %302 : <f32>
                            %349 = ascvec.reinterpret %348 : <f32> -> <ui32>
                            %350 = ascvec.shr %349, 5, %302 : <ui32> -> <ui32>
                            %351 = ascvec.add %350, %335, %302 : <ui32>
                            %352 = ascvec.reinterpret %324 : <f32> -> <ui32>
                            %353 = ascvec.reinterpret %352 : <ui32> -> <f32>
                            %354 = ascvec.max_scalar %353, %cst, %302 : <f32>, f32
                            %355 = ascvec.add %353, %354, %302 : <f32>
                            %356 = ascvec.reinterpret %355 : <f32> -> <ui32>
                            %357 = ascvec.shr %356, 5, %302 : <ui32> -> <ui32>
                            %358 = ascvec.add %357, %336, %302 : <ui32>
                            %359 = ascvec.reinterpret %326 : <f32> -> <ui32>
                            %360 = ascvec.reinterpret %359 : <ui32> -> <f32>
                            %361 = ascvec.max_scalar %360, %cst, %302 : <f32>, f32
                            %362 = ascvec.add %360, %361, %302 : <f32>
                            %363 = ascvec.reinterpret %362 : <f32> -> <ui32>
                            %364 = ascvec.shr %363, 5, %302 : <ui32> -> <ui32>
                            %365 = ascvec.add %364, %337, %302 : <ui32>
                            %366 = ascvec.reinterpret %344 : <ui32> -> <ui16>
                            %367 = ascvec.reinterpret %351 : <ui32> -> <ui16>
                            %result0_13, %result1_14 = ascvec.deintlv %366, %367 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %368 = ascvec.reinterpret %358 : <ui32> -> <ui16>
                            %369 = ascvec.reinterpret %365 : <ui32> -> <ui16>
                            %result0_15, %result1_16 = ascvec.deintlv %368, %369 : <ui16>, <ui16> -> <ui16>, <ui16>
                            %370 = ascvec.reinterpret %result1_14 : <ui16> -> <ui8>
                            %371 = ascvec.reinterpret %result1_16 : <ui16> -> <ui8>
                            %result0_17, %result1_18 = ascvec.deintlv %370, %371 : <ui8>, <ui8> -> <ui8>, <ui8>
                            %372 = ascvec.gather_reg %result0_17, %312 : <ui8>, <ui8> -> <ui8>
                            %373 = ascvec.add_scalar %372, %c60_i8, %304 : <ui8>, i8
                            %374 = ascvec.cmp_ne_scalar %373, %c-1_i8, %304 : <ui8>, i8 -> !ascvec.mask
                            %375 = ascvec.min_scalar %373, %c123_i8, %374 : <ui8>, i8
                            %376 = arith.muli %arg17, %c512_i32 : i32
                            %377 = arith.index_cast %376 : i32 to index
                            %result0_19, %result1_20 = ascvec.loadalign_deintlv %arg12[%377] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                            %378 = ascvec.and %result1_20, %306, %304 : <ui8>, <ui8> -> <ui8>
                            %379 = ascvec.or %375, %378, %304 : <ui8>, <ui8> -> <ui8>
                            %380 = ascvec.arange %c0_i16 : i16 -> <ui16>
                            %381 = arith.muli %arg17, %c16_i32 : i32
                            %382 = arith.trunci %381 : i32 to i16
                            %383 = ascvec.add_scalar %380, %382, %303 : <ui16>, i16
                            %384 = ascvec.gather %arg13[%c0_12], %383, %mask : memref<?xui16, 1>, <ui16>, !ascvec.mask -> <ui16>
                            %385 = ascvec.pack %384 src_elem_bits 16 dst_elem_bits 8 {part = "low"} : <ui16> -> <ui8>
                            %386 = ascvec.gather_reg %385, %307 : <ui8>, <ui8> -> <ui8>
                            %387 = ascvec.cmp_eq_scalar %386, %c-1_i8, %304 : <ui8>, i8 -> !ascvec.mask
                            %388 = ascvec.duplicate_scalar %c127_i8, %387 merge %379 : !ascvec.vreg<ui8> {mode = "merging"} : i8 -> <ui8>
                            ascvec.storealign %arg14[%315], %388, %304 : memref<32768xui8, 1>, index, <ui8>
                          }
                          ascvec.mem_bar vst_vld
                        } {cannir.vf_mode = "simd"}
                        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                        ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
                      }
                    }
                  }
                }
              } else {
                %224 = arith.divsi %99, %c256_i64 : i64
                %225 = arith.addi %99, %c1023_i64 : i64
                %226 = arith.remsi %225, %c1024_i64 : i64
                %227 = arith.cmpi ne, %226, %c0_i64 : i64
                %228 = arith.cmpi slt, %226, %c0_i64 : i64
                %229 = arith.andi %227, %228 : i1
                %230 = arith.divsi %225, %c1024_i64 : i64
                %231 = arith.extui %229 : i1 to i64
                %232 = arith.subi %230, %231 : i64
                %233 = arith.trunci %1 : i64 to i32
                %234 = arith.extui %233 : i32 to i64
                %235 = arith.subi %1, %234 : i64
                %236 = arith.remsi %235, %c4294967296_i64 : i64
                %237 = arith.cmpi ne, %236, %c0_i64 : i64
                %238 = arith.cmpi slt, %236, %c0_i64 : i64
                %239 = arith.andi %237, %238 : i1
                %240 = arith.divsi %235, %c4294967296_i64 : i64
                %241 = arith.extui %239 : i1 to i64
                %242 = arith.subi %240, %241 : i64
                %243 = arith.trunci %242 : i64 to i32
                %244 = arith.divui %212, %c16_i64 : i64
                %245 = arith.trunci %244 : i64 to i32
                %246 = arith.extui %245 : i32 to i64
                %247 = arith.subi %244, %246 : i64
                %248 = arith.remsi %247, %c4294967296_i64 : i64
                %249 = arith.cmpi ne, %248, %c0_i64 : i64
                %250 = arith.cmpi slt, %248, %c0_i64 : i64
                %251 = arith.andi %249, %250 : i1
                %252 = arith.divsi %247, %c4294967296_i64 : i64
                %253 = arith.extui %251 : i1 to i64
                %254 = arith.subi %252, %253 : i64
                %255 = arith.trunci %254 : i64 to i32
                %256 = arith.trunci %4 : i64 to i32
                %257 = arith.extui %256 : i32 to i64
                %258 = arith.subi %4, %257 : i64
                %259 = arith.remsi %258, %c4294967296_i64 : i64
                %260 = arith.cmpi ne, %259, %c0_i64 : i64
                %261 = arith.cmpi slt, %259, %c0_i64 : i64
                %262 = arith.andi %260, %261 : i1
                %263 = arith.divsi %258, %c4294967296_i64 : i64
                %264 = arith.extui %262 : i1 to i64
                %265 = arith.subi %263, %264 : i64
                %266 = arith.trunci %265 : i64 to i32
                %267 = arith.divui %256, %c4_i32 : i32
                %268 = arith.remui %266, %c4_i32 : i32
                %269 = arith.muli %268, %c1073741824_i32 : i32
                %270 = arith.addi %267, %269 : i32
                %271 = arith.divui %266, %c4_i32 : i32
                %272 = arith.muli %232, %c64_i64 : i64
                %273 = arith.divui %244, %c16384_i64 : i64
                %274 = arith.addi %244, %272 : i64
                %275 = arith.subi %274, %c1_i64 : i64
                %276 = arith.divui %275, %c16384_i64 : i64
                %277 = arith.cmpi eq, %273, %276 : i64
                scf.if %277 {
                  %283 = arith.trunci %273 : i64 to i32
                  %284 = arith.extui %283 : i32 to i64
                  %285 = arith.subi %273, %284 : i64
                  %286 = arith.remsi %285, %c4294967296_i64 : i64
                  %287 = arith.cmpi ne, %286, %c0_i64 : i64
                  %288 = arith.cmpi slt, %286, %c0_i64 : i64
                  %289 = arith.andi %287, %288 : i1
                  %290 = arith.divsi %285, %c4294967296_i64 : i64
                  %291 = arith.extui %289 : i1 to i64
                  %292 = arith.subi %290, %291 : i64
                  %293 = arith.trunci %292 : i64 to i32
                  %294 = arith.addi %270, %283 : i32
                  %295 = arith.addi %271, %293 : i32
                  %296 = arith.cmpi ult, %294, %270 : i32
                  %297 = arith.extui %296 : i1 to i32
                  %298 = arith.addi %295, %297 : i32
                  %299 = arith.extui %294 : i32 to i64
                  %300 = arith.muli %299, %c3449720151_i64 : i64
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
                  %312 = arith.xori %298, %243 : i32
                  %313 = arith.extui %312 : i32 to i64
                  %314 = arith.muli %313, %c3449720151_i64 : i64
                  %315 = arith.trunci %314 : i64 to i32
                  %316 = arith.extui %315 : i32 to i64
                  %317 = arith.subi %314, %316 : i64
                  %318 = arith.remsi %317, %c4294967296_i64 : i64
                  %319 = arith.cmpi ne, %318, %c0_i64 : i64
                  %320 = arith.cmpi slt, %318, %c0_i64 : i64
                  %321 = arith.andi %319, %320 : i1
                  %322 = arith.divsi %317, %c4294967296_i64 : i64
                  %323 = arith.extui %321 : i1 to i64
                  %324 = arith.subi %322, %323 : i64
                  %325 = arith.trunci %324 : i64 to i32
                  %326 = arith.addi %233, %c-1640531527_i32 : i32
                  %327 = arith.addi %243, %c-1150833019_i32 : i32
                  %328 = arith.xori %325, %301 : i32
                  %329 = arith.xori %328, %326 : i32
                  %330 = arith.extui %329 : i32 to i64
                  %331 = arith.muli %330, %c3528531795_i64 : i64
                  %332 = arith.trunci %331 : i64 to i32
                  %333 = arith.extui %332 : i32 to i64
                  %334 = arith.subi %331, %333 : i64
                  %335 = arith.remsi %334, %c4294967296_i64 : i64
                  %336 = arith.cmpi ne, %335, %c0_i64 : i64
                  %337 = arith.cmpi slt, %335, %c0_i64 : i64
                  %338 = arith.andi %336, %337 : i1
                  %339 = arith.divsi %334, %c4294967296_i64 : i64
                  %340 = arith.extui %338 : i1 to i64
                  %341 = arith.subi %339, %340 : i64
                  %342 = arith.trunci %341 : i64 to i32
                  %343 = arith.xori %311, %233 : i32
                  %344 = arith.addi %233, %c1013904242_i32 : i32
                  %345 = arith.xori %315, %344 : i32
                  %346 = arith.addi %243, %c1993301258_i32 : i32
                  %347 = arith.xori %342, %346 : i32
                  ascsimt.vf(%245, %343, %327, %345, %347, %233, %243, %332, %alloc_9, %272 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: memref<8192xi32, 1>, %arg18: i64):
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
                    %c16384_i32 = arith.constant 16384 : i32
                    %348 = ascsimt.thread_idx {ascsimt.scope = 51 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %349 = arith.extsi %348 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg19 = %349 to %arg18 step %c256_i64_11  : i64 {
                      %350 = arith.trunci %arg19 : i64 to i32
                      %351 = arith.addi %arg9, %350 : i32
                      %352 = arith.remui %351, %c16384_i32 : i32
                      %353 = arith.xori %352, %arg10 : i32
                      %354 = arith.muli %353, %c-766435501_i32 : i32
                      %355 = "ascsimt.bitcast"(%353) : (i32) -> ui32
                      %356 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %357 = ascsimt.math_binary %355, %356, "mulhi" : ui32, ui32 -> ui32
                      %358 = "ascsimt.bitcast"(%357) : (ui32) -> i32
                      %359 = arith.xori %358, %arg11 : i32
                      %360 = arith.muli %359, %c-845247145_i32 : i32
                      %361 = "ascsimt.bitcast"(%359) : (i32) -> ui32
                      %362 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %363 = ascsimt.math_binary %361, %362, "mulhi" : ui32, ui32 -> ui32
                      %364 = "ascsimt.bitcast"(%363) : (ui32) -> i32
                      %365 = arith.xori %364, %arg12 : i32
                      %366 = arith.xori %354, %arg13 : i32
                      %367:4 = scf.for %arg20 = %c3_i32 to %c10_i32 step %c1_i32_16 iter_args(%arg21 = %365, %arg22 = %360, %arg23 = %366, %arg24 = %arg16) -> (i32, i32, i32, i32)  : i32 {
                        %376 = arith.muli %arg21, %c-766435501_i32 : i32
                        %377 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %378 = ascsimt.math_binary %377, %356, "mulhi" : ui32, ui32 -> ui32
                        %379 = "ascsimt.bitcast"(%378) : (ui32) -> i32
                        %380 = arith.muli %arg23, %c-845247145_i32 : i32
                        %381 = "ascsimt.bitcast"(%arg23) : (i32) -> ui32
                        %382 = ascsimt.math_binary %381, %362, "mulhi" : ui32, ui32 -> ui32
                        %383 = "ascsimt.bitcast"(%382) : (ui32) -> i32
                        %384 = arith.muli %arg20, %c-1640531527_i32_18 : i32
                        %385 = arith.addi %arg14, %384 : i32
                        %386 = arith.muli %arg20, %c-1150833019_i32_17 : i32
                        %387 = arith.addi %arg15, %386 : i32
                        %388 = arith.xori %383, %arg22 : i32
                        %389 = arith.xori %388, %385 : i32
                        %390 = arith.xori %379, %arg24 : i32
                        %391 = arith.xori %390, %387 : i32
                        scf.yield %389, %380, %391, %376 : i32, i32, i32, i32
                      }
                      %368 = arith.muli %arg19, %c4_i64_15 : i64
                      %369 = arith.index_cast %368 : i64 to index
                      ascsimt.store %arg17[%369], %367#0 : memref<8192xi32, 1>, i32
                      %370 = arith.addi %368, %c1_i64_14 : i64
                      %371 = arith.index_cast %370 : i64 to index
                      ascsimt.store %arg17[%371], %367#1 : memref<8192xi32, 1>, i32
                      %372 = arith.addi %368, %c2_i64_13 : i64
                      %373 = arith.index_cast %372 : i64 to index
                      ascsimt.store %arg17[%373], %367#2 : memref<8192xi32, 1>, i32
                      %374 = arith.addi %368, %c3_i64_12 : i64
                      %375 = arith.index_cast %374 : i64 to index
                      ascsimt.store %arg17[%375], %367#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                } else {
                  ascsimt.vf(%245, %255, %270, %271, %233, %243, %alloc_9, %272 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                  ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
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
                    %c16384_i32 = arith.constant 16384 : i32
                    %283 = ascsimt.thread_idx {ascsimt.scope = 52 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                    %284 = arith.extsi %283 {cannir.vf_threads = 256 : i64} : i32 to i64
                    scf.for %arg17 = %284 to %arg16 step %c256_i64_11  : i64 {
                      %285 = arith.trunci %arg17 : i64 to i32
                      %286 = arith.addi %arg9, %285 : i32
                      %287 = arith.cmpi ult, %286, %arg9 : i32
                      %288 = arith.extui %287 : i1 to i32
                      %289 = arith.addi %arg10, %288 : i32
                      %290 = arith.divui %286, %c16384_i32 : i32
                      %291 = arith.remui %289, %c16384_i32 : i32
                      %292 = arith.muli %291, %c262144_i32 : i32
                      %293 = arith.addi %290, %292 : i32
                      %294 = arith.divui %289, %c16384_i32 : i32
                      %295 = arith.addi %arg11, %293 : i32
                      %296 = arith.addi %arg12, %294 : i32
                      %297 = arith.cmpi ult, %295, %arg11 : i32
                      %298 = arith.extui %297 : i1 to i32
                      %299 = arith.addi %296, %298 : i32
                      %300 = arith.remui %286, %c16384_i32 : i32
                      %301:4 = scf.for %arg18 = %c0_i32_17 to %c10_i32 step %c1_i32_16 iter_args(%arg19 = %c0_i32_17, %arg20 = %300, %arg21 = %295, %arg22 = %299) -> (i32, i32, i32, i32)  : i32 {
                        %310 = arith.muli %arg19, %c-766435501_i32 : i32
                        %311 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                        %312 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %313 = ascsimt.math_binary %311, %312, "mulhi" : ui32, ui32 -> ui32
                        %314 = "ascsimt.bitcast"(%313) : (ui32) -> i32
                        %315 = arith.muli %arg21, %c-845247145_i32 : i32
                        %316 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                        %317 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                        %318 = ascsimt.math_binary %316, %317, "mulhi" : ui32, ui32 -> ui32
                        %319 = "ascsimt.bitcast"(%318) : (ui32) -> i32
                        %320 = arith.muli %arg18, %c-1640531527_i32_19 : i32
                        %321 = arith.addi %arg13, %320 : i32
                        %322 = arith.muli %arg18, %c-1150833019_i32_18 : i32
                        %323 = arith.addi %arg14, %322 : i32
                        %324 = arith.xori %319, %arg20 : i32
                        %325 = arith.xori %324, %321 : i32
                        %326 = arith.xori %314, %arg22 : i32
                        %327 = arith.xori %326, %323 : i32
                        scf.yield %325, %315, %327, %310 : i32, i32, i32, i32
                      }
                      %302 = arith.muli %arg17, %c4_i64_15 : i64
                      %303 = arith.index_cast %302 : i64 to index
                      ascsimt.store %arg15[%303], %301#0 : memref<8192xi32, 1>, i32
                      %304 = arith.addi %302, %c1_i64_14 : i64
                      %305 = arith.index_cast %304 : i64 to index
                      ascsimt.store %arg15[%305], %301#1 : memref<8192xi32, 1>, i32
                      %306 = arith.addi %302, %c2_i64_13 : i64
                      %307 = arith.index_cast %306 : i64 to index
                      ascsimt.store %arg15[%307], %301#2 : memref<8192xi32, 1>, i32
                      %308 = arith.addi %302, %c3_i64_12 : i64
                      %309 = arith.index_cast %308 : i64 to index
                      ascsimt.store %arg15[%309], %301#3 : memref<8192xi32, 1>, i32
                    } {cannir.vf_threads = 256 : i64}
                  } {threads = 256 : i64}
                }
                %278 = arith.trunci %32 : i64 to i32
                %279 = arith.divsi %278, %c2_i32 : i32
                %280 = arith.trunci %279 : i32 to i8
                %281 = arith.trunci %224 : i64 to i32
                ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
                ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                %282 = ascscalar.bitcast_memref %134 byte_offset = %c0_i64 : memref<32768xbf16, 1> -> memref<?xui8, 1>
                ascvec.vf(%280, %alloc_7, %134, %282, %alloc_9, %alloc_6, %281 : i8, memref<2048xui8, 1>, memref<32768xbf16, 1>, memref<?xui8, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, i32) {
                ^bb0(%arg9: i8, %arg10: memref<2048xui8, 1>, %arg11: memref<32768xbf16, 1>, %arg12: memref<?xui8, 1>, %arg13: memref<8192xi32, 1>, %arg14: memref<32768xui8, 1>, %arg15: i32):
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
                  %c32_i8 = arith.constant 32 : i8
                  %c65535_i32 = arith.constant 65535 : i32
                  %c3_i8 = arith.constant 3 : i8
                  %c0_i8 = arith.constant 0 : i8
                  %c-128_i8 = arith.constant -128 : i8
                  %283 = ascvec.create_mask : !ascvec.mask
                  %284 = ascvec.create_mask elem_bits 16 : !ascvec.mask
                  %285 = ascvec.create_mask elem_bits 8 : !ascvec.mask
                  %286 = ascvec.duplicate_scalar %c-128_i8, %285 : i8 -> <ui8>
                  %287 = ascvec.arange %c0_i8 : i8 -> <ui8>
                  %288 = ascvec.duplicate_scalar %c3_i8, %285 : i8 -> <ui8>
                  %289 = ascvec.and %287, %288, %285 : <ui8>, <ui8> -> <ui8>
                  %290 = ascvec.shl %289, 6, %285 : <ui8> -> <ui8>
                  %291 = ascvec.shr %287, 2, %285 : <ui8> -> <ui8>
                  %292 = ascvec.or %290, %291, %285 : <ui8>, <ui8> -> <ui8>
                  %293 = ascvec.duplicate_scalar %c65535_i32, %283 : i32 -> <ui32>
                  %294 = ascvec.cmp_lt_scalar %287, %c32_i8, %285 : <ui8>, i8 -> !ascvec.mask
                  %295 = ascvec.cmp_lt_scalar %289, %arg9, %285 : <ui8>, i8 -> !ascvec.mask
                  %296 = ascvec.mask_and %294, %295, %285
                  ascvec.store_unalign_begin %arg10 : memref<2048xui8, 1>
                  scf.for %arg16 = %c0_i32_12 to %arg15 step %c1_i32_11  : i32 {
                    %297 = arith.muli %arg16, %c256_i32 : i32
                    %298 = arith.index_cast %297 : i32 to index
                    %result0, %result1 = ascvec.loadalign_deintlv %arg11[%298] {dintlv_width = "b16"} : memref<32768xbf16, 1>, index -> <bf16>, <bf16>
                    %299 = ascvec.duplicate_scalar %c32767_i16, %284 : i16 -> <ui16>
                    %300 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
                    %301 = ascvec.and %300, %299, %284 : <ui16>, <ui16> -> <ui16>
                    %302 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
                    %303 = ascvec.and %302, %299, %284 : <ui16>, <ui16> -> <ui16>
                    %304 = ascvec.max %301, %303, %284 : <ui16>
                    %305 = ascvec.reduce_max_datablock %304, %284 : <ui16>
                    %306 = ascvec.unpack %305 {part = "lower"} : <ui16> -> <ui32>
                    %307 = ascvec.shl %306, 16, %283 : <ui32> -> <ui32>
                    %308 = arith.muli %arg16, %c512_i32 : i32
                    %309 = arith.index_cast %308 : i32 to index
                    %result0_15, %result1_16 = ascvec.loadalign_deintlv %arg12[%309] {dintlv_width = "b8"} : memref<?xui8, 1>, index -> <ui8>, <ui8>
                    %310 = ascvec.and %result1_16, %286, %285 : <ui8>, <ui8> -> <ui8>
                    %311 = ascvec.reinterpret %310 : <ui8> -> <ui32>
                    %312 = ascvec.shr %307, 23, %283 : <ui32> -> <ui32>
                    %313 = ascvec.cmp_ge_scalar %312, %c255_i32, %283 : <ui32>, i32 -> !ascvec.mask
                    %314 = ascvec.max_scalar %312, %c15_i32, %283 : <ui32>, i32
                    %315 = ascvec.add_scalar %314, %c-15_i32, %283 : <ui32>, i32
                    %316 = ascvec.duplicate_scalar %c255_i32, %313 merge %315 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                    %317 = ascvec.duplicate_scalar %c254_i32, %283 : i32 -> <ui32>
                    %318 = ascvec.sub %317, %316, %283 : <ui32>
                    %319 = ascvec.cmp_ne_scalar %316, %c0_i32_12, %283 : <ui32>, i32 -> !ascvec.mask
                    %320 = ascvec.shl %318, 23, %319 : <ui32> -> <ui32>
                    %321 = ascvec.reinterpret %320 : <ui32> -> <f32>
                    %322 = ascvec.reinterpret %321 : <f32> -> <ui32>
                    %323 = ascvec.shr %322, 16, %283 : <ui32> -> <ui32>
                    %324 = ascvec.pack %323 src_elem_bits 32 dst_elem_bits 16 {part = "low"} : <ui32> -> <ui16>
                    %325 = ascvec.arange %c0_i16 : i16 -> <ui16>
                    %326 = ascvec.shr %325, 4, %284 : <ui16> -> <ui16>
                    %327 = ascvec.gather_reg %324, %326 : <ui16>, <ui16> -> <ui16>
                    %328 = ascvec.reinterpret %327 : <ui16> -> <bf16>
                    %329 = ascvec.reinterpret %301 : <ui16> -> <bf16>
                    %330 = ascvec.mul %329, %328, %284 : <bf16>
                    %331 = ascvec.reinterpret %303 : <ui16> -> <bf16>
                    %332 = ascvec.mul %331, %328, %284 : <bf16>
                    %333 = ascvec.cast %330, %284 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %334 = ascvec.cast %330, %284 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %335 = ascvec.cast %332, %284 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %336 = ascvec.cast %332, %284 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
                    %337 = ascvec.reinterpret %328 : <bf16> -> <ui32>
                    %338 = arith.muli %arg16, %c64_i32 : i32
                    %339 = arith.index_cast %338 : i32 to index
                    %340 = ascvec.loadalign %arg13[%339] : memref<8192xi32, 1>, index -> <i32>
                    %341 = ascvec.reinterpret %340 : <i32> -> <ui32>
                    %342 = ascvec.shl %341, 8, %283 : <ui32> -> <ui32>
                    %343 = ascvec.shr %341, 24, %283 : <ui32> -> <ui32>
                    %344 = ascvec.or %342, %343, %283 : <ui32>, <ui32> -> <ui32>
                    %345 = ascvec.shr %341, 16, %283 : <ui32> -> <ui32>
                    %346 = ascvec.and %344, %293, %283 : <ui32>, <ui32> -> <ui32>
                    %347 = ascvec.and %341, %293, %283 : <ui32>, <ui32> -> <ui32>
                    %348 = ascvec.shr %344, 16, %283 : <ui32> -> <ui32>
                    %349 = ascvec.reinterpret %333 : <f32> -> <ui32>
                    %350 = ascvec.reinterpret %349 : <ui32> -> <f32>
                    %351 = ascvec.max_scalar %350, %cst, %283 : <f32>, f32
                    %352 = ascvec.add %350, %351, %283 : <f32>
                    %353 = ascvec.reinterpret %352 : <f32> -> <ui32>
                    %354 = ascvec.shr %353, 5, %283 : <ui32> -> <ui32>
                    %355 = ascvec.add %354, %345, %283 : <ui32>
                    %356 = ascvec.reinterpret %335 : <f32> -> <ui32>
                    %357 = ascvec.reinterpret %356 : <ui32> -> <f32>
                    %358 = ascvec.max_scalar %357, %cst, %283 : <f32>, f32
                    %359 = ascvec.add %357, %358, %283 : <f32>
                    %360 = ascvec.reinterpret %359 : <f32> -> <ui32>
                    %361 = ascvec.shr %360, 5, %283 : <ui32> -> <ui32>
                    %362 = ascvec.add %361, %346, %283 : <ui32>
                    %363 = ascvec.reinterpret %334 : <f32> -> <ui32>
                    %364 = ascvec.reinterpret %363 : <ui32> -> <f32>
                    %365 = ascvec.max_scalar %364, %cst, %283 : <f32>, f32
                    %366 = ascvec.add %364, %365, %283 : <f32>
                    %367 = ascvec.reinterpret %366 : <f32> -> <ui32>
                    %368 = ascvec.shr %367, 5, %283 : <ui32> -> <ui32>
                    %369 = ascvec.add %368, %347, %283 : <ui32>
                    %370 = ascvec.reinterpret %336 : <f32> -> <ui32>
                    %371 = ascvec.reinterpret %370 : <ui32> -> <f32>
                    %372 = ascvec.max_scalar %371, %cst, %283 : <f32>, f32
                    %373 = ascvec.add %371, %372, %283 : <f32>
                    %374 = ascvec.reinterpret %373 : <f32> -> <ui32>
                    %375 = ascvec.shr %374, 5, %283 : <ui32> -> <ui32>
                    %376 = ascvec.add %375, %348, %283 : <ui32>
                    %377 = ascvec.reinterpret %355 : <ui32> -> <ui16>
                    %378 = ascvec.reinterpret %362 : <ui32> -> <ui16>
                    %result0_17, %result1_18 = ascvec.deintlv %377, %378 : <ui16>, <ui16> -> <ui16>, <ui16>
                    %379 = ascvec.reinterpret %369 : <ui32> -> <ui16>
                    %380 = ascvec.reinterpret %376 : <ui32> -> <ui16>
                    %result0_19, %result1_20 = ascvec.deintlv %379, %380 : <ui16>, <ui16> -> <ui16>, <ui16>
                    %381 = ascvec.reinterpret %result1_18 : <ui16> -> <ui8>
                    %382 = ascvec.reinterpret %result1_20 : <ui16> -> <ui8>
                    %result0_21, %result1_22 = ascvec.deintlv %381, %382 : <ui8>, <ui8> -> <ui8>, <ui8>
                    %383 = ascvec.gather_reg %result0_21, %292 : <ui8>, <ui8> -> <ui8>
                    %384 = ascvec.reinterpret %383 : <ui8> -> <ui32>
                    %385 = ascvec.reinterpret %384 : <ui32> -> <ui8>
                    %386 = ascvec.add_scalar %385, %c60_i8, %285 : <ui8>, i8
                    %387 = ascvec.cmp_ne_scalar %386, %c-1_i8, %285 : <ui8>, i8 -> !ascvec.mask
                    %388 = ascvec.min_scalar %386, %c123_i8, %387 : <ui8>, i8
                    %389 = ascvec.reinterpret %388 : <ui8> -> <ui32>
                    %390 = ascvec.or %389, %311, %283 : <ui32>, <ui32> -> <ui32>
                    %391 = ascvec.reinterpret %337 : <ui32> -> <ui32>
                    %392 = ascvec.cmp_eq_scalar %391, %c-8323200_i32, %283 : <ui32>, i32 -> !ascvec.mask
                    %393 = ascvec.duplicate_scalar %c2139062143_i32, %392 merge %390 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                    %394 = ascvec.reinterpret %393 : <ui32> -> <ui8>
                    ascvec.storealign %arg14[%298], %394, %285 : memref<32768xui8, 1>, index, <ui8>
                    %395 = ascvec.reinterpret %316 : <ui32> -> <ui8>
                    %396 = ascvec.squeeze_store_reg %395, %296 : <ui8>, !ascvec.mask -> <ui8>
                    ascvec.store_unalign %arg10[%c0_13], %396 : memref<2048xui8, 1>, <ui8>
                  }
                  %c0_14 = arith.constant 0 : index
                  %c1 = arith.constant 1 : index
                  scf.for %arg16 = %c0_14 to %c1 step %c1 {
                    ascvec.store_unalign_post %arg10[%c0_13] : memref<2048xui8, 1>
                  }
                  ascvec.mem_bar vst_vld
                } {cannir.vf_mode = "simd"}
                ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
                ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              }
            }
          } else {
            %211 = arith.muli %93, %12 : i64
            %212 = arith.addi %211, %96 : i64
            %213 = arith.remsi %arg3, %c256_i64 : i64
            %214 = arith.cmpi ne, %213, %c0_i64 : i64
            %215 = arith.cmpi slt, %213, %c0_i64 : i64
            %216 = arith.andi %214, %215 : i1
            %217 = arith.select %216, %c256_i64, %c0_i64 : i64
            %218 = arith.addi %213, %217 overflow<nsw> : i64
            %219 = arith.cmpi eq, %218, %c0_i64 : i64
            scf.if %219 {
              %220 = arith.addi %99, %c63_i64 : i64
              %221 = arith.remsi %220, %c64_i64 : i64
              %222 = arith.cmpi ne, %221, %c0_i64 : i64
              %223 = arith.cmpi slt, %221, %c0_i64 : i64
              %224 = arith.andi %222, %223 : i1
              %225 = arith.divsi %220, %c64_i64 : i64
              %226 = arith.extui %224 : i1 to i64
              %227 = arith.subi %225, %226 : i64
              %228 = arith.addi %99, %c1023_i64 : i64
              %229 = arith.remsi %228, %c1024_i64 : i64
              %230 = arith.cmpi ne, %229, %c0_i64 : i64
              %231 = arith.cmpi slt, %229, %c0_i64 : i64
              %232 = arith.andi %230, %231 : i1
              %233 = arith.divsi %228, %c1024_i64 : i64
              %234 = arith.extui %232 : i1 to i64
              %235 = arith.subi %233, %234 : i64
              %236 = arith.trunci %1 : i64 to i32
              %237 = arith.extui %236 : i32 to i64
              %238 = arith.subi %1, %237 : i64
              %239 = arith.remsi %238, %c4294967296_i64 : i64
              %240 = arith.cmpi ne, %239, %c0_i64 : i64
              %241 = arith.cmpi slt, %239, %c0_i64 : i64
              %242 = arith.andi %240, %241 : i1
              %243 = arith.divsi %238, %c4294967296_i64 : i64
              %244 = arith.extui %242 : i1 to i64
              %245 = arith.subi %243, %244 : i64
              %246 = arith.trunci %245 : i64 to i32
              %247 = arith.divui %212, %c32_i64 : i64
              %248 = arith.trunci %247 : i64 to i32
              %249 = arith.extui %248 : i32 to i64
              %250 = arith.subi %247, %249 : i64
              %251 = arith.remsi %250, %c4294967296_i64 : i64
              %252 = arith.cmpi ne, %251, %c0_i64 : i64
              %253 = arith.cmpi slt, %251, %c0_i64 : i64
              %254 = arith.andi %252, %253 : i1
              %255 = arith.divsi %250, %c4294967296_i64 : i64
              %256 = arith.extui %254 : i1 to i64
              %257 = arith.subi %255, %256 : i64
              %258 = arith.trunci %257 : i64 to i32
              %259 = arith.trunci %4 : i64 to i32
              %260 = arith.extui %259 : i32 to i64
              %261 = arith.subi %4, %260 : i64
              %262 = arith.remsi %261, %c4294967296_i64 : i64
              %263 = arith.cmpi ne, %262, %c0_i64 : i64
              %264 = arith.cmpi slt, %262, %c0_i64 : i64
              %265 = arith.andi %263, %264 : i1
              %266 = arith.divsi %261, %c4294967296_i64 : i64
              %267 = arith.extui %265 : i1 to i64
              %268 = arith.subi %266, %267 : i64
              %269 = arith.trunci %268 : i64 to i32
              %270 = arith.divui %259, %c4_i32 : i32
              %271 = arith.remui %269, %c4_i32 : i32
              %272 = arith.muli %271, %c1073741824_i32 : i32
              %273 = arith.addi %270, %272 : i32
              %274 = arith.divui %269, %c4_i32 : i32
              %275 = arith.muli %235, %c32_i64 : i64
              %276 = arith.divui %247, %c16384_i64 : i64
              %277 = arith.addi %247, %275 : i64
              %278 = arith.subi %277, %c1_i64 : i64
              %279 = arith.divui %278, %c16384_i64 : i64
              %280 = arith.cmpi eq, %276, %279 : i64
              scf.if %280 {
                %286 = arith.trunci %276 : i64 to i32
                %287 = arith.extui %286 : i32 to i64
                %288 = arith.subi %276, %287 : i64
                %289 = arith.remsi %288, %c4294967296_i64 : i64
                %290 = arith.cmpi ne, %289, %c0_i64 : i64
                %291 = arith.cmpi slt, %289, %c0_i64 : i64
                %292 = arith.andi %290, %291 : i1
                %293 = arith.divsi %288, %c4294967296_i64 : i64
                %294 = arith.extui %292 : i1 to i64
                %295 = arith.subi %293, %294 : i64
                %296 = arith.trunci %295 : i64 to i32
                %297 = arith.addi %273, %286 : i32
                %298 = arith.addi %274, %296 : i32
                %299 = arith.cmpi ult, %297, %273 : i32
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
                %315 = arith.xori %301, %246 : i32
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
                %329 = arith.addi %236, %c-1640531527_i32 : i32
                %330 = arith.addi %246, %c-1150833019_i32 : i32
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
                %346 = arith.xori %314, %236 : i32
                %347 = arith.addi %236, %c1013904242_i32 : i32
                %348 = arith.xori %318, %347 : i32
                %349 = arith.addi %246, %c1993301258_i32 : i32
                %350 = arith.xori %345, %349 : i32
                ascsimt.vf(%248, %346, %330, %348, %350, %236, %246, %335, %alloc_9, %275 : i32, i32, i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: memref<8192xi32, 1>, %arg18: i64):
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
                  %c16384_i32 = arith.constant 16384 : i32
                  %351 = ascsimt.thread_idx {ascsimt.scope = 54 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                  %352 = arith.extsi %351 {cannir.vf_threads = 256 : i64} : i32 to i64
                  scf.for %arg19 = %352 to %arg18 step %c256_i64_11  : i64 {
                    %353 = arith.trunci %arg19 : i64 to i32
                    %354 = arith.addi %arg9, %353 : i32
                    %355 = arith.remui %354, %c16384_i32 : i32
                    %356 = arith.xori %355, %arg10 : i32
                    %357 = arith.muli %356, %c-766435501_i32 : i32
                    %358 = "ascsimt.bitcast"(%356) : (i32) -> ui32
                    %359 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                    %360 = ascsimt.math_binary %358, %359, "mulhi" : ui32, ui32 -> ui32
                    %361 = "ascsimt.bitcast"(%360) : (ui32) -> i32
                    %362 = arith.xori %361, %arg11 : i32
                    %363 = arith.muli %362, %c-845247145_i32 : i32
                    %364 = "ascsimt.bitcast"(%362) : (i32) -> ui32
                    %365 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                    %366 = ascsimt.math_binary %364, %365, "mulhi" : ui32, ui32 -> ui32
                    %367 = "ascsimt.bitcast"(%366) : (ui32) -> i32
                    %368 = arith.xori %367, %arg12 : i32
                    %369 = arith.xori %357, %arg13 : i32
                    %370:4 = scf.for %arg20 = %c3_i32 to %c10_i32 step %c1_i32_16 iter_args(%arg21 = %368, %arg22 = %363, %arg23 = %369, %arg24 = %arg16) -> (i32, i32, i32, i32)  : i32 {
                      %379 = arith.muli %arg21, %c-766435501_i32 : i32
                      %380 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                      %381 = ascsimt.math_binary %380, %359, "mulhi" : ui32, ui32 -> ui32
                      %382 = "ascsimt.bitcast"(%381) : (ui32) -> i32
                      %383 = arith.muli %arg23, %c-845247145_i32 : i32
                      %384 = "ascsimt.bitcast"(%arg23) : (i32) -> ui32
                      %385 = ascsimt.math_binary %384, %365, "mulhi" : ui32, ui32 -> ui32
                      %386 = "ascsimt.bitcast"(%385) : (ui32) -> i32
                      %387 = arith.muli %arg20, %c-1640531527_i32_18 : i32
                      %388 = arith.addi %arg14, %387 : i32
                      %389 = arith.muli %arg20, %c-1150833019_i32_17 : i32
                      %390 = arith.addi %arg15, %389 : i32
                      %391 = arith.xori %386, %arg22 : i32
                      %392 = arith.xori %391, %388 : i32
                      %393 = arith.xori %382, %arg24 : i32
                      %394 = arith.xori %393, %390 : i32
                      scf.yield %392, %383, %394, %379 : i32, i32, i32, i32
                    }
                    %371 = arith.muli %arg19, %c4_i64_15 : i64
                    %372 = arith.index_cast %371 : i64 to index
                    ascsimt.store %arg17[%372], %370#0 : memref<8192xi32, 1>, i32
                    %373 = arith.addi %371, %c1_i64_14 : i64
                    %374 = arith.index_cast %373 : i64 to index
                    ascsimt.store %arg17[%374], %370#1 : memref<8192xi32, 1>, i32
                    %375 = arith.addi %371, %c2_i64_13 : i64
                    %376 = arith.index_cast %375 : i64 to index
                    ascsimt.store %arg17[%376], %370#2 : memref<8192xi32, 1>, i32
                    %377 = arith.addi %371, %c3_i64_12 : i64
                    %378 = arith.index_cast %377 : i64 to index
                    ascsimt.store %arg17[%378], %370#3 : memref<8192xi32, 1>, i32
                  } {cannir.vf_threads = 256 : i64}
                } {threads = 256 : i64}
              } else {
                ascsimt.vf(%248, %258, %273, %274, %236, %246, %alloc_9, %275 : i32, i32, i32, i32, i32, i32, memref<8192xi32, 1>, i64) {
                ^bb0(%arg9: i32, %arg10: i32, %arg11: i32, %arg12: i32, %arg13: i32, %arg14: i32, %arg15: memref<8192xi32, 1>, %arg16: i64):
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
                  %c16384_i32 = arith.constant 16384 : i32
                  %286 = ascsimt.thread_idx {ascsimt.scope = 55 : i64, axis = "x", cannir.vf_threads = 256 : i64} : i32
                  %287 = arith.extsi %286 {cannir.vf_threads = 256 : i64} : i32 to i64
                  scf.for %arg17 = %287 to %arg16 step %c256_i64_11  : i64 {
                    %288 = arith.trunci %arg17 : i64 to i32
                    %289 = arith.addi %arg9, %288 : i32
                    %290 = arith.cmpi ult, %289, %arg9 : i32
                    %291 = arith.extui %290 : i1 to i32
                    %292 = arith.addi %arg10, %291 : i32
                    %293 = arith.divui %289, %c16384_i32 : i32
                    %294 = arith.remui %292, %c16384_i32 : i32
                    %295 = arith.muli %294, %c262144_i32 : i32
                    %296 = arith.addi %293, %295 : i32
                    %297 = arith.divui %292, %c16384_i32 : i32
                    %298 = arith.addi %arg11, %296 : i32
                    %299 = arith.addi %arg12, %297 : i32
                    %300 = arith.cmpi ult, %298, %arg11 : i32
                    %301 = arith.extui %300 : i1 to i32
                    %302 = arith.addi %299, %301 : i32
                    %303 = arith.remui %289, %c16384_i32 : i32
                    %304:4 = scf.for %arg18 = %c0_i32_17 to %c10_i32 step %c1_i32_16 iter_args(%arg19 = %c0_i32_17, %arg20 = %303, %arg21 = %298, %arg22 = %302) -> (i32, i32, i32, i32)  : i32 {
                      %313 = arith.muli %arg19, %c-766435501_i32 : i32
                      %314 = "ascsimt.bitcast"(%arg19) : (i32) -> ui32
                      %315 = "ascsimt.cast"(%c-766435501_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %316 = ascsimt.math_binary %314, %315, "mulhi" : ui32, ui32 -> ui32
                      %317 = "ascsimt.bitcast"(%316) : (ui32) -> i32
                      %318 = arith.muli %arg21, %c-845247145_i32 : i32
                      %319 = "ascsimt.bitcast"(%arg21) : (i32) -> ui32
                      %320 = "ascsimt.cast"(%c-845247145_i32) <{rounding = 4 : i32, saturate = false}> : (i32) -> ui32
                      %321 = ascsimt.math_binary %319, %320, "mulhi" : ui32, ui32 -> ui32
                      %322 = "ascsimt.bitcast"(%321) : (ui32) -> i32
                      %323 = arith.muli %arg18, %c-1640531527_i32_19 : i32
                      %324 = arith.addi %arg13, %323 : i32
                      %325 = arith.muli %arg18, %c-1150833019_i32_18 : i32
                      %326 = arith.addi %arg14, %325 : i32
                      %327 = arith.xori %322, %arg20 : i32
                      %328 = arith.xori %327, %324 : i32
                      %329 = arith.xori %317, %arg22 : i32
                      %330 = arith.xori %329, %326 : i32
                      scf.yield %328, %318, %330, %313 : i32, i32, i32, i32
                    }
                    %305 = arith.muli %arg17, %c4_i64_15 : i64
                    %306 = arith.index_cast %305 : i64 to index
                    ascsimt.store %arg15[%306], %304#0 : memref<8192xi32, 1>, i32
                    %307 = arith.addi %305, %c1_i64_14 : i64
                    %308 = arith.index_cast %307 : i64 to index
                    ascsimt.store %arg15[%308], %304#1 : memref<8192xi32, 1>, i32
                    %309 = arith.addi %305, %c2_i64_13 : i64
                    %310 = arith.index_cast %309 : i64 to index
                    ascsimt.store %arg15[%310], %304#2 : memref<8192xi32, 1>, i32
                    %311 = arith.addi %305, %c3_i64_12 : i64
                    %312 = arith.index_cast %311 : i64 to index
                    ascsimt.store %arg15[%312], %304#3 : memref<8192xi32, 1>, i32
                  } {cannir.vf_threads = 256 : i64}
                } {threads = 256 : i64}
              }
              %281 = arith.trunci %227 : i64 to i32
              %282 = arith.trunci %99 : i64 to i32
              %283 = arith.trunci %32 : i64 to i32
              %284 = arith.muli %281, %c2_i32 : i32
              %285 = arith.cmpi eq, %283, %c4_i32 : i32
              ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%99, %134, %alloc_9, %alloc_6, %alloc_8, %281, %282, %284, %285, %alloc_7 : i64, memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, memref<1024xui32, 1>, i32, i32, i32, i1, memref<2048xui8, 1>) {
              ^bb0(%arg9: i64, %arg10: memref<32768xbf16, 1>, %arg11: memref<8192xi32, 1>, %arg12: memref<32768xui8, 1>, %arg13: memref<1024xui32, 1>, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i1, %arg18: memref<2048xui8, 1>):
                %c2139095040_i32 = arith.constant 2139095040 : i32
                %c127_i32 = arith.constant 127 : i32
                %c123_i32 = arith.constant 123 : i32
                %c947912704_i32 = arith.constant 947912704 : i32
                %cst = arith.constant 4.2949673E+9 : f32
                %c29360128_i32 = arith.constant 29360128 : i32
                %c128_i32 = arith.constant 128 : i32
                %c65535_i32 = arith.constant 65535 : i32
                %c2_i32_11 = arith.constant 2 : i32
                %c7_i32 = arith.constant 7 : i32
                %c1_i32_12 = arith.constant 1 : i32
                %c4_i32_13 = arith.constant 4 : i32
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
                %286 = ascvec.create_mask : !ascvec.mask
                %287 = ascvec.arange %c0_i32_16 : i32 -> <ui32>
                %288 = ascvec.cmp_lt_scalar %287, %c32_i32, %286 : <ui32>, i32 -> !ascvec.mask
                %289 = ascvec.cmp_ge_scalar %287, %c32_i32, %286 : <ui32>, i32 -> !ascvec.mask
                scf.for %arg19 = %c0_i32_16 to %arg14 step %c1_i32_12  : i32 {
                  %291 = arith.muli %arg19, %c64_i32 : i32
                  %292 = arith.trunci %arg9 : i64 to i32
                  %293 = arith.subi %292, %291 : i32
                  %mask, %remaining_out = ascvec.update_mask %293 : !ascvec.mask
                  %294 = arith.index_cast %291 : i32 to index
                  %295 = ascvec.loadalign_unpack %arg10[%294] {mode = "b16_to_b32"} : memref<32768xbf16, 1>, index -> <bf16>
                  %296 = ascvec.cast %295, %286 {rounding = 6 : i32} : <bf16> -> <f32>
                  %297 = ascvec.duplicate_scalar %cst_14, %286 : f32 -> <f32>
                  %298 = ascvec.select %296, %297, %mask : <f32>
                  %299 = ascvec.reinterpret %298 : <f32> -> <ui32>
                  %300 = ascvec.duplicate_scalar %c2147483647_i32, %286 : i32 -> <ui32>
                  %301 = ascvec.and %299, %300, %286 : <ui32>, <ui32> -> <ui32>
                  %302 = ascvec.reduce_max %301, %288 : <ui32>
                  %303 = ascvec.duplicate %302, %286 : <ui32> -> <ui32>
                  %304 = ascvec.reduce_max %301, %289 : <ui32>
                  %305 = ascvec.duplicate %304, %286 : <ui32> -> <ui32>
                  %306 = ascvec.select %303, %305, %288 : <ui32>
                  %307 = ascvec.shr %306, 23, %286 : <ui32> -> <ui32>
                  %308 = ascvec.cmp_ge_scalar %307, %c255_i32, %286 : <ui32>, i32 -> !ascvec.mask
                  %309 = ascvec.max_scalar %307, %c15_i32, %286 : <ui32>, i32
                  %310 = ascvec.add_scalar %309, %c-15_i32, %286 : <ui32>, i32
                  %311 = ascvec.duplicate_scalar %c255_i32, %308 merge %310 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %312 = ascvec.duplicate_scalar %c254_i32, %286 : i32 -> <ui32>
                  %313 = ascvec.sub %312, %311, %286 : <ui32>
                  %314 = ascvec.cmp_ne_scalar %311, %c0_i32_16, %286 : <ui32>, i32 -> !ascvec.mask
                  %315 = ascvec.shl %313, 23, %314 : <ui32> -> <ui32>
                  %316 = ascvec.duplicate_scalar %c2139160576_i32, %308 merge %315 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %317 = ascvec.reinterpret %316 : <ui32> -> <f32>
                  %318 = ascvec.mul %298, %317, %286 : <f32>
                  %319 = ascvec.cast %318, %286 : <f32> -> <bf16>
                  %320 = ascvec.cast %319, %286 {rounding = 6 : i32} : <bf16> -> <f32>
                  %321 = arith.remsi %arg19, %c4_i32_13 : i32
                  %322 = arith.cmpi ne, %321, %c0_i32_16 : i32
                  %323 = arith.cmpi slt, %321, %c0_i32_16 : i32
                  %324 = arith.andi %322, %323 : i1
                  %325 = arith.select %324, %c4_i32_13, %c0_i32_16 : i32
                  %326 = arith.addi %321, %325 overflow<nsw> : i32
                  %327 = arith.muli %326, %c64_i32 overflow<nsw> : i32
                  %328 = ascvec.add_scalar %287, %327, %286 : <ui32>, i32
                  %329 = arith.divsi %arg19, %c4_i32_13 : i32
                  %330 = arith.extui %324 : i1 to i32
                  %331 = arith.subi %329, %330 : i32
                  %332 = ascvec.shr %328, 3, %286 : <ui32> -> <ui32>
                  %333 = arith.muli %331, %c32_i32 : i32
                  %334 = ascvec.add_scalar %332, %333, %286 : <ui32>, i32
                  %335 = ascvec.gather %arg11[%c0_15], %334, %286 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                  %336 = ascvec.reinterpret %335 : <i32> -> <ui32>
                  %337 = ascvec.duplicate_scalar %c7_i32, %286 : i32 -> <ui32>
                  %338 = ascvec.and %328, %337, %286 : <ui32>, <ui32> -> <ui32>
                  %339 = ascvec.shl %336, 11, %286 : <ui32> -> <ui32>
                  %340 = ascvec.shr %336, 21, %286 : <ui32> -> <ui32>
                  %341 = ascvec.or %339, %340, %286 : <ui32>, <ui32> -> <ui32>
                  %342 = ascvec.shr %341, 3, %286 : <ui32> -> <ui32>
                  %343 = ascvec.shl %341, 29, %286 : <ui32> -> <ui32>
                  %344 = ascvec.or %342, %343, %286 : <ui32>, <ui32> -> <ui32>
                  %345 = ascvec.duplicate_scalar %c1_i32_12, %286 : i32 -> <ui32>
                  %346 = ascvec.and %338, %345, %286 : <ui32>, <ui32> -> <ui32>
                  %347 = ascvec.cmp_ne_scalar %346, %c0_i32_16, %286 : <ui32>, i32 -> !ascvec.mask
                  %348 = ascvec.select %344, %341, %347 : <ui32>
                  %349 = ascvec.shr %348, 8, %286 : <ui32> -> <ui32>
                  %350 = ascvec.shl %348, 24, %286 : <ui32> -> <ui32>
                  %351 = ascvec.or %349, %350, %286 : <ui32>, <ui32> -> <ui32>
                  %352 = ascvec.duplicate_scalar %c2_i32_11, %286 : i32 -> <ui32>
                  %353 = ascvec.and %338, %352, %286 : <ui32>, <ui32> -> <ui32>
                  %354 = ascvec.cmp_ne_scalar %353, %c0_i32_16, %286 : <ui32>, i32 -> !ascvec.mask
                  %355 = ascvec.select %351, %348, %354 : <ui32>
                  %356 = ascvec.shr %355, 16, %286 : <ui32> -> <ui32>
                  %357 = ascvec.shl %355, 16, %286 : <ui32> -> <ui32>
                  %358 = ascvec.or %356, %357, %286 : <ui32>, <ui32> -> <ui32>
                  %359 = ascvec.duplicate_scalar %c4_i32_13, %286 : i32 -> <ui32>
                  %360 = ascvec.and %338, %359, %286 : <ui32>, <ui32> -> <ui32>
                  %361 = ascvec.cmp_ne_scalar %360, %c0_i32_16, %286 : <ui32>, i32 -> !ascvec.mask
                  %362 = ascvec.select %358, %355, %361 : <ui32>
                  %363 = ascvec.duplicate_scalar %c65535_i32, %286 : i32 -> <ui32>
                  %364 = ascvec.and %362, %363, %286 : <ui32>, <ui32> -> <ui32>
                  %365 = ascvec.reinterpret %320 : <f32> -> <ui32>
                  %366 = ascvec.shr %365, 24, %286 : <ui32> -> <ui32>
                  %367 = ascvec.duplicate_scalar %c128_i32, %286 : i32 -> <ui32>
                  %368 = ascvec.and %366, %367, %286 : <ui32>, <ui32> -> <ui32>
                  %369 = ascvec.and %365, %300, %286 : <ui32>, <ui32> -> <ui32>
                  %370 = ascvec.shr %369, 5, %286 : <ui32> -> <ui32>
                  %371 = ascvec.duplicate_scalar %c29360128_i32, %286 : i32 -> <ui32>
                  %372 = ascvec.sub %370, %371, %286 : <ui32>
                  %373 = ascvec.reinterpret %369 : <ui32> -> <f32>
                  %374 = ascvec.mul_scalar %373, %cst, %286 : <f32>, f32
                  %375 = ascvec.cast %374, %286 {rounding = 4 : i32} : <f32> -> <i32>
                  %376 = ascvec.reinterpret %375 : <i32> -> <ui32>
                  %377 = ascvec.cmp_ge_scalar %369, %c947912704_i32, %286 : <ui32>, i32 -> !ascvec.mask
                  %378 = ascvec.select %372, %376, %377 : <ui32>
                  %379 = ascvec.and %364, %363, %286 : <ui32>, <ui32> -> <ui32>
                  %380 = ascvec.add %378, %379, %286 : <ui32>
                  %381 = ascvec.shr %380, 16, %286 : <ui32> -> <ui32>
                  %382 = ascvec.min_scalar %381, %c123_i32, %286 : <ui32>, i32
                  %383 = ascvec.or %382, %368, %286 : <ui32>, <ui32> -> <ui32>
                  %384 = ascvec.duplicate_scalar %c127_i32, %286 : i32 -> <ui32>
                  %385 = ascvec.cmp_gt_scalar %369, %c2139095040_i32, %286 : <ui32>, i32 -> !ascvec.mask
                  %386 = ascvec.select %384, %383, %385 : <ui32>
                  ascvec.storealign_pack %arg12[%294], %386, %mask {mode = "b32_to_b8"} : memref<32768xui8, 1>, index, <ui32>
                  %387 = arith.muli %arg19, %c2_i32_11 : i32
                  %388 = arith.index_cast %387 : i32 to index
                  ascvec.storealign_1st %arg13[%388], %311 : memref<1024xui32, 1>, index, <ui32>
                  %389 = ascvec.duplicate_scalar %c32_i32, %286 : i32 -> <ui32>
                  %390 = ascvec.gather_reg %311, %389 : <ui32>, <ui32> -> <ui32>
                  %391 = arith.addi %387, %c1_i32_12 : i32
                  %392 = arith.index_cast %391 : i32 to index
                  ascvec.storealign_1st %arg13[%392], %390 : memref<1024xui32, 1>, index, <ui32>
                }
                ascvec.mem_bar vst_vld
                %290 = scf.if %arg17 -> (i32) {
                  %291 = arith.remsi %arg15, %c32_i32 : i32
                  %292 = arith.cmpi ne, %291, %c0_i32_16 : i32
                  %293 = arith.cmpi slt, %291, %c0_i32_16 : i32
                  %294 = arith.andi %292, %293 : i1
                  %295 = arith.divsi %arg15, %c32_i32 : i32
                  %296 = arith.extui %294 : i1 to i32
                  %297 = arith.subi %295, %296 : i32
                  %298 = arith.muli %297, %c2_i32_11 : i32
                  scf.yield %298 : i32
                } else {
                  scf.yield %arg16 : i32
                }
                scf.for %arg19 = %c0_i32_16 to %290 step %c64_i32  : i32 {
                  %291 = arith.subi %290, %arg19 : i32
                  %mask, %remaining_out = ascvec.update_mask %291 : !ascvec.mask
                  %292 = ascvec.add_scalar %287, %arg19, %286 : <ui32>, i32
                  scf.if %arg17 {
                    %293 = ascvec.shr %292, 1, %286 : <ui32> -> <ui32>
                    %294 = ascvec.gather %arg13[%c0_15], %293, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %295 = ascvec.duplicate_scalar %c1_i32_12, %286 : i32 -> <ui32>
                    %296 = ascvec.and %287, %295, %286 : <ui32>, <ui32> -> <ui32>
                    %297 = ascvec.cmp_eq_scalar %296, %c0_i32_16, %286 : <ui32>, i32 -> !ascvec.mask
                    %298 = ascvec.duplicate_scalar %c0_i32_16, %286 : i32 -> <ui32>
                    %299 = ascvec.select %294, %298, %297 : <ui32>
                    %300 = arith.index_cast %arg19 : i32 to index
                    %c0_17 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg20 = %c0_17 to %c1 step %c1 {
                      ascvec.storealign_pack %arg18[%300], %299, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  } else {
                    %293 = ascvec.gather %arg13[%c0_15], %292, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %294 = arith.index_cast %arg19 : i32 to index
                    %c0_17 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg20 = %c0_17 to %c1 step %c1 {
                      ascvec.storealign_pack %arg18[%294], %293, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  }
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
            } else {
              %220 = arith.addi %99, %c63_i64 : i64
              %221 = arith.remsi %220, %c64_i64 : i64
              %222 = arith.cmpi ne, %221, %c0_i64 : i64
              %223 = arith.cmpi slt, %221, %c0_i64 : i64
              %224 = arith.andi %222, %223 : i1
              %225 = arith.divsi %220, %c64_i64 : i64
              %226 = arith.extui %224 : i1 to i64
              %227 = arith.subi %225, %226 : i64
              %228 = arith.addi %99, %c1023_i64 : i64
              %229 = arith.remsi %228, %c1024_i64 : i64
              %230 = arith.cmpi ne, %229, %c0_i64 : i64
              %231 = arith.cmpi slt, %229, %c0_i64 : i64
              %232 = arith.andi %230, %231 : i1
              %233 = arith.divsi %228, %c1024_i64 : i64
              %234 = arith.extui %232 : i1 to i64
              %235 = arith.subi %233, %234 : i64
              %236 = arith.trunci %1 : i64 to i32
              %237 = arith.extui %236 : i32 to i64
              %238 = arith.subi %1, %237 : i64
              %239 = arith.remsi %238, %c4294967296_i64 : i64
              %240 = arith.cmpi ne, %239, %c0_i64 : i64
              %241 = arith.cmpi slt, %239, %c0_i64 : i64
              %242 = arith.andi %240, %241 : i1
              %243 = arith.divsi %238, %c4294967296_i64 : i64
              %244 = arith.extui %242 : i1 to i64
              %245 = arith.subi %243, %244 : i64
              %246 = arith.trunci %245 : i64 to i32
              %247 = arith.remsi %212, %c16_i64 : i64
              %248 = arith.cmpi ne, %247, %c0_i64 : i64
              %249 = arith.cmpi slt, %247, %c0_i64 : i64
              %250 = arith.andi %248, %249 : i1
              %251 = arith.divsi %212, %c16_i64 : i64
              %252 = arith.extui %250 : i1 to i64
              %253 = arith.subi %251, %252 : i64
              %254 = arith.remsi %arg3, %c16_i64 : i64
              %255 = arith.cmpi ne, %254, %c0_i64 : i64
              %256 = arith.cmpi slt, %254, %c0_i64 : i64
              %257 = arith.andi %255, %256 : i1
              %258 = arith.divsi %arg3, %c16_i64 : i64
              %259 = arith.extui %257 : i1 to i64
              %260 = arith.subi %258, %259 : i64
              %261 = arith.remsi %253, %260 : i64
              %262 = arith.cmpi ne, %261, %c0_i64 : i64
              %263 = arith.cmpi slt, %261, %c0_i64 : i64
              %264 = arith.cmpi slt, %260, %c0_i64 : i64
              %265 = arith.xori %263, %264 : i1
              %266 = arith.andi %262, %265 : i1
              %267 = arith.divsi %253, %260 : i64
              %268 = arith.extui %266 : i1 to i64
              %269 = arith.subi %267, %268 : i64
              %270 = arith.select %266, %260, %c0_i64 : i64
              %271 = arith.addi %261, %270 : i64
              %272 = arith.trunci %269 : i64 to i32
              %273 = arith.extui %272 : i32 to i64
              %274 = arith.subi %269, %273 : i64
              %275 = arith.remsi %274, %c4294967296_i64 : i64
              %276 = arith.cmpi ne, %275, %c0_i64 : i64
              %277 = arith.cmpi slt, %275, %c0_i64 : i64
              %278 = arith.andi %276, %277 : i1
              %279 = arith.divsi %274, %c4294967296_i64 : i64
              %280 = arith.extui %278 : i1 to i64
              %281 = arith.subi %279, %280 : i64
              %282 = arith.trunci %281 : i64 to i32
              %283 = arith.trunci %271 : i64 to i32
              %284 = arith.extui %283 : i32 to i64
              %285 = arith.subi %271, %284 : i64
              %286 = arith.remsi %285, %c4294967296_i64 : i64
              %287 = arith.cmpi ne, %286, %c0_i64 : i64
              %288 = arith.cmpi slt, %286, %c0_i64 : i64
              %289 = arith.andi %287, %288 : i1
              %290 = arith.divsi %285, %c4294967296_i64 : i64
              %291 = arith.extui %289 : i1 to i64
              %292 = arith.subi %290, %291 : i64
              %293 = arith.trunci %292 : i64 to i32
              %294 = arith.remsi %c64_i64, %260 : i64
              %295 = arith.cmpi ne, %294, %c0_i64 : i64
              %296 = arith.cmpi slt, %294, %c0_i64 : i64
              %297 = arith.xori %296, %264 : i1
              %298 = arith.andi %295, %297 : i1
              %299 = arith.divsi %c64_i64, %260 : i64
              %300 = arith.extui %298 : i1 to i64
              %301 = arith.subi %299, %300 : i64
              %302 = arith.select %298, %260, %c0_i64 : i64
              %303 = arith.addi %294, %302 : i64
              %304 = arith.trunci %235 : i64 to i32
              ascvec.vf(%260, %4, %236, %246, %alloc_9, %303, %301, %304, %272, %282, %283, %293 : i64, i64, i32, i32, memref<8192xi32, 1>, i64, i64, i32, i32, i32, i32, i32) {
              ^bb0(%arg9: i64, %arg10: i64, %arg11: i32, %arg12: i32, %arg13: memref<8192xi32, 1>, %arg14: i64, %arg15: i64, %arg16: i32, %arg17: i32, %arg18: i32, %arg19: i32, %arg20: i32):
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
                %310:4 = scf.for %arg21 = %c0_i32_19 to %arg16 step %c1_i32_11 iter_args(%arg22 = %arg17, %arg23 = %arg18, %arg24 = %arg19, %arg25 = %arg20) -> (i32, i32, i32, i32)  : i32 {
                  %311 = arith.extui %arg22 : i32 to i64
                  %312 = arith.extui %arg23 : i32 to i64
                  %313 = arith.muli %312, %c4294967296_i64_20 : i64
                  %314 = arith.addi %311, %313 : i64
                  %315 = arith.extui %arg24 : i32 to i64
                  %316 = arith.extui %arg25 : i32 to i64
                  %317 = arith.muli %316, %c4294967296_i64_20 : i64
                  %318 = arith.addi %315, %317 : i64
                  %319 = arith.muli %arg21, %c256_i32 : i32
                  %320 = ascvec.create_mask : !ascvec.mask
                  %321 = ascvec.arange %c0_i32_19 : i32 -> <ui32>
                  %322 = arith.muli %314, %arg9 : i64
                  %323 = arith.addi %322, %318 : i64
                  %324 = arith.trunci %323 : i64 to i32
                  %325 = arith.extui %324 : i32 to i64
                  %326 = arith.subi %323, %325 : i64
                  %327 = arith.remsi %326, %c4294967296_i64_20 : i64
                  %328 = arith.cmpi ne, %327, %c0_i64_18 : i64
                  %329 = arith.cmpi slt, %327, %c0_i64_18 : i64
                  %330 = arith.andi %328, %329 : i1
                  %331 = arith.divsi %326, %c4294967296_i64_20 : i64
                  %332 = arith.extui %330 : i1 to i64
                  %333 = arith.subi %331, %332 : i64
                  %334 = arith.trunci %333 : i64 to i32
                  %335 = ascvec.duplicate_scalar %324, %320 : i32 -> <ui32>
                  %336 = ascvec.duplicate_scalar %334, %320 : i32 -> <ui32>
                  %337 = ascvec.duplicate_scalar %c0_i32_19, %320 : i32 -> <ui32>
                  %carry, %dst = ascvec.addc %335, %321, %320 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %carry_21, %dst_22 = ascvec.addc %336, %337, %carry, %320 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %338 = arith.trunci %arg10 : i64 to i32
                  %339 = arith.extui %338 : i32 to i64
                  %340 = arith.subi %arg10, %339 : i64
                  %341 = arith.remsi %340, %c4294967296_i64_20 : i64
                  %342 = arith.cmpi ne, %341, %c0_i64_18 : i64
                  %343 = arith.cmpi slt, %341, %c0_i64_18 : i64
                  %344 = arith.andi %342, %343 : i1
                  %345 = arith.divsi %340, %c4294967296_i64_20 : i64
                  %346 = arith.extui %344 : i1 to i64
                  %347 = arith.subi %345, %346 : i64
                  %348 = arith.trunci %347 : i64 to i32
                  %349 = arith.divui %338, %c4_i32_17 : i32
                  %350 = arith.remui %348, %c4_i32_17 : i32
                  %351 = arith.muli %350, %c1073741824_i32_16 : i32
                  %352 = arith.addi %349, %351 : i32
                  %353 = arith.divui %348, %c4_i32_17 : i32
                  %354 = ascvec.shr %dst, 14, %320 : <ui32> -> <ui32>
                  %355 = ascvec.shl %dst_22, 18, %320 : <ui32> -> <ui32>
                  %356 = ascvec.or %354, %355, %320 : <ui32>, <ui32> -> <ui32>
                  %357 = ascvec.shr %dst_22, 14, %320 : <ui32> -> <ui32>
                  %358 = ascvec.duplicate_scalar %352, %320 : i32 -> <ui32>
                  %359 = ascvec.duplicate_scalar %353, %320 : i32 -> <ui32>
                  %carry_23, %dst_24 = ascvec.addc %358, %356, %320 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %carry_25, %dst_26 = ascvec.addc %359, %357, %carry_23, %320 : <ui32>, <ui32> -> !ascvec.mask, <ui32>
                  %360 = ascvec.duplicate_scalar %c16383_i32, %320 : i32 -> <ui32>
                  %361 = ascvec.and %dst, %360, %320 : <ui32>, <ui32> -> <ui32>
                  %362 = ascvec.duplicate_scalar %c-766435501_i32, %320 : i32 -> <ui32>
                  %363 = ascvec.duplicate_scalar %c-845247145_i32, %320 : i32 -> <ui32>
                  %dst0, %dst1 = ascvec.mull %337, %362, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_27, %dst1_28 = ascvec.mull %dst_24, %363, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %364 = ascvec.xor %dst1_28, %361, %320 : <ui32>, <ui32> -> <ui32>
                  %365 = ascvec.xor %dst1, %dst_26, %320 : <ui32>, <ui32> -> <ui32>
                  %366 = ascvec.duplicate_scalar %arg11, %320 : i32 -> <ui32>
                  %367 = ascvec.duplicate_scalar %arg12, %320 : i32 -> <ui32>
                  %368 = ascvec.xor %364, %366, %320 : <ui32>, <ui32> -> <ui32>
                  %369 = ascvec.xor %365, %367, %320 : <ui32>, <ui32> -> <ui32>
                  %dst0_29, %dst1_30 = ascvec.mull %368, %362, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_31, %dst1_32 = ascvec.mull %369, %363, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %370 = ascvec.xor %dst1_32, %dst0_27, %320 : <ui32>, <ui32> -> <ui32>
                  %371 = ascvec.xor %dst1_30, %dst0, %320 : <ui32>, <ui32> -> <ui32>
                  %372 = arith.addi %arg11, %c-1640531527_i32_15 : i32
                  %373 = ascvec.duplicate_scalar %372, %320 : i32 -> <ui32>
                  %374 = arith.addi %arg12, %c-1150833019_i32_14 : i32
                  %375 = ascvec.duplicate_scalar %374, %320 : i32 -> <ui32>
                  %376 = ascvec.xor %370, %373, %320 : <ui32>, <ui32> -> <ui32>
                  %377 = ascvec.xor %371, %375, %320 : <ui32>, <ui32> -> <ui32>
                  %dst0_33, %dst1_34 = ascvec.mull %376, %362, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_35, %dst1_36 = ascvec.mull %377, %363, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %378 = ascvec.xor %dst1_36, %dst0_31, %320 : <ui32>, <ui32> -> <ui32>
                  %379 = ascvec.xor %dst1_34, %dst0_29, %320 : <ui32>, <ui32> -> <ui32>
                  %380 = arith.addi %arg11, %c1013904242_i32_13 : i32
                  %381 = ascvec.duplicate_scalar %380, %320 : i32 -> <ui32>
                  %382 = arith.addi %arg12, %c1993301258_i32_12 : i32
                  %383 = ascvec.duplicate_scalar %382, %320 : i32 -> <ui32>
                  %384 = ascvec.xor %378, %381, %320 : <ui32>, <ui32> -> <ui32>
                  %385 = ascvec.xor %379, %383, %320 : <ui32>, <ui32> -> <ui32>
                  %dst0_37, %dst1_38 = ascvec.mull %384, %362, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_39, %dst1_40 = ascvec.mull %385, %363, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %386 = ascvec.xor %dst1_40, %dst0_35, %320 : <ui32>, <ui32> -> <ui32>
                  %387 = ascvec.xor %dst1_38, %dst0_33, %320 : <ui32>, <ui32> -> <ui32>
                  %388 = arith.addi %arg11, %c-626627285_i32 : i32
                  %389 = ascvec.duplicate_scalar %388, %320 : i32 -> <ui32>
                  %390 = arith.addi %arg12, %c842468239_i32 : i32
                  %391 = ascvec.duplicate_scalar %390, %320 : i32 -> <ui32>
                  %392 = ascvec.xor %386, %389, %320 : <ui32>, <ui32> -> <ui32>
                  %393 = ascvec.xor %387, %391, %320 : <ui32>, <ui32> -> <ui32>
                  %dst0_41, %dst1_42 = ascvec.mull %392, %362, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_43, %dst1_44 = ascvec.mull %393, %363, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %394 = ascvec.xor %dst1_44, %dst0_39, %320 : <ui32>, <ui32> -> <ui32>
                  %395 = ascvec.xor %dst1_42, %dst0_37, %320 : <ui32>, <ui32> -> <ui32>
                  %396 = arith.addi %arg11, %c2027808484_i32 : i32
                  %397 = ascvec.duplicate_scalar %396, %320 : i32 -> <ui32>
                  %398 = arith.addi %arg12, %c-308364780_i32 : i32
                  %399 = ascvec.duplicate_scalar %398, %320 : i32 -> <ui32>
                  %400 = ascvec.xor %394, %397, %320 : <ui32>, <ui32> -> <ui32>
                  %401 = ascvec.xor %395, %399, %320 : <ui32>, <ui32> -> <ui32>
                  %dst0_45, %dst1_46 = ascvec.mull %400, %362, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_47, %dst1_48 = ascvec.mull %401, %363, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %402 = ascvec.xor %dst1_48, %dst0_43, %320 : <ui32>, <ui32> -> <ui32>
                  %403 = ascvec.xor %dst1_46, %dst0_41, %320 : <ui32>, <ui32> -> <ui32>
                  %404 = arith.addi %arg11, %c387276957_i32 : i32
                  %405 = ascvec.duplicate_scalar %404, %320 : i32 -> <ui32>
                  %406 = arith.addi %arg12, %c-1459197799_i32 : i32
                  %407 = ascvec.duplicate_scalar %406, %320 : i32 -> <ui32>
                  %408 = ascvec.xor %402, %405, %320 : <ui32>, <ui32> -> <ui32>
                  %409 = ascvec.xor %403, %407, %320 : <ui32>, <ui32> -> <ui32>
                  %dst0_49, %dst1_50 = ascvec.mull %408, %362, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_51, %dst1_52 = ascvec.mull %409, %363, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %410 = ascvec.xor %dst1_52, %dst0_47, %320 : <ui32>, <ui32> -> <ui32>
                  %411 = ascvec.xor %dst1_50, %dst0_45, %320 : <ui32>, <ui32> -> <ui32>
                  %412 = arith.addi %arg11, %c-1253254570_i32 : i32
                  %413 = ascvec.duplicate_scalar %412, %320 : i32 -> <ui32>
                  %414 = arith.addi %arg12, %c1684936478_i32 : i32
                  %415 = ascvec.duplicate_scalar %414, %320 : i32 -> <ui32>
                  %416 = ascvec.xor %410, %413, %320 : <ui32>, <ui32> -> <ui32>
                  %417 = ascvec.xor %411, %415, %320 : <ui32>, <ui32> -> <ui32>
                  %dst0_53, %dst1_54 = ascvec.mull %416, %362, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_55, %dst1_56 = ascvec.mull %417, %363, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %418 = ascvec.xor %dst1_56, %dst0_51, %320 : <ui32>, <ui32> -> <ui32>
                  %419 = ascvec.xor %dst1_54, %dst0_49, %320 : <ui32>, <ui32> -> <ui32>
                  %420 = arith.addi %arg11, %c1401181199_i32 : i32
                  %421 = ascvec.duplicate_scalar %420, %320 : i32 -> <ui32>
                  %422 = arith.addi %arg12, %c534103459_i32 : i32
                  %423 = ascvec.duplicate_scalar %422, %320 : i32 -> <ui32>
                  %424 = ascvec.xor %418, %421, %320 : <ui32>, <ui32> -> <ui32>
                  %425 = ascvec.xor %419, %423, %320 : <ui32>, <ui32> -> <ui32>
                  %dst0_57, %dst1_58 = ascvec.mull %424, %362, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_59, %dst1_60 = ascvec.mull %425, %363, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %426 = ascvec.xor %dst1_60, %dst0_55, %320 : <ui32>, <ui32> -> <ui32>
                  %427 = ascvec.xor %dst1_58, %dst0_53, %320 : <ui32>, <ui32> -> <ui32>
                  %428 = arith.addi %arg11, %c-239350328_i32 : i32
                  %429 = ascvec.duplicate_scalar %428, %320 : i32 -> <ui32>
                  %430 = arith.addi %arg12, %c-616729560_i32 : i32
                  %431 = ascvec.duplicate_scalar %430, %320 : i32 -> <ui32>
                  %432 = ascvec.xor %426, %429, %320 : <ui32>, <ui32> -> <ui32>
                  %433 = ascvec.xor %427, %431, %320 : <ui32>, <ui32> -> <ui32>
                  %dst0_61, %dst1_62 = ascvec.mull %432, %362, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %dst0_63, %dst1_64 = ascvec.mull %433, %363, %320 : <ui32>, <ui32> -> <ui32>, <ui32>
                  %434 = ascvec.xor %dst1_64, %dst0_59, %320 : <ui32>, <ui32> -> <ui32>
                  %435 = ascvec.xor %dst1_62, %dst0_57, %320 : <ui32>, <ui32> -> <ui32>
                  %436 = arith.addi %arg11, %c-1879881855_i32 : i32
                  %437 = ascvec.duplicate_scalar %436, %320 : i32 -> <ui32>
                  %438 = arith.addi %arg12, %c-1767562579_i32 : i32
                  %439 = ascvec.duplicate_scalar %438, %320 : i32 -> <ui32>
                  %440 = ascvec.xor %434, %437, %320 : <ui32>, <ui32> -> <ui32>
                  %441 = ascvec.xor %435, %439, %320 : <ui32>, <ui32> -> <ui32>
                  %442 = ascvec.reinterpret %440 : <ui32> -> <i32>
                  %443 = arith.index_cast %319 : i32 to index
                  ascvec.storealign %arg13[%443], %442, %320 : memref<8192xi32, 1>, index, <i32>
                  %444 = arith.addi %319, %c64_i32 : i32
                  %445 = ascvec.reinterpret %dst0_63 : <ui32> -> <i32>
                  %446 = arith.index_cast %444 : i32 to index
                  ascvec.storealign %arg13[%446], %445, %320 : memref<8192xi32, 1>, index, <i32>
                  %447 = arith.addi %319, %c128_i32 : i32
                  %448 = ascvec.reinterpret %441 : <ui32> -> <i32>
                  %449 = arith.index_cast %447 : i32 to index
                  ascvec.storealign %arg13[%449], %448, %320 : memref<8192xi32, 1>, index, <i32>
                  %450 = arith.addi %319, %c192_i32 : i32
                  %451 = ascvec.reinterpret %dst0_61 : <ui32> -> <i32>
                  %452 = arith.index_cast %450 : i32 to index
                  ascvec.storealign %arg13[%452], %451, %320 : memref<8192xi32, 1>, index, <i32>
                  %453 = arith.addi %318, %arg14 : i64
                  %454 = arith.cmpi sge, %453, %arg9 : i64
                  %455 = arith.extui %454 : i1 to i64
                  %456 = arith.addi %314, %arg15 : i64
                  %457 = arith.addi %456, %455 : i64
                  %458 = arith.muli %455, %arg9 : i64
                  %459 = arith.subi %453, %458 : i64
                  %460 = arith.trunci %457 : i64 to i32
                  %461 = arith.extui %460 : i32 to i64
                  %462 = arith.subi %457, %461 : i64
                  %463 = arith.remsi %462, %c4294967296_i64_20 : i64
                  %464 = arith.cmpi ne, %463, %c0_i64_18 : i64
                  %465 = arith.cmpi slt, %463, %c0_i64_18 : i64
                  %466 = arith.andi %464, %465 : i1
                  %467 = arith.divsi %462, %c4294967296_i64_20 : i64
                  %468 = arith.extui %466 : i1 to i64
                  %469 = arith.subi %467, %468 : i64
                  %470 = arith.trunci %469 : i64 to i32
                  %471 = arith.trunci %459 : i64 to i32
                  %472 = arith.extui %471 : i32 to i64
                  %473 = arith.subi %459, %472 : i64
                  %474 = arith.remsi %473, %c4294967296_i64_20 : i64
                  %475 = arith.cmpi ne, %474, %c0_i64_18 : i64
                  %476 = arith.cmpi slt, %474, %c0_i64_18 : i64
                  %477 = arith.andi %475, %476 : i1
                  %478 = arith.divsi %473, %c4294967296_i64_20 : i64
                  %479 = arith.extui %477 : i1 to i64
                  %480 = arith.subi %478, %479 : i64
                  %481 = arith.trunci %480 : i64 to i32
                  scf.yield %460, %470, %471, %481 : i32, i32, i32, i32
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              %305 = arith.trunci %227 : i64 to i32
              %306 = arith.trunci %99 : i64 to i32
              %307 = arith.trunci %32 : i64 to i32
              %308 = arith.muli %305, %c2_i32 : i32
              %309 = arith.cmpi eq, %307, %c4_i32 : i32
              ascvec.get_buf v, %105, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%99, %134, %alloc_9, %alloc_6, %alloc_8, %305, %306, %308, %309, %alloc_7 : i64, memref<32768xbf16, 1>, memref<8192xi32, 1>, memref<32768xui8, 1>, memref<1024xui32, 1>, i32, i32, i32, i1, memref<2048xui8, 1>) {
              ^bb0(%arg9: i64, %arg10: memref<32768xbf16, 1>, %arg11: memref<8192xi32, 1>, %arg12: memref<32768xui8, 1>, %arg13: memref<1024xui32, 1>, %arg14: i32, %arg15: i32, %arg16: i32, %arg17: i1, %arg18: memref<2048xui8, 1>):
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
                %310 = ascvec.create_mask : !ascvec.mask
                %311 = ascvec.arange %c0_i32_16 : i32 -> <ui32>
                %312 = ascvec.cmp_lt_scalar %311, %c32_i32, %310 : <ui32>, i32 -> !ascvec.mask
                %313 = ascvec.cmp_ge_scalar %311, %c32_i32, %310 : <ui32>, i32 -> !ascvec.mask
                scf.for %arg19 = %c0_i32_16 to %arg14 step %c1_i32_13  : i32 {
                  %315 = arith.muli %arg19, %c64_i32 : i32
                  %316 = arith.trunci %arg9 : i64 to i32
                  %317 = arith.subi %316, %315 : i32
                  %mask, %remaining_out = ascvec.update_mask %317 : !ascvec.mask
                  %318 = arith.index_cast %315 : i32 to index
                  %319 = ascvec.loadalign_unpack %arg10[%318] {mode = "b16_to_b32"} : memref<32768xbf16, 1>, index -> <bf16>
                  %320 = ascvec.cast %319, %310 {rounding = 6 : i32} : <bf16> -> <f32>
                  %321 = ascvec.duplicate_scalar %cst_14, %310 : f32 -> <f32>
                  %322 = ascvec.select %320, %321, %mask : <f32>
                  %323 = ascvec.reinterpret %322 : <f32> -> <ui32>
                  %324 = ascvec.duplicate_scalar %c2147483647_i32, %310 : i32 -> <ui32>
                  %325 = ascvec.and %323, %324, %310 : <ui32>, <ui32> -> <ui32>
                  %326 = ascvec.reduce_max %325, %312 : <ui32>
                  %327 = ascvec.duplicate %326, %310 : <ui32> -> <ui32>
                  %328 = ascvec.reduce_max %325, %313 : <ui32>
                  %329 = ascvec.duplicate %328, %310 : <ui32> -> <ui32>
                  %330 = ascvec.select %327, %329, %312 : <ui32>
                  %331 = ascvec.shr %330, 23, %310 : <ui32> -> <ui32>
                  %332 = ascvec.cmp_ge_scalar %331, %c255_i32, %310 : <ui32>, i32 -> !ascvec.mask
                  %333 = ascvec.max_scalar %331, %c15_i32, %310 : <ui32>, i32
                  %334 = ascvec.add_scalar %333, %c-15_i32, %310 : <ui32>, i32
                  %335 = ascvec.duplicate_scalar %c255_i32, %332 merge %334 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %336 = ascvec.duplicate_scalar %c254_i32, %310 : i32 -> <ui32>
                  %337 = ascvec.sub %336, %335, %310 : <ui32>
                  %338 = ascvec.cmp_ne_scalar %335, %c0_i32_16, %310 : <ui32>, i32 -> !ascvec.mask
                  %339 = ascvec.shl %337, 23, %338 : <ui32> -> <ui32>
                  %340 = ascvec.duplicate_scalar %c2139160576_i32, %332 merge %339 : !ascvec.vreg<ui32> {mode = "merging"} : i32 -> <ui32>
                  %341 = ascvec.reinterpret %340 : <ui32> -> <f32>
                  %342 = ascvec.mul %322, %341, %310 : <f32>
                  %343 = ascvec.shr %311, 2, %310 : <ui32> -> <ui32>
                  %344 = ascvec.duplicate_scalar %c3_i32, %310 : i32 -> <ui32>
                  %345 = ascvec.and %343, %344, %310 : <ui32>, <ui32> -> <ui32>
                  %346 = ascvec.shl %345, 6, %310 : <ui32> -> <ui32>
                  %347 = ascvec.shr %311, 4, %310 : <ui32> -> <ui32>
                  %348 = ascvec.add %346, %347, %310 : <ui32>
                  %349 = arith.remsi %arg19, %c16_i32 : i32
                  %350 = arith.cmpi ne, %349, %c0_i32_16 : i32
                  %351 = arith.cmpi slt, %349, %c0_i32_16 : i32
                  %352 = arith.andi %350, %351 : i1
                  %353 = arith.divsi %arg19, %c16_i32 : i32
                  %354 = arith.extui %352 : i1 to i32
                  %355 = arith.subi %353, %354 : i32
                  %356 = arith.muli %355, %c256_i32 : i32
                  %357 = arith.select %352, %c16_i32, %c0_i32_16 : i32
                  %358 = arith.addi %349, %357 overflow<nsw> : i32
                  %359 = arith.muli %358, %c4_i32_12 overflow<nsw> : i32
                  %360 = arith.addi %356, %359 : i32
                  %361 = ascvec.add_scalar %348, %360, %310 : <ui32>, i32
                  %362 = ascvec.gather %arg11[%c0_15], %361, %310 : memref<8192xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                  %363 = ascvec.reinterpret %362 : <i32> -> <ui32>
                  %364 = ascvec.shl %363, 8, %310 : <ui32> -> <ui32>
                  %365 = ascvec.shr %363, 24, %310 : <ui32> -> <ui32>
                  %366 = ascvec.or %364, %365, %310 : <ui32>, <ui32> -> <ui32>
                  %367 = ascvec.and %311, %344, %310 : <ui32>, <ui32> -> <ui32>
                  %368 = ascvec.duplicate_scalar %c1_i32_13, %310 : i32 -> <ui32>
                  %369 = ascvec.and %367, %368, %310 : <ui32>, <ui32> -> <ui32>
                  %370 = ascvec.cmp_eq_scalar %369, %c0_i32_16, %310 : <ui32>, i32 -> !ascvec.mask
                  %371 = ascvec.select %363, %366, %370 : <ui32>
                  %372 = ascvec.cmp_eq_scalar %367, %c0_i32_16, %310 : <ui32>, i32 -> !ascvec.mask
                  %373 = ascvec.cmp_eq_scalar %367, %c3_i32, %310 : <ui32>, i32 -> !ascvec.mask
                  %374 = ascvec.mask_or %372, %373, %310
                  %375 = ascvec.shr %371, 16, %310 : <ui32> -> <ui32>
                  %376 = ascvec.duplicate_scalar %c65535_i32, %310 : i32 -> <ui32>
                  %377 = ascvec.and %371, %376, %310 : <ui32>, <ui32> -> <ui32>
                  %378 = ascvec.select %375, %377, %374 : <ui32>
                  %379 = ascvec.reinterpret %342 : <f32> -> <ui32>
                  %380 = ascvec.shr %379, 24, %310 : <ui32> -> <ui32>
                  %381 = ascvec.duplicate_scalar %c128_i32, %310 : i32 -> <ui32>
                  %382 = ascvec.and %380, %381, %310 : <ui32>, <ui32> -> <ui32>
                  %383 = ascvec.and %379, %324, %310 : <ui32>, <ui32> -> <ui32>
                  %384 = ascvec.shr %383, 5, %310 : <ui32> -> <ui32>
                  %385 = ascvec.duplicate_scalar %c29360128_i32, %310 : i32 -> <ui32>
                  %386 = ascvec.sub %384, %385, %310 : <ui32>
                  %387 = ascvec.reinterpret %383 : <ui32> -> <f32>
                  %388 = ascvec.mul_scalar %387, %cst, %310 : <f32>, f32
                  %389 = ascvec.cast %388, %310 {rounding = 4 : i32} : <f32> -> <i32>
                  %390 = ascvec.reinterpret %389 : <i32> -> <ui32>
                  %391 = ascvec.cmp_ge_scalar %383, %c947912704_i32, %310 : <ui32>, i32 -> !ascvec.mask
                  %392 = ascvec.select %386, %390, %391 : <ui32>
                  %393 = ascvec.and %378, %376, %310 : <ui32>, <ui32> -> <ui32>
                  %394 = ascvec.add %392, %393, %310 : <ui32>
                  %395 = ascvec.shr %394, 16, %310 : <ui32> -> <ui32>
                  %396 = ascvec.min_scalar %395, %c123_i32, %310 : <ui32>, i32
                  %397 = ascvec.or %396, %382, %310 : <ui32>, <ui32> -> <ui32>
                  %398 = ascvec.duplicate_scalar %c127_i32, %310 : i32 -> <ui32>
                  %399 = ascvec.cmp_gt_scalar %383, %c2139095040_i32, %310 : <ui32>, i32 -> !ascvec.mask
                  %400 = ascvec.select %398, %397, %399 : <ui32>
                  ascvec.storealign_pack %arg12[%318], %400, %mask {mode = "b32_to_b8"} : memref<32768xui8, 1>, index, <ui32>
                  %401 = arith.muli %arg19, %c2_i32_11 : i32
                  %402 = arith.index_cast %401 : i32 to index
                  ascvec.storealign_1st %arg13[%402], %335 : memref<1024xui32, 1>, index, <ui32>
                  %403 = ascvec.duplicate_scalar %c32_i32, %310 : i32 -> <ui32>
                  %404 = ascvec.gather_reg %335, %403 : <ui32>, <ui32> -> <ui32>
                  %405 = arith.addi %401, %c1_i32_13 : i32
                  %406 = arith.index_cast %405 : i32 to index
                  ascvec.storealign_1st %arg13[%406], %404 : memref<1024xui32, 1>, index, <ui32>
                }
                ascvec.mem_bar vst_vld
                %314 = scf.if %arg17 -> (i32) {
                  %315 = arith.remsi %arg15, %c32_i32 : i32
                  %316 = arith.cmpi ne, %315, %c0_i32_16 : i32
                  %317 = arith.cmpi slt, %315, %c0_i32_16 : i32
                  %318 = arith.andi %316, %317 : i1
                  %319 = arith.divsi %arg15, %c32_i32 : i32
                  %320 = arith.extui %318 : i1 to i32
                  %321 = arith.subi %319, %320 : i32
                  %322 = arith.muli %321, %c2_i32_11 : i32
                  scf.yield %322 : i32
                } else {
                  scf.yield %arg16 : i32
                }
                scf.for %arg19 = %c0_i32_16 to %314 step %c64_i32  : i32 {
                  %315 = arith.subi %314, %arg19 : i32
                  %mask, %remaining_out = ascvec.update_mask %315 : !ascvec.mask
                  %316 = ascvec.add_scalar %311, %arg19, %310 : <ui32>, i32
                  scf.if %arg17 {
                    %317 = ascvec.shr %316, 1, %310 : <ui32> -> <ui32>
                    %318 = ascvec.gather %arg13[%c0_15], %317, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %319 = ascvec.duplicate_scalar %c1_i32_13, %310 : i32 -> <ui32>
                    %320 = ascvec.and %311, %319, %310 : <ui32>, <ui32> -> <ui32>
                    %321 = ascvec.cmp_eq_scalar %320, %c0_i32_16, %310 : <ui32>, i32 -> !ascvec.mask
                    %322 = ascvec.duplicate_scalar %c0_i32_16, %310 : i32 -> <ui32>
                    %323 = ascvec.select %318, %322, %321 : <ui32>
                    %324 = arith.index_cast %arg19 : i32 to index
                    %c0_17 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg20 = %c0_17 to %c1 step %c1 {
                      ascvec.storealign_pack %arg18[%324], %323, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  } else {
                    %317 = ascvec.gather %arg13[%c0_15], %316, %mask : memref<1024xui32, 1>, <ui32>, !ascvec.mask -> <ui32>
                    %318 = arith.index_cast %arg19 : i32 to index
                    %c0_17 = arith.constant 0 : index
                    %c1 = arith.constant 1 : index
                    scf.for %arg20 = %c0_17 to %c1 step %c1 {
                      ascvec.storealign_pack %arg18[%318], %317, %mask {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <ui32>
                    }
                  }
                }
                ascvec.mem_bar vst_vld
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %105, 0 {cannir.buffer.generated_sync}
            }
          }
          %139 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %140 = cannir.extract_shape %139 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, -1)>
          %141 = cannir.extract_dim %140[0] : !cannir.shape<(-1, -1)>
          %142 = arith.minsi %93, %141 : i64
          %143 = arith.subi %141, %142 : i64
          %144 = arith.minsi %143, %c1_i64 : i64
          %145 = arith.cmpi sle, %93, %141 : i64
          %146 = arith.select %145, %144, %c0_i64 : i64
          %147 = arith.cmpi sgt, %146, %c1_i64 : i64
          %148 = arith.select %147, %c1_i64, %146 : i64
          %149 = cannir.extract_dim %140[1] : !cannir.shape<(-1, -1)>
          %150 = arith.divsi %149, %117 : i64
          %151 = arith.minsi %95, %150 : i64
          %152 = arith.muli %151, %70 : i64
          %153 = arith.subi %149, %152 : i64
          %154 = arith.minsi %70, %153 : i64
          %155 = arith.cmpi sle, %95, %150 : i64
          %156 = arith.select %155, %154, %c0_i64 : i64
          %157 = arith.cmpi sgt, %156, %c32768_i64 : i64
          %158 = arith.select %157, %c32768_i64, %156 : i64
          %159 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %160 = cannir.extract_stride %159 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %161 = cannir.extract_dim %160[0] : !cannir.stride<(-1, 1)>
          %162 = arith.muli %93, %161 : i64
          %163 = arith.addi %162, %96 : i64
          %164 = arith.muli %163, %c8_i64 : i64
          %165 = ascscalar.memref_address %alloc_0 : memref<1xui8>
          %166 = arith.divui %164, %c8_i64 : i64
          %167 = arith.addi %165, %166 : i64
          %168 = ascscalar.bind_gm_ptr %167 : memref<1xui8>
          %169 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<32768xui8, 1> -> memref<?xui8, 1>
          %170 = cannir.extract_stride %139 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %171 = cannir.extract_dim %170[0] : !cannir.stride<(-1, 1)>
          %172 = arith.cmpi ugt, %148, %c0_i64 : i64
          %173 = arith.cmpi ugt, %158, %c0_i64 : i64
          %174 = arith.andi %172, %173 : i1
          scf.if %174 {
            ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
            %211 = arith.trunci %148 : i64 to i32
            %212 = arith.trunci %158 : i64 to i32
            ascvec.copy_ub2gm %168[%c0], %169[%c0], %211, %212, %171, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
          }
          %175 = arith.muli %69, %32 : i64
          %176 = cannir.extract_dim %22[0] : !cannir.shape<(-1, -1)>
          %177 = arith.minsi %93, %176 : i64
          %178 = arith.subi %176, %177 : i64
          %179 = arith.minsi %178, %c1_i64 : i64
          %180 = arith.cmpi sle, %93, %176 : i64
          %181 = arith.select %180, %179, %c0_i64 : i64
          %182 = arith.cmpi sgt, %181, %c1_i64 : i64
          %183 = arith.select %182, %c1_i64, %181 : i64
          %184 = arith.maxsi %175, %c1_i64 : i64
          %185 = arith.divsi %23, %184 : i64
          %186 = arith.minsi %95, %185 : i64
          %187 = arith.muli %186, %175 : i64
          %188 = arith.subi %23, %187 : i64
          %189 = arith.minsi %175, %188 : i64
          %190 = arith.cmpi sle, %95, %185 : i64
          %191 = arith.select %190, %189, %c0_i64 : i64
          %192 = arith.cmpi sgt, %191, %c2048_i64 : i64
          %193 = arith.select %192, %c2048_i64, %191 : i64
          %194 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, ui8>, <<(-1, -1)>, <(-1, 1)>>, <<(-1, -1)>, <(-1, 1)>>, identity>
          %195 = cannir.extract_stride %194 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %196 = cannir.extract_dim %195[0] : !cannir.stride<(-1, 1)>
          %197 = arith.muli %93, %196 : i64
          %198 = arith.muli %95, %175 : i64
          %199 = arith.addi %197, %198 : i64
          %200 = arith.muli %199, %c8_i64 : i64
          %201 = ascscalar.memref_address %alloc_1 : memref<1xui8>
          %202 = arith.divui %200, %c8_i64 : i64
          %203 = arith.addi %201, %202 : i64
          %204 = ascscalar.bind_gm_ptr %203 : memref<1xui8>
          %205 = ascscalar.bitcast_memref %alloc_7 byte_offset = %c0_i64 : memref<2048xui8, 1> -> memref<?xui8, 1>
          %206 = cannir.extract_stride %21 : <<(-1, -1)>, <(-1, 1)>> -> <(-1, 1)>
          %207 = cannir.extract_dim %206[0] : !cannir.stride<(-1, 1)>
          %208 = arith.cmpi ugt, %183, %c0_i64 : i64
          %209 = arith.cmpi ugt, %193, %c0_i64 : i64
          %210 = arith.andi %208, %209 : i1
          scf.if %210 {
            ascvec.get_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
            %211 = arith.trunci %183 : i64 to i32
            %212 = arith.trunci %193 : i64 to i32
            ascvec.copy_ub2gm %204[%c0], %205[%c0], %211, %212, %207, %c2048_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
            ascvec.release_buf mte3, %c3_i64, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %107 : i64
        }
      }
    }
    return
  }
}
