module {
  func.func @run_df9c3635df26c5e9e9f0e538444fa00d(%arg0: !cannir.tensor<<gm, ui64>, <<(5)>, <(1)>>, <<(5)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, ui64>, <<(15)>, <(1)>>, <<(15)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, i64>, <<(3)>, <(1)>>, <<(3)>, <(1)>>, identity>, %arg3: !cannir.tensor<<gm, ui64>, <<(15)>, <(1)>>, <<(15)>, <(1)>>, identity>, %arg4: i64) attributes {cannbotdsl.contract_digest = "df9c3635df26c5e9e9f0e538444fa00dbf5d6a433b87562a76cb12d038855dbe", cannbotdsl.public_slot_count = 5 : i64, cannbotdsl.slot_dtype_bits = [64, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [1, 1, 0, 1, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 2], cannbotdsl.slot_type_digests = ["572dc042accdc771a372cd58a2b40df8df96e69f57a2d8afaa9952f47617db18", "2479edd870911dc5030c5df0a39d58d609ea81b6aefb96a2db5f061de5c2baa6", "b733892275f62059f4d6b5d67018aadcbf4dcf085d9fc1c5bca1efc3a35629d1", "ef5b946132df9afa9cb803c1ae21c45b2e7307a5b20f7ddf92b659cb780efbbe", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %c24_i64 = arith.constant 24 : i64
    cannir.kernel_launch @kernel_gmm_kernel_0(%c24_i64) [%arg0, %arg1, %arg2, %arg3, %arg4] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3>, name = "gmm_kernel", op_type = "gmm_kernel", roles = array<i64: 0, 0, 0, 0>}} : !cannir.tensor<<gm, ui64>, <<(5)>, <(1)>>, <<(5)>, <(1)>>, identity>, !cannir.tensor<<gm, ui64>, <<(15)>, <(1)>>, <<(15)>, <(1)>>, identity>, !cannir.tensor<<gm, i64>, <<(3)>, <(1)>>, <<(3)>, <(1)>>, identity>, !cannir.tensor<<gm, ui64>, <<(15)>, <(1)>>, <<(15)>, <(1)>>, identity>, i64
    return
  }
  func.func @kernel_gmm_kernel_0(%arg0: !cannir.tensor<<gm, ui64>, <<(5)>, <(1)>>, <<(5)>, <(1)>>, identity>, %arg1: !cannir.tensor<<gm, ui64>, <<(15)>, <(1)>>, <<(15)>, <(1)>>, identity>, %arg2: !cannir.tensor<<gm, i64>, <<(3)>, <(1)>>, <<(3)>, <(1)>>, identity>, %arg3: !cannir.tensor<<gm, ui64>, <<(15)>, <(1)>>, <<(15)>, <(1)>>, identity>, %arg4: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_gmm_kernel_0", cannir.resource.ids_prepared} {
    %c32_i32 = arith.constant 32 : i32
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %c128_i32 = arith.constant 128 : i32
    %c0 = arith.constant 0 : index
    %c24_i64 = arith.constant 24 : i64
    %c2251799813685248_i64 = arith.constant 2251799813685248 : i64
    %false = arith.constant false
    %c128_i64 = arith.constant 128 : i64
    %c32_i64 = arith.constant 32 : i64
    %c64_i64 = arith.constant 64 : i64
    %c16_i64 = arith.constant 16 : i64
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c15_i64 = arith.constant 15 : i64
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index
    %c15 = arith.constant 15 : index
    %c127_i64 = arith.constant 127 : i64
    %c11_i64 = arith.constant 11 : i64
    %c10_i64 = arith.constant 10 : i64
    %c9_i64 = arith.constant 9 : i64
    %c8_i64 = arith.constant 8 : i64
    %c7_i64 = arith.constant 7 : i64
    %c6_i64 = arith.constant 6 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<5xui64>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<15xui64>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<3xi64>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<15xui64>
    %alloc_3 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xf32, 5>
    %alloc_4 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<4096xf16, 4>
    %alloc_5 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xf16, 4>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<4096xf16, 3>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xf16, 3>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<4096xf16, 2>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 40960 : i64} : memref<4096xf16, 2>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 57344 : i64} : memref<4096xf16, 2>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<4096xf16, 2>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xf16, 2>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<4096xf16, 2>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 24576 : i64} : memref<4096xf16, 2>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<4096xf16, 2>
    %0 = ascscalar.get_block_idx
    %1 = ascscalar.get_ctrl
    %2 = arith.remsi %1, %c2251799813685248_i64 : i64
    %3 = arith.cmpi ne, %2, %c0_i64 : i64
    %4 = arith.cmpi slt, %2, %c0_i64 : i64
    %5 = arith.andi %3, %4 : i1
    %6 = arith.divsi %1, %c2251799813685248_i64 : i64
    %7 = arith.extui %5 : i1 to i64
    %8 = arith.subi %6, %7 : i64
    %9 = arith.remsi %8, %c2_i64 : i64
    %10 = arith.cmpi ne, %9, %c0_i64 : i64
    %11 = arith.cmpi slt, %9, %c0_i64 : i64
    %12 = arith.andi %10, %11 : i1
    %13 = arith.select %12, %c2_i64, %c0_i64 : i64
    %14 = arith.addi %9, %13 overflow<nsw> : i64
    %15 = arith.subi %c1_i64, %14 overflow<nsw> : i64
    %16 = arith.muli %15, %c2251799813685248_i64 overflow<nsw> : i64
    %17 = arith.addi %1, %16 : i64
    ascscalar.set_ctrl %17 : i64
    %18:4 = scf.for %arg5 = %c0_i64 to %arg4 step %c1_i64 iter_args(%arg6 = %c0_i64, %arg7 = %c0_i64, %arg8 = %c0_i64, %arg9 = %c0_i64) -> (i64, i64, i64, i64)  : i64 {
      %19 = memref.load %alloc[%c0] : memref<5xui64>
      %20 = cannir.scalar_storage_cast %19 : ui64 -> i64
      %21 = ascscalar.bind_gm_ptr %20 : memref<7488xf16>
      %22 = arith.muli %arg5, %c5_i64 : i64
      %23 = arith.index_cast %22 : i64 to index
      %24 = memref.load %alloc_0[%23] : memref<15xui64>
      %25 = cannir.scalar_storage_cast %24 : ui64 -> i64
      %26 = ascscalar.bind_gm_ptr %25 : memref<1xf16>
      %27 = arith.index_cast %22 : i64 to index
      %28 = memref.load %alloc_2[%27] : memref<15xui64>
      %29 = cannir.scalar_storage_cast %28 : ui64 -> i64
      %30 = ascscalar.bind_gm_ptr %29 : memref<8192xf16>
      %31 = arith.index_cast %arg5 : i64 to index
      %32 = memref.load %alloc_1[%31] : memref<3xi64>
      %33 = arith.subi %32, %arg7 : i64
      %34 = arith.cmpi sgt, %33, %c0_i64 : i64
      %35:3 = scf.if %34 -> (i64, i64, i64) {
        %36 = arith.divsi %33, %c128_i64 : i64
        %37 = arith.remsi %33, %c128_i64 : i64
        %38 = arith.cmpi ne, %37, %c0_i64 : i64
        %39 = arith.extui %38 : i1 to i64
        %40 = arith.addi %36, %39 : i64
        %41 = arith.addi %arg6, %c8_i64 overflow<nsw> : i64
        %42 = arith.cmpi sge, %0, %arg6 : i64
        %43 = arith.addi %0, %c24_i64 : i64
        %44 = arith.select %42, %0, %43 : i64
        %45:2 = scf.for %arg10 = %44 to %41 step %c24_i64 iter_args(%arg11 = %arg8, %arg12 = %arg9) -> (i64, i64)  : i64 {
          %47 = arith.subi %arg10, %arg6 : i64
          %48 = arith.remsi %47, %c4_i64 : i64
          %49 = arith.cmpi ne, %48, %c0_i64 : i64
          %50 = arith.cmpi slt, %48, %c0_i64 : i64
          %51 = arith.andi %49, %50 : i1
          %52 = arith.divsi %47, %c4_i64 : i64
          %53 = arith.extui %51 : i1 to i64
          %54 = arith.subi %52, %53 : i64
          %55 = arith.remsi %54, %c2_i64 : i64
          %56 = arith.cmpi ne, %55, %c0_i64 : i64
          %57 = arith.cmpi slt, %55, %c0_i64 : i64
          %58 = arith.andi %56, %57 : i1
          %59 = arith.divsi %54, %c2_i64 : i64
          %60 = arith.extui %58 : i1 to i64
          %61 = arith.subi %59, %60 : i64
          %62 = arith.cmpi slt, %61, %c0_i64 : i64
          %63 = arith.select %62, %61, %c0_i64 : i64
          %64:2 = scf.if %62 -> (i64, i64) {
            %97 = arith.muli %61, %c2_i64 : i64
            %98 = arith.remsi %47, %c2_i64 : i64
            %99 = arith.cmpi ne, %98, %c0_i64 : i64
            %100 = arith.cmpi slt, %98, %c0_i64 : i64
            %101 = arith.andi %99, %100 : i1
            %102 = arith.select %101, %c2_i64, %c0_i64 : i64
            %103 = arith.addi %98, %102 overflow<nsw> : i64
            %104 = arith.addi %97, %103 : i64
            %105 = arith.divsi %47, %c2_i64 : i64
            %106 = arith.extui %101 : i1 to i64
            %107 = arith.subi %105, %106 : i64
            %108 = arith.remsi %107, %c4_i64 : i64
            %109 = arith.cmpi ne, %108, %c0_i64 : i64
            %110 = arith.cmpi slt, %108, %c0_i64 : i64
            %111 = arith.andi %109, %110 : i1
            %112 = arith.select %111, %c4_i64, %c0_i64 : i64
            %113 = arith.addi %108, %112 overflow<nsw> : i64
            scf.yield %104, %113 : i64, i64
          } else {
            %97 = arith.remsi %47, %c2_i64 : i64
            %98 = arith.cmpi ne, %97, %c0_i64 : i64
            %99 = arith.cmpi slt, %97, %c0_i64 : i64
            %100 = arith.andi %98, %99 : i1
            %101 = arith.select %100, %c2_i64, %c0_i64 : i64
            %102 = arith.addi %97, %101 overflow<nsw> : i64
            %103 = arith.divsi %47, %c2_i64 : i64
            %104 = arith.extui %100 : i1 to i64
            %105 = arith.subi %103, %104 : i64
            %106 = arith.remsi %105, %c4_i64 : i64
            %107 = arith.cmpi ne, %106, %c0_i64 : i64
            %108 = arith.cmpi slt, %106, %c0_i64 : i64
            %109 = arith.andi %107, %108 : i1
            %110 = arith.select %109, %c4_i64, %c0_i64 : i64
            %111 = arith.addi %106, %110 overflow<nsw> : i64
            scf.yield %102, %111 : i64, i64
          }
          %65 = arith.remsi %63, %c2_i64 : i64
          %66 = arith.cmpi ne, %65, %c0_i64 : i64
          %67 = arith.cmpi slt, %65, %c0_i64 : i64
          %68 = arith.andi %66, %67 : i1
          %69 = arith.select %68, %c2_i64, %c0_i64 : i64
          %70 = arith.addi %65, %69 overflow<nsw> : i64
          %71 = arith.cmpi ne, %70, %c0_i64 : i64
          %72 = arith.subi %c3_i64, %64#1 overflow<nsw> : i64
          %73 = arith.select %71, %72, %64#1 : i64
          %74 = arith.muli %64#0, %c32_i64 : i64
          %75 = arith.muli %73, %c32_i64 overflow<nsw> : i64
          %76 = arith.subi %c64_i64, %74 : i64
          %77 = arith.minsi %76, %c32_i64 : i64
          %78 = arith.subi %c128_i64, %75 overflow<nsw> : i64
          %79 = arith.minsi %78, %c32_i64 : i64
          %80:2 = scf.for %arg13 = %c0_i64 to %40 step %c1_i64 iter_args(%arg14 = %arg11, %arg15 = %arg12) -> (i64, i64)  : i64 {
            %97 = arith.muli %arg13, %c128_i64 : i64
            %98 = arith.subi %33, %97 : i64
            %99 = arith.minsi %98, %c128_i64 : i64
            %100 = arith.cmpi eq, %arg14, %c2_i64 : i64
            %101 = ascscalar.memref_address %alloc_15 : memref<4096xf16, 2>
            %102 = ascscalar.memref_address %alloc_14 : memref<4096xf16, 2>
            %103 = arith.select %100, %101, %102 : i64
            %104 = arith.cmpi eq, %arg14, %c1_i64 : i64
            %105 = ascscalar.memref_address %alloc_13 : memref<4096xf16, 2>
            %106 = arith.select %104, %105, %103 : i64
            %107 = arith.cmpi eq, %arg14, %c0_i64 : i64
            %108 = ascscalar.memref_address %alloc_12 : memref<4096xf16, 2>
            %109 = arith.select %107, %108, %106 : i64
            %110 = arith.cmpi eq, %arg14, %c2_i64 : i64
            %111 = arith.select %110, %c2_i64, %c3_i64 : i64
            %112 = arith.cmpi eq, %arg14, %c1_i64 : i64
            %113 = arith.select %112, %c1_i64, %111 : i64
            %114 = arith.cmpi eq, %arg14, %c0_i64 : i64
            %115 = arith.select %114, %c0_i64, %113 : i64
            %116 = arith.addi %arg14, %c1_i64 : i64
            %117 = arith.remui %116, %c4_i64 : i64
            %118 = ascscalar.bind_local_ptr %109 : memref<1xf16, 2>
            %119 = arith.divui %74, %c16_i64 : i64
            %120 = arith.addi %97, %arg7 : i64
            %121 = arith.muli %120, %c4_i64 : i64
            %122 = arith.addi %119, %121 : i64
            %123 = arith.muli %122, %c16_i64 : i64
            asccube.get_buf mte2, %115, 0 {cannir.buffer.generated_sync}
            %124 = arith.cmpi ugt, %99, %c0_i64 : i64
            %125 = arith.cmpi ugt, %77, %c0_i64 : i64
            %126 = arith.andi %124, %125 : i1
            scf.if %126 {
              %158 = arith.index_cast %123 : i64 to index
              %159 = arith.trunci %99 : i64 to i32
              %160 = arith.trunci %77 : i64 to i32
              asccube.copy_gm2l1 %118[%c0], %21[%158], %159, %160, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xf16, 2>, memref<7488xf16>
            }
            asccube.release_buf mte2, %115, 0 {cannir.buffer.generated_sync}
            %127 = ascscalar.memref_address %alloc_11 : memref<4096xf16, 2>
            %128 = ascscalar.memref_address %alloc_10 : memref<4096xf16, 2>
            %129 = arith.select %100, %127, %128 : i64
            %130 = ascscalar.memref_address %alloc_9 : memref<4096xf16, 2>
            %131 = arith.select %104, %130, %129 : i64
            %132 = ascscalar.memref_address %alloc_8 : memref<4096xf16, 2>
            %133 = arith.select %107, %132, %131 : i64
            %134 = arith.cmpi eq, %arg14, %c2_i64 : i64
            %135 = arith.select %134, %c6_i64, %c7_i64 : i64
            %136 = arith.cmpi eq, %arg14, %c1_i64 : i64
            %137 = arith.select %136, %c5_i64, %135 : i64
            %138 = arith.cmpi eq, %arg14, %c0_i64 : i64
            %139 = arith.select %138, %c4_i64, %137 : i64
            %140 = ascscalar.bind_local_ptr %133 : memref<1xf16, 2>
            %141 = arith.divui %97, %c16_i64 : i64
            %142 = arith.muli %141, %c128_i64 : i64
            %143 = arith.divui %75, %c16_i64 : i64
            %144 = arith.addi %142, %143 : i64
            %145 = arith.muli %144, %c16_i64 : i64
            asccube.get_buf mte2, %139, 0 {cannir.buffer.generated_sync}
            %146 = arith.cmpi ugt, %99, %c0_i64 : i64
            %147 = arith.cmpi ugt, %79, %c0_i64 : i64
            %148 = arith.andi %146, %147 : i1
            scf.if %148 {
              %158 = arith.index_cast %145 : i64 to index
              %159 = arith.trunci %99 : i64 to i32
              %160 = arith.trunci %79 : i64 to i32
              asccube.copy_gm2l1 %140[%c0], %26[%158], %159, %160, %c256_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c1_i32 {nd2nz = true} : memref<1xf16, 2>, memref<1xf16>
            }
            asccube.release_buf mte2, %139, 0 {cannir.buffer.generated_sync}
            %149 = arith.addi %99, %c127_i64 : i64
            %150 = arith.remsi %149, %c128_i64 : i64
            %151 = arith.cmpi ne, %150, %c0_i64 : i64
            %152 = arith.cmpi slt, %150, %c0_i64 : i64
            %153 = arith.andi %151, %152 : i1
            %154 = arith.divsi %149, %c128_i64 : i64
            %155 = arith.extui %153 : i1 to i64
            %156 = arith.subi %154, %155 : i64
            %157 = scf.for %arg16 = %c0_i64 to %156 step %c1_i64 iter_args(%arg17 = %arg15) -> (i64)  : i64 {
              %158 = arith.cmpi eq, %arg13, %c0_i64 : i64
              %159 = arith.cmpi eq, %arg17, %c0_i64 : i64
              %160 = ascscalar.memref_address %alloc_7 : memref<4096xf16, 3>
              %161 = ascscalar.memref_address %alloc_6 : memref<4096xf16, 3>
              %162 = arith.select %159, %160, %161 : i64
              %163 = arith.cmpi eq, %arg17, %c0_i64 : i64
              %164 = arith.select %163, %c8_i64, %c9_i64 : i64
              %165 = arith.addi %arg17, %c1_i64 : i64
              %166 = arith.remui %165, %c2_i64 : i64
              %167 = arith.divsi %77, %c32_i64 : i64
              %168 = arith.minsi %167, %c0_i64 : i64
              %169 = arith.muli %168, %c32_i64 : i64
              %170 = arith.subi %77, %169 : i64
              %171 = arith.minsi %170, %c32_i64 : i64
              %172 = arith.cmpi sge, %167, %c0_i64 : i64
              %173 = arith.select %172, %171, %c0_i64 : i64
              %174 = arith.divsi %99, %c128_i64 : i64
              %175 = arith.minsi %arg16, %174 : i64
              %176 = arith.muli %175, %c128_i64 : i64
              %177 = arith.subi %99, %176 : i64
              %178 = arith.minsi %177, %c128_i64 : i64
              %179 = arith.cmpi sle, %arg16, %174 : i64
              %180 = arith.select %179, %178, %c0_i64 : i64
              %181 = arith.divsi %173, %c16_i64 : i64
              %182 = arith.remsi %173, %c16_i64 : i64
              %183 = arith.addi %182, %c15_i64 overflow<nsw> : i64
              %184 = arith.divsi %183, %c16_i64 : i64
              %185 = arith.addi %181, %184 : i64
              %186 = arith.muli %185, %c256_i64 : i64
              %187 = ascscalar.bind_local_ptr %162 : memref<1xf16, 3>
              %188 = ascscalar.bind_local_ptr %109 : memref<1xf16, 2>
              %189 = arith.index_cast %173 : i64 to index
              %190 = arith.index_cast %180 : i64 to index
              %191 = arith.muli %arg16, %c8_i64 : i64
              %192 = arith.addi %190, %c15 : index
              %193 = arith.divui %192, %c16 : index
              %194 = arith.index_cast %193 : index to i64
              %195 = arith.addi %189, %c15 : index
              %196 = arith.divui %195, %c16 : index
              %197 = arith.index_cast %196 : index to i64
              %198 = arith.index_cast %186 : i64 to index
              %199 = arith.divui %198, %c256 : index
              %200 = arith.index_cast %199 : index to i64
              %201 = arith.cmpi ugt, %194, %c0_i64 : i64
              %202 = arith.cmpi ugt, %197, %c0_i64 : i64
              %203 = arith.andi %201, %202 : i1
              scf.if %203 {
                asccube.get_buf mte1, %164, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %115, 0 {cannir.buffer.generated_sync}
                %230 = arith.cmpi ugt, %194, %c0_i64 : i64
                %231 = arith.cmpi ugt, %197, %c0_i64 : i64
                %232 = arith.andi %230, %231 : i1
                scf.if %232 {
                  %233 = arith.trunci %191 : i64 to i32
                  %234 = arith.trunci %194 : i64 to i32
                  %235 = arith.trunci %197 : i64 to i32
                  %236 = arith.trunci %200 : i64 to i32
                  asccube.copy_l12l0a %187[%c0], %188[%c0], %233, %c0_i32, %234, %235, %c8_i32, %236 {transpose = true} : memref<1xf16, 3>, memref<1xf16, 2>
                }
                asccube.release_buf mte1, %115, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %164, 0 {cannir.buffer.generated_sync}
              }
              %204 = ascscalar.memref_address %alloc_5 : memref<4096xf16, 4>
              %205 = ascscalar.memref_address %alloc_4 : memref<4096xf16, 4>
              %206 = arith.select %159, %204, %205 : i64
              %207 = arith.cmpi eq, %arg17, %c0_i64 : i64
              %208 = arith.select %207, %c10_i64, %c11_i64 : i64
              %209 = ascscalar.bind_local_ptr %206 : memref<1xf16, 4>
              %210 = ascscalar.bind_local_ptr %133 : memref<1xf16, 2>
              %211 = arith.index_cast %180 : i64 to index
              %212 = arith.muli %arg16, %c8_i64 : i64
              %213 = arith.addi %211, %c15 : index
              %214 = arith.divui %213, %c16 : index
              %215 = arith.index_cast %214 : index to i64
              %216 = arith.cmpi ugt, %215, %c0_i64 : i64
              scf.if %216 {
                asccube.get_buf mte1, %208, 0 {cannir.buffer.generated_sync}
                asccube.get_buf mte1, %139, 0 {cannir.buffer.generated_sync}
                %230 = arith.cmpi ugt, %215, %c0_i64 : i64
                scf.if %230 {
                  %231 = arith.trunci %212 : i64 to i32
                  %232 = arith.trunci %215 : i64 to i32
                  asccube.copy_l12l0b %209[%c0], %210[%c0], %231, %c0_i32, %232, %c2_i32, %c8_i32, %c2_i32 {transpose = true} : memref<1xf16, 4>, memref<1xf16, 2>
                }
                asccube.release_buf mte1, %139, 0 {cannir.buffer.generated_sync}
                asccube.release_buf mte1, %208, 0 {cannir.buffer.generated_sync}
              }
              %217 = arith.subi %40, %c1_i64 : i64
              %218 = arith.cmpi eq, %arg13, %217 : i64
              %219 = arith.subi %156, %c1_i64 : i64
              %220 = arith.cmpi eq, %arg16, %219 : i64
              %221 = arith.select %218, %220, %false : i1
              %222 = arith.select %221, %c3_i64, %c2_i64 : i64
              %223 = arith.trunci %222 : i64 to i32
              %224 = ascscalar.bitcast_memref %alloc_3 byte_offset = %c0_i64 : memref<1024xf32, 5> -> memref<?xf32, 5>
              %225 = ascscalar.bind_local_ptr %162 : memref<1xf16, 3>
              %226 = ascscalar.bind_local_ptr %206 : memref<1xf16, 4>
              %227 = arith.cmpi ugt, %173, %c0_i64 : i64
              %228 = arith.cmpi ugt, %180, %c0_i64 : i64
              %229 = arith.andi %227, %228 : i1
              scf.if %229 {
                scf.if %158 {
                  asccube.get_buf m, %164, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf m, %208, 0 {cannir.buffer.generated_sync}
                  %230 = arith.trunci %173 : i64 to i32
                  %231 = arith.trunci %180 : i64 to i32
                  asccube.mmad %224[%c0], %225, %226, %230, %231, %c32_i32, %223 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
                  asccube.release_buf m, %208, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf m, %164, 0 {cannir.buffer.generated_sync}
                } else {
                  asccube.get_buf m, %164, 0 {cannir.buffer.generated_sync}
                  asccube.get_buf m, %208, 0 {cannir.buffer.generated_sync}
                  %230 = arith.trunci %173 : i64 to i32
                  %231 = arith.trunci %180 : i64 to i32
                  asccube.mmad %224[%c0], %225, %226, %230, %231, %c32_i32, %223 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<1xf16, 3>, memref<1xf16, 4>
                  asccube.release_buf m, %208, 0 {cannir.buffer.generated_sync}
                  asccube.release_buf m, %164, 0 {cannir.buffer.generated_sync}
                }
              }
              scf.yield %166 : i64
            }
            scf.yield %117, %157 : i64, i64
          }
          %81 = ascscalar.bitcast_memref %alloc_3 byte_offset = %c0_i64 : memref<1024xf32, 5> -> memref<?xf32, 5>
          %82 = arith.divui %74, %c16_i64 : i64
          %83 = arith.muli %82, %c128_i64 : i64
          %84 = arith.divui %75, %c16_i64 : i64
          %85 = arith.addi %83, %84 : i64
          %86 = arith.muli %85, %c16_i64 : i64
          %87 = arith.divsi %77, %c16_i64 : i64
          %88 = arith.remsi %77, %c16_i64 : i64
          %89 = arith.addi %88, %c15_i64 overflow<nsw> : i64
          %90 = arith.divsi %89, %c16_i64 : i64
          %91 = arith.addi %87, %90 : i64
          %92 = arith.muli %91, %c256_i64 : i64
          %93 = arith.index_cast %92 : i64 to index
          %94 = arith.divui %93, %c16 : index
          %95 = arith.index_cast %94 : index to i64
          %96 = arith.cmpi ugt, %77, %c0_i64 : i64
          scf.if %96 {
            %97 = arith.trunci %79 : i64 to i32
            %98 = arith.trunci %77 : i64 to i32
            %99 = arith.trunci %95 : i64 to i32
            %100 = arith.index_cast %86 : i64 to index
            asccube.copy_l0c2gm %30[%100], %81[%c0], %97, %98, %c128_i32, %99, %c3_i32 l2_cache_ctl %c1_i32 {nz2nd_en = true, relu_en = false} : memref<8192xf16>, memref<?xf32, 5>
          }
          scf.yield %80#0, %80#1 : i64, i64
        }
        %46 = arith.remsi %41, %c24_i64 : i64
        scf.yield %46, %45#0, %45#1 : i64, i64, i64
      } else {
        scf.yield %arg6, %arg8, %arg9 : i64, i64, i64
      }
      scf.yield %35#0, %32, %35#1, %35#2 : i64, i64, i64, i64
    }
    ascscalar.set_ctrl %1 : i64
    return
  }
}
