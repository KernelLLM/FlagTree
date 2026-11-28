module {
  func.func @run_c4a7ab246cfde18bfbfb72841022a2d3(%arg0: !cannir.tensor<<gm, bf16>, <<(8192, 4096)>, <(4096, 1)>>, <<(8192, 4096)>, <(4096, 1)>>, identity>, %arg1: !cannir.tensor<<gm, f32>, <<(1024, 64)>, <(64, 1)>>, <<(1024, 64)>, <(64, 1)>>, identity>, %arg2: !cannir.tensor<<gm, f32>, <<(1024, 64)>, <(64, 1)>>, <<(1024, 64)>, <(64, 1)>>, identity>, %arg3: !cannir.tensor<<gm, f8E4M3FN>, <<(8192, 4096)>, <(4096, 1)>>, <<(8192, 4096)>, <(4096, 1)>>, identity>, %arg4: !cannir.tensor<<gm, ui8>, <<(8192, 64, 2)>, <(128, 2, 1)>>, <<(8192, 64, 2)>, <(128, 2, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f8E4M3FN>, <<(5120, 8192)>, <(8192, 1)>>, <<(5120, 8192)>, <(8192, 1)>>, identity>, %arg6: !cannir.tensor<<gm, ui8>, <<(5120, 128, 2)>, <(256, 2, 1)>>, <<(5120, 128, 2)>, <(256, 2, 1)>>, identity>, %arg7: !cannir.tensor<<gm, ui8>, <<(8192, 4096)>, <(4096, 1)>>, <<(8192, 4096)>, <(4096, 1)>>, identity>, %arg8: !cannir.tensor<<gm, ui8>, <<(8192, 128)>, <(128, 1)>>, <<(8192, 128)>, <(128, 1)>>, identity>, %arg9: !cannir.tensor<<gm, bf16>, <<(1024, 8192)>, <(8192, 1)>>, <<(1024, 8192)>, <(8192, 1)>>, identity>, %arg10: !cannir.tensor<<gm, ui8>, <<(1024, 8192)>, <(8192, 1)>>, <<(1024, 8192)>, <(8192, 1)>>, identity>, %arg11: !cannir.tensor<<gm, ui8>, <<(1024, 256)>, <(256, 1)>>, <<(1024, 256)>, <(256, 1)>>, identity>, %arg12: !cannir.tensor<<gm, bf16>, <<(1024, 5120)>, <(5120, 1)>>, <<(1024, 5120)>, <(5120, 1)>>, identity>) attributes {cannbotdsl.contract_digest = "c4a7ab246cfde18bfbfb72841022a2d3c4b1545e1389658305936b8bc8035a13", cannbotdsl.public_slot_count = 13 : i64, cannbotdsl.slot_dtype_bits = [16, 32, 32, 8, 8, 8, 8, 8, 8, 16, 8, 8, 16], cannbotdsl.slot_dtype_codes = [4, 2, 2, 10, 1, 10, 1, 1, 1, 4, 1, 1, 4], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_type_digests = ["3843a377a73c2d97203078791020f489730ff5a96760690f1a35cca9fb54d1d2", "a937d472a7d9a5fa21f1628fbef3a3e41df1cdde47adc8944b8c4c208d0928ce", "a937d472a7d9a5fa21f1628fbef3a3e41df1cdde47adc8944b8c4c208d0928ce", "65b88b1eb93812f9dab6d543f5ec016eb613c0bd5636ac04a89e5f528cd1757e", "16cbddfff4ae34a1ce7f7c6254039c5b867d985ca07cd3ba59881dfc2110b044", "bdef267dfc9834767b03312323ac36c71542166cd9d91838915b833e62203191", "fb00e682a5ec86284319f174844f01626cabe3c9ebb69e7211f323c60bd16da8", "e670362ba2dcc8c0aadc33417d0263ca91f0d0bc3e67b83d3ffaca47a0f13c5b", "e5d3042bf0e3e6c250b18dc948ce4d2030330027c85e49995c52cb1a38544077", "b7f90534ce9234570e6a2d25f5cb83879c22d2263d4c227f23cdf0604e9f55ba", "38b28d3d206d2974c5d974e546fd8348c1269d94cc2c3d6086d605ca8c49a41e", "21aba7abb28c5fcdea9d0c4addc95d42f6759251ed6c219f59477cb1a1191dd2", "c6bf11a92c85a0501ee5d1a0c74b55bc724a080137f56fe0440f6e0771dbf84c"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %c32_i64 = arith.constant 32 : i64
    cannir.kernel_launch @kernel_AttnEpilogueTP1Kernel___call___0(%c32_i64) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, 8, 10, 10, 11, 11, 12>, name = "AttnEpilogueTP1Kernel___call__", op_type = "AttnEpilogueTP1Kernel___call__", roles = array<i64: 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1>}} : !cannir.tensor<<gm, bf16>, <<(8192, 4096)>, <(4096, 1)>>, <<(8192, 4096)>, <(4096, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1024, 64)>, <(64, 1)>>, <<(1024, 64)>, <(64, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1024, 64)>, <(64, 1)>>, <<(1024, 64)>, <(64, 1)>>, identity>, !cannir.tensor<<gm, f8E4M3FN>, <<(8192, 4096)>, <(4096, 1)>>, <<(8192, 4096)>, <(4096, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(8192, 64, 2)>, <(128, 2, 1)>>, <<(8192, 64, 2)>, <(128, 2, 1)>>, identity>, !cannir.tensor<<gm, f8E4M3FN>, <<(5120, 8192)>, <(8192, 1)>>, <<(5120, 8192)>, <(8192, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(5120, 128, 2)>, <(256, 2, 1)>>, <<(5120, 128, 2)>, <(256, 2, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(8192, 4096)>, <(4096, 1)>>, <<(8192, 4096)>, <(4096, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(8192, 128)>, <(128, 1)>>, <<(8192, 128)>, <(128, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(1024, 8192)>, <(8192, 1)>>, <<(1024, 8192)>, <(8192, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(1024, 8192)>, <(8192, 1)>>, <<(1024, 8192)>, <(8192, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(1024, 256)>, <(256, 1)>>, <<(1024, 256)>, <(256, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(1024, 5120)>, <(5120, 1)>>, <<(1024, 5120)>, <(5120, 1)>>, identity>
    return
  }
  func.func @kernel_AttnEpilogueTP1Kernel___call___0_aiv(%arg0: !cannir.tensor<<gm, bf16>, <<(8192, 4096)>, <(4096, 1)>>, <<(8192, 4096)>, <(4096, 1)>>, identity>, %arg1: !cannir.tensor<<gm, f32>, <<(1024, 64)>, <(64, 1)>>, <<(1024, 64)>, <(64, 1)>>, identity>, %arg2: !cannir.tensor<<gm, f32>, <<(1024, 64)>, <(64, 1)>>, <<(1024, 64)>, <(64, 1)>>, identity>, %arg3: !cannir.tensor<<gm, f8E4M3FN>, <<(8192, 4096)>, <(4096, 1)>>, <<(8192, 4096)>, <(4096, 1)>>, identity>, %arg4: !cannir.tensor<<gm, ui8>, <<(8192, 64, 2)>, <(128, 2, 1)>>, <<(8192, 64, 2)>, <(128, 2, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f8E4M3FN>, <<(5120, 8192)>, <(8192, 1)>>, <<(5120, 8192)>, <(8192, 1)>>, identity>, %arg6: !cannir.tensor<<gm, ui8>, <<(5120, 128, 2)>, <(256, 2, 1)>>, <<(5120, 128, 2)>, <(256, 2, 1)>>, identity>, %arg7: !cannir.tensor<<gm, ui8>, <<(8192, 4096)>, <(4096, 1)>>, <<(8192, 4096)>, <(4096, 1)>>, identity>, %arg8: !cannir.tensor<<gm, ui8>, <<(8192, 128)>, <(128, 1)>>, <<(8192, 128)>, <(128, 1)>>, identity>, %arg9: !cannir.tensor<<gm, bf16>, <<(1024, 8192)>, <(8192, 1)>>, <<(1024, 8192)>, <(8192, 1)>>, identity>, %arg10: !cannir.tensor<<gm, ui8>, <<(1024, 8192)>, <(8192, 1)>>, <<(1024, 8192)>, <(8192, 1)>>, identity>, %arg11: !cannir.tensor<<gm, ui8>, <<(1024, 256)>, <(256, 1)>>, <<(1024, 256)>, <(256, 1)>>, identity>, %arg12: !cannir.tensor<<gm, bf16>, <<(1024, 5120)>, <(5120, 1)>>, <<(1024, 5120)>, <(5120, 1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_AttnEpilogueTP1Kernel___call___0", cannir.resource.ids_prepared} {
    %c4096_i32 = arith.constant 4096 : i32
    %c32_i64 = arith.constant 32 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i64 = arith.constant 256 : i64
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c256_i32 = arith.constant 256 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %true = arith.constant true
    %c128_i64 = arith.constant 128 : i64
    %false = arith.constant false
    %c32_i32 = arith.constant 32 : i32
    %c8191_i64 = arith.constant 8191 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c8_i64 = arith.constant 8 : i64
    %c6_i64 = arith.constant 6 : i64
    %c5_i64 = arith.constant 5 : i64
    %c4_i64 = arith.constant 4 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<33554432xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<65536xf32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<65536xf32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<33554432xui8>
    %alloc_3 = memref.alloc() {cannir.arg_index = 8 : i64} : memref<1048576xui8>
    %alloc_4 = memref.alloc() {cannir.arg_index = 10 : i64} : memref<8388608xui8>
    %alloc_5 = memref.alloc() {cannir.arg_index = 11 : i64} : memref<262144xui8>
    %alloc_6 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xbf16, 1>
    %alloc_7 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<4096xbf16, 1>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<8192xui8, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 24576 : i64} : memref<128xui8, 1>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 24704 : i64} : memref<64xf32, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 24960 : i64} : memref<64xf32, 1>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 25216 : i64} : memref<512xf32, 1>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 27264 : i64} : memref<256xui16, 1>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 27776 : i64} : memref<2048xui8, 1>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 29824 : i64} : memref<2048xui8, 1>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 31872 : i64} : memref<16384xf32, 1>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 97408 : i64} : memref<16384xbf16, 1>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 130176 : i64} : memref<32768xui8, 1>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 162944 : i64} : memref<768xui8, 1>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 163712 : i64} : memref<2048xui8, 1>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 165760 : i64} : memref<1024xui16, 1>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 167808 : i64} : memref<8192xui8, 1>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 176000 : i64} : memref<8192xui8, 1>
    %0 = ascscalar.get_block_idx
    %1 = ascscalar.get_subblock_dim
    %2 = arith.muli %0, %1 : i64
    %3 = ascscalar.get_subblock_id
    %4 = arith.addi %2, %3 : i64
    %5 = arith.muli %1, %c32_i64 : i64
    %6 = arith.addi %5, %c8191_i64 : i64
    %7 = arith.divsi %6, %5 : i64
    scf.for %arg13 = %c0_i64 to %7 step %c1_i64  : i64 {
      %9 = arith.muli %arg13, %5 : i64
      %10 = arith.addi %4, %9 : i64
      %11 = arith.cmpi slt, %10, %c8192_i64 : i64
      scf.if %11 {
        %12 = arith.divsi %10, %c8_i64 : i64
        %13 = arith.remsi %10, %c8_i64 : i64
        %14 = ascscalar.bitcast_memref %alloc_6 byte_offset = %c0_i64 : memref<4096xbf16, 1> -> memref<?xbf16, 1>
        %15 = arith.muli %12, %c8_i64 : i64
        %16 = arith.addi %15, %13 : i64
        %17 = arith.muli %16, %c8192_i64 : i64
        ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%14, %alloc, %c0_i32, %c0_i32, %c0_i64, %17, %c1_i64, %c8192_i64, %c0_i64, %c8192_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 67108864 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 0 : i64, ub_capacity = 8192 : i64}> : (memref<?xbf16, 1>, memref<33554432xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
        %18 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<64xf32, 1> -> memref<?xf32, 1>
        %19 = arith.muli %12, %c256_i64 : i64
        "ascvec.copy_gm2ub_plan"(%18, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %19, %c1_i64, %c256_i64, %c0_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 262144 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 24704 : i64, ub_capacity = 256 : i64}> : (memref<?xf32, 1>, memref<65536xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        %20 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<64xf32, 1> -> memref<?xf32, 1>
        ascvec.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%20, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %19, %c1_i64, %c256_i64, %c0_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 262144 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 24960 : i64, ub_capacity = 256 : i64}> : (memref<?xf32, 1>, memref<65536xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_6, %alloc_12, %alloc_10, %alloc_11, %alloc_7 : memref<4096xbf16, 1>, memref<512xf32, 1>, memref<64xf32, 1>, memref<64xf32, 1>, memref<4096xbf16, 1>) {
        ^bb0(%arg14: memref<4096xbf16, 1>, %arg15: memref<512xf32, 1>, %arg16: memref<64xf32, 1>, %arg17: memref<64xf32, 1>, %arg18: memref<4096xbf16, 1>):
          %c7_i32 = arith.constant 7 : i32
          %c1_i32_24 = arith.constant 1 : i32
          %c8_i32 = arith.constant 8 : i32
          %c0_i32_25 = arith.constant 0 : i32
          %c64_i32 = arith.constant 64 : i32
          %c0_26 = arith.constant 0 : index
          %c448_i32 = arith.constant 448 : i32
          %c512_i32 = arith.constant 512 : i32
          %35 = ascvec.create_mask : !ascvec.mask
          scf.for %arg19 = %c0_i32_25 to %c8_i32 step %c1_i32_24  : i32 {
            %36 = arith.muli %arg19, %c512_i32 overflow<nsw> : i32
            %37 = arith.addi %36, %c448_i32 overflow<nsw> : i32
            %38 = arith.index_cast %37 : i32 to index
            %39 = ascvec.loadalign_unpack %arg14[%38] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
            %40 = arith.muli %arg19, %c64_i32 overflow<nsw> : i32
            %41 = ascvec.cast %39, %35 {rounding = 6 : i32} : <bf16> -> <f32>
            %42 = arith.index_cast %40 : i32 to index
            ascvec.storealign %arg15[%42], %41, %35 : memref<512xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
          %result0, %result1 = ascvec.loadalign_deintlv %arg16[%c0_26] {dintlv_width = "b32"} : memref<64xf32, 1>, index -> <f32>, <f32>
          %result0_27, %result1_28 = ascvec.loadalign_deintlv %arg17[%c0_26] {dintlv_width = "b32"} : memref<64xf32, 1>, index -> <f32>, <f32>
          scf.for %arg19 = %c0_i32_25 to %c8_i32 step %c1_i32_24  : i32 {
            %36 = arith.muli %arg19, %c512_i32 overflow<nsw> : i32
            scf.for %arg20 = %c0_i32_25 to %c7_i32 step %c1_i32_24  : i32 {
              %48 = arith.muli %arg20, %c64_i32 overflow<nsw> : i32
              %49 = arith.addi %36, %48 overflow<nsw> : i32
              %50 = arith.index_cast %49 : i32 to index
              %51 = ascvec.loadalign_unpack %arg14[%50] {mode = "b16_to_b32"} : memref<4096xbf16, 1>, index -> <bf16>
              ascvec.storealign_pack %arg18[%50], %51, %35 {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
            }
            %37 = arith.muli %arg19, %c64_i32 overflow<nsw> : i32
            %38 = arith.index_cast %37 : i32 to index
            %result0_29, %result1_30 = ascvec.loadalign_deintlv %arg15[%38] {dintlv_width = "b32"} : memref<512xf32, 1>, index -> <f32>, <f32>
            %39 = ascvec.mul %result0_29, %result0, %35 : <f32>
            %40 = ascvec.mul %result1_30, %result0_27, %35 : <f32>
            %41 = ascvec.add %39, %40, %35 : <f32>
            %42 = ascvec.mul %result1_30, %result0, %35 : <f32>
            %43 = ascvec.mul %result0_29, %result0_27, %35 : <f32>
            %44 = ascvec.sub %42, %43, %35 : <f32>
            %low, %high = ascvec.intlv %41, %44 : <f32>, <f32> -> <f32>, <f32>
            %45 = arith.addi %36, %c448_i32 overflow<nsw> : i32
            %46 = ascvec.cast %low, %35 : <f32> -> <bf16>
            %47 = arith.index_cast %45 : i32 to index
            ascvec.storealign_pack %arg18[%47], %46, %35 {mode = "b32_to_b16"} : memref<4096xbf16, 1>, index, <bf16>
          }
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_9, %alloc_7, %alloc_13, %alloc_14, %alloc_15, %alloc_8 : memref<128xui8, 1>, memref<4096xbf16, 1>, memref<256xui16, 1>, memref<2048xui8, 1>, memref<2048xui8, 1>, memref<8192xui8, 1>) {
        ^bb0(%arg14: memref<128xui8, 1>, %arg15: memref<4096xbf16, 1>, %arg16: memref<256xui16, 1>, %arg17: memref<2048xui8, 1>, %arg18: memref<2048xui8, 1>, %arg19: memref<8192xui8, 1>):
          %c2_i32 = arith.constant 2 : i32
          %c64_i32 = arith.constant 64 : i32
          %c8_i32 = arith.constant 8 : i32
          %c128_i32_24 = arith.constant 128 : i32
          %c1_i32_25 = arith.constant 1 : i32
          %c0_i32_26 = arith.constant 0 : i32
          %c16_i32 = arith.constant 16 : i32
          %c0_i16 = arith.constant 0 : i16
          %c256_i32_27 = arith.constant 256 : i32
          %c0_28 = arith.constant 0 : index
          %c1_i8 = arith.constant 1 : i8
          %c0_i8 = arith.constant 0 : i8
          %c262_i16 = arith.constant 262 : i16
          %c8_i16 = arith.constant 8 : i16
          %c32640_i16 = arith.constant 32640 : i16
          %35 = ascvec.create_mask elem_bits 16 : !ascvec.mask
          %36 = ascvec.create_mask : !ascvec.mask
          %37 = ascvec.create_mask elem_bits 8 pat_mode "PAT_VL16" : !ascvec.mask
          %38 = ascvec.create_mask elem_bits 16 pat_mode "PAT_VL16" : !ascvec.mask
          %39 = ascvec.duplicate_scalar %c32640_i16, %35 : i16 -> <ui16>
          %40 = ascvec.duplicate_scalar %c8_i16, %35 : i16 -> <ui16>
          %41 = ascvec.duplicate_scalar %c262_i16, %35 : i16 -> <ui16>
          %42 = ascvec.arange %c0_i8 : i8 -> <ui8>
          %43 = ascvec.duplicate_scalar %c1_i8, %37 : i8 -> <ui8>
          %44 = ascvec.and %42, %43, %37 : <ui8>, <ui8> -> <ui8>
          %45 = ascvec.cmp_eq_scalar %44, %c0_i8, %37 : <ui8>, i8 -> !ascvec.mask
          ascvec.store_unalign_begin %arg14 : memref<128xui8, 1>
          scf.for %arg20 = %c0_i32_26 to %c16_i32 step %c1_i32_25  : i32 {
            %46 = arith.muli %arg20, %c256_i32_27 overflow<nsw> : i32
            %47 = arith.index_cast %46 : i32 to index
            %result0, %result1 = ascvec.loadalign_deintlv %arg15[%47] {dintlv_width = "b16"} : memref<4096xbf16, 1>, index -> <bf16>, <bf16>
            %48 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
            %49 = ascvec.and %48, %39, %35 : <ui16>, <ui16> -> <ui16>
            %50 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
            %51 = ascvec.and %50, %39, %35 : <ui16>, <ui16> -> <ui16>
            %52 = ascvec.max %49, %51, %35 : <ui16>
            %53 = ascvec.reduce_max_datablock %52, %35 : <ui16>
            %54 = ascvec.shr %53, 7, %35 : <ui16> -> <ui16>
            %55 = ascvec.max_scalar %54, %c8_i16, %35 : <ui16>, i16
            %56 = ascvec.sub %55, %40, %35 : <ui16>
            %57 = ascvec.cast %56, %35 {reg_layout = 0 : i32} : <ui16> -> <ui8>
            %58 = ascvec.squeeze_store_reg %57, %45 : <ui8>, !ascvec.mask -> <ui8>
            ascvec.store_unalign %arg14[%c0_28], %58 : memref<128xui8, 1>, <ui8>
            %59 = ascvec.sub %41, %55, %35 : <ui16>
            %60 = ascvec.shl %59, 7, %35 : <ui16> -> <ui16>
            %61 = ascvec.duplicate_scalar %c0_i16, %35 : i16 -> <ui16>
            %62 = ascvec.cmp_eq_scalar %55, %c8_i16, %35 : <ui16>, i16 -> !ascvec.mask
            %63 = ascvec.select %61, %60, %62 : <ui16>
            %low, %high = ascvec.intlv %63, %63 : <ui16>, <ui16> -> <ui16>, <ui16>
            %64 = arith.muli %arg20, %c16_i32 overflow<nsw> : i32
            %65 = arith.index_cast %64 : i32 to index
            ascvec.storealign %arg16[%65], %low, %38 : memref<256xui16, 1>, index, <ui16>
          }
          %c0_29 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg20 = %c0_29 to %c1 step %c1 {
            ascvec.store_unalign_post %arg14[%c0_28] : memref<128xui8, 1>
          }
          ascvec.mem_bar vst_vld
          scf.for %arg20 = %c0_i32_26 to %c16_i32 step %c1_i32_25  : i32 {
            scf.for %arg21 = %c0_i32_26 to %c2_i32 step %c1_i32_25  : i32 {
              %46 = arith.muli %arg20, %c256_i32_27 overflow<nsw> : i32
              %47 = arith.muli %arg21, %c128_i32_24 overflow<nsw> : i32
              %48 = arith.addi %46, %47 overflow<nsw> : i32
              %49 = arith.index_cast %48 : i32 to index
              %50 = ascvec.loadalign %arg15[%49] : memref<4096xbf16, 1>, index -> <bf16>
              %51 = arith.muli %arg20, %c16_i32 overflow<nsw> : i32
              %52 = arith.muli %arg21, %c8_i32 overflow<nsw> : i32
              %53 = arith.addi %51, %52 overflow<nsw> : i32
              %54 = arith.index_cast %53 : i32 to index
              %55 = ascvec.loadalign_brc %arg16[%54] {brc_width = "e2b_b16"} : memref<256xui16, 1>, index -> <ui16>
              %56 = ascvec.reinterpret %55 : <ui16> -> <bf16>
              %57 = ascvec.mul %50, %56, %35 : <bf16>
              %58 = ascvec.cast %57, %35 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
              %59 = ascvec.cast %57, %35 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
              %60 = ascvec.cast %58, %36 {reg_layout = 0 : i32, saturate = true} : <f32> -> <f8E4M3FN>
              %61 = ascvec.cast %59, %36 {reg_layout = 0 : i32, saturate = true} : <f32> -> <f8E4M3FN>
              %62 = arith.muli %arg20, %c128_i32_24 overflow<nsw> : i32
              %63 = arith.muli %arg21, %c64_i32 overflow<nsw> : i32
              %64 = arith.addi %62, %63 overflow<nsw> : i32
              %65 = arith.index_cast %64 : i32 to index
              ascvec.storealign_pack %arg17[%65], %60, %36 {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <f8E4M3FN>
              ascvec.storealign_pack %arg18[%65], %61, %36 {mode = "b32_to_b8"} : memref<2048xui8, 1>, index, <f8E4M3FN>
            }
          }
          ascvec.mem_bar vst_vld
          scf.for %arg20 = %c0_i32_26 to %c16_i32 step %c1_i32_25  : i32 {
            scf.for %arg21 = %c0_i32_26 to %c2_i32 step %c1_i32_25  : i32 {
              %46 = arith.muli %arg20, %c256_i32_27 overflow<nsw> : i32
              %47 = arith.muli %arg21, %c128_i32_24 overflow<nsw> : i32
              %48 = arith.addi %46, %47 overflow<nsw> : i32
              %49 = arith.muli %arg20, %c128_i32_24 overflow<nsw> : i32
              %50 = arith.muli %arg21, %c64_i32 overflow<nsw> : i32
              %51 = arith.addi %49, %50 overflow<nsw> : i32
              %52 = arith.index_cast %51 : i32 to index
              %53 = ascvec.loadalign %arg17[%52] : memref<2048xui8, 1>, index -> <ui8>
              %54 = ascvec.loadalign %arg18[%52] : memref<2048xui8, 1>, index -> <ui8>
              %55 = arith.index_cast %48 : i32 to index
              ascvec.storealign_intlv %arg19[%55], %53, %54 {intlv_width = "b8"} : memref<8192xui8, 1>, <ui8>, <ui8>
            }
          }
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %c2_i64, 0 {cannir.buffer.generated_sync}
        %21 = arith.muli %13, %c1024_i64 overflow<nsw> : i64
        %22 = arith.addi %21, %12 : i64
        %23 = arith.muli %22, %c32768_i64 : i64
        %24 = ascscalar.memref_address %alloc_2 : memref<33554432xui8>
        %25 = arith.divui %23, %c8_i64 : i64
        %26 = arith.addi %24, %25 : i64
        %27 = ascscalar.bind_gm_ptr %26 : memref<1xui8>
        %28 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<8192xui8, 1> -> memref<?xui8, 1>
        ascvec.get_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
        ascvec.copy_ub2gm %27[%c0], %28[%c0], %c1_i32, %c4096_i32, %c0_i64, %c4096_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
        ascvec.release_buf mte3, %c1_i64, 0 {cannir.buffer.generated_sync}
        %29 = arith.muli %22, %c1024_i64 : i64
        %30 = ascscalar.memref_address %alloc_3 : memref<1048576xui8>
        %31 = arith.divui %29, %c8_i64 : i64
        %32 = arith.addi %30, %31 : i64
        %33 = ascscalar.bind_gm_ptr %32 : memref<1xui8>
        %34 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<128xui8, 1> -> memref<?xui8, 1>
        ascvec.get_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
        ascvec.copy_ub2gm %33[%c0], %34[%c0], %c1_i32, %c128_i32, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
        ascvec.release_buf mte3, %c2_i64, 0 {cannir.buffer.generated_sync}
      }
    }
    ascvec.sync_all
    ascvec.sync_block_arrive mte3, %c0_i64
    ascvec.sync_block_arrive mte3, %c1_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c1_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c2_i64
    ascvec.sync_intra_arrive v, %c0_i64
    %8:2 = scf.for %arg13 = %0 to %c256_i64 step %c32_i64 iter_args(%arg14 = %c1_i64, %arg15 = %c0_i64) -> (i64, i64)  : i64 {
      %9 = arith.divsi %arg13, %c32_i64 : i64
      %10 = arith.divsi %arg13, %c4_i64 : i64
      %11 = arith.remsi %10, %c8_i64 : i64
      %12 = arith.cmpi ne, %11, %c0_i64 : i64
      %13 = arith.cmpi slt, %11, %c0_i64 : i64
      %14 = arith.andi %12, %13 : i1
      %15 = arith.select %14, %c8_i64, %c0_i64 : i64
      %16 = arith.addi %11, %15 overflow<nsw> : i64
      %17 = arith.remsi %arg13, %c4_i64 : i64
      %18 = arith.muli %9, %c1024_i64 : i64
      %19 = arith.cmpi slt, %arg13, %c32_i64 : i64
      %20 = arith.muli %16, %c128_i64 overflow<nsw> : i64
      %21 = arith.muli %17, %c256_i64 overflow<nsw> : i64
      %22 = arith.cmpi eq, %19, %false : i1
      %23 = arith.addi %arg15, %c1_i64 overflow<nsw> : i64
      %24 = arith.remui %23, %c2_i64 : i64
      %25:2 = scf.for %arg16 = %c0_i64 to %c8_i64 step %c1_i64 iter_args(%arg17 = %c0_i64, %arg18 = %arg14) -> (i64, i64)  : i64 {
        %72 = arith.addi %arg17, %c1_i64 overflow<nsw> : i64
        %73 = arith.remui %72, %c2_i64 : i64
        %74 = arith.cmpi sge, %arg16, %c1_i64 : i64
        %75 = arith.select %74, %true, %22 : i1
        %76 = scf.if %75 -> (i64) {
          %77 = arith.addi %arg18, %c1_i64 : i64
          %78 = arith.remui %77, %c2_i64 : i64
          scf.yield %78 : i64
        } else {
          scf.yield %arg18 : i64
        }
        scf.yield %73, %76 : i64, i64
      }
      %26 = arith.subi %c1024_i64, %20 overflow<nsw> : i64
      %27 = arith.minsi %26, %c128_i64 : i64
      %28 = arith.subi %c1024_i64, %21 overflow<nsw> : i64
      %29 = arith.minsi %28, %c256_i64 : i64
      %30 = arith.addi %18, %21 : i64
      %31 = arith.remsi %27, %c2_i64 : i64
      %32 = arith.cmpi ne, %31, %c0_i64 : i64
      %33 = arith.cmpi slt, %31, %c0_i64 : i64
      %34 = arith.andi %32, %33 : i1
      %35 = arith.divsi %27, %c2_i64 : i64
      %36 = arith.extui %34 : i1 to i64
      %37 = arith.subi %35, %36 : i64
      ascvec.sync_intra_wait v, %c0_i64 {cannir.sync.banks = 3 : i32}
      ascvec.vf(%alloc_16, %alloc_17 : memref<16384xf32, 1>, memref<16384xbf16, 1>) {
      ^bb0(%arg16: memref<16384xf32, 1>, %arg17: memref<16384xbf16, 1>):
        %c64_i32 = arith.constant 64 : i32
        %c16384_i32 = arith.constant 16384 : i32
        %c0_i32_24 = arith.constant 0 : i32
        %72 = ascvec.create_mask : !ascvec.mask
        scf.for %arg18 = %c0_i32_24 to %c16384_i32 step %c64_i32  : i32 {
          %73 = arith.index_cast %arg18 : i32 to index
          %74 = ascvec.loadalign %arg16[%73] : memref<16384xf32, 1>, index -> <f32>
          %75 = ascvec.cast %74, %72 : <f32> -> <bf16>
          ascvec.storealign_pack %arg17[%73], %75, %72 {mode = "b32_to_b16"} : memref<16384xbf16, 1>, index, <bf16>
        }
      } {cannir.vf_mode = "simd"}
      ascvec.sync_intra_arrive v, %c0_i64 {cannir.sync.banks = 3 : i32}
      ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%alloc_19, %alloc_17, %alloc_21, %alloc_22, %alloc_23, %alloc_18 : memref<768xui8, 1>, memref<16384xbf16, 1>, memref<1024xui16, 1>, memref<8192xui8, 1>, memref<8192xui8, 1>, memref<32768xui8, 1>) {
      ^bb0(%arg16: memref<768xui8, 1>, %arg17: memref<16384xbf16, 1>, %arg18: memref<1024xui16, 1>, %arg19: memref<8192xui8, 1>, %arg20: memref<8192xui8, 1>, %arg21: memref<32768xui8, 1>):
        %c2_i32 = arith.constant 2 : i32
        %c8_i32 = arith.constant 8 : i32
        %c128_i32_24 = arith.constant 128 : i32
        %c1_i32_25 = arith.constant 1 : i32
        %c64_i32 = arith.constant 64 : i32
        %c0_i32_26 = arith.constant 0 : i32
        %c16_i32 = arith.constant 16 : i32
        %c0_i16 = arith.constant 0 : i16
        %c256_i32_27 = arith.constant 256 : i32
        %c0_28 = arith.constant 0 : index
        %c1_i8 = arith.constant 1 : i8
        %c0_i8 = arith.constant 0 : i8
        %c262_i16 = arith.constant 262 : i16
        %c8_i16 = arith.constant 8 : i16
        %c32640_i16 = arith.constant 32640 : i16
        %72 = ascvec.create_mask elem_bits 16 : !ascvec.mask
        %73 = ascvec.create_mask : !ascvec.mask
        %74 = ascvec.create_mask elem_bits 8 pat_mode "PAT_VL16" : !ascvec.mask
        %75 = ascvec.create_mask elem_bits 16 pat_mode "PAT_VL16" : !ascvec.mask
        %76 = ascvec.duplicate_scalar %c32640_i16, %72 : i16 -> <ui16>
        %77 = ascvec.duplicate_scalar %c8_i16, %72 : i16 -> <ui16>
        %78 = ascvec.duplicate_scalar %c262_i16, %72 : i16 -> <ui16>
        %79 = ascvec.arange %c0_i8 : i8 -> <ui8>
        %80 = ascvec.duplicate_scalar %c1_i8, %74 : i8 -> <ui8>
        %81 = ascvec.and %79, %80, %74 : <ui8>, <ui8> -> <ui8>
        %82 = ascvec.cmp_eq_scalar %81, %c0_i8, %74 : <ui8>, i8 -> !ascvec.mask
        ascvec.store_unalign_begin %arg16 : memref<768xui8, 1>
        scf.for %arg22 = %c0_i32_26 to %c64_i32 step %c1_i32_25  : i32 {
          %83 = arith.muli %arg22, %c256_i32_27 overflow<nsw> : i32
          %84 = arith.index_cast %83 : i32 to index
          %result0, %result1 = ascvec.loadalign_deintlv %arg17[%84] {dintlv_width = "b16"} : memref<16384xbf16, 1>, index -> <bf16>, <bf16>
          %85 = ascvec.reinterpret %result0 : <bf16> -> <ui16>
          %86 = ascvec.and %85, %76, %72 : <ui16>, <ui16> -> <ui16>
          %87 = ascvec.reinterpret %result1 : <bf16> -> <ui16>
          %88 = ascvec.and %87, %76, %72 : <ui16>, <ui16> -> <ui16>
          %89 = ascvec.max %86, %88, %72 : <ui16>
          %90 = ascvec.reduce_max_datablock %89, %72 : <ui16>
          %91 = ascvec.shr %90, 7, %72 : <ui16> -> <ui16>
          %92 = ascvec.max_scalar %91, %c8_i16, %72 : <ui16>, i16
          %93 = ascvec.sub %92, %77, %72 : <ui16>
          %94 = ascvec.cast %93, %72 {reg_layout = 0 : i32} : <ui16> -> <ui8>
          %95 = ascvec.squeeze_store_reg %94, %82 : <ui8>, !ascvec.mask -> <ui8>
          ascvec.store_unalign %arg16[%c0_28], %95 : memref<768xui8, 1>, <ui8>
          %96 = ascvec.sub %78, %92, %72 : <ui16>
          %97 = ascvec.shl %96, 7, %72 : <ui16> -> <ui16>
          %98 = ascvec.duplicate_scalar %c0_i16, %72 : i16 -> <ui16>
          %99 = ascvec.cmp_eq_scalar %92, %c8_i16, %72 : <ui16>, i16 -> !ascvec.mask
          %100 = ascvec.select %98, %97, %99 : <ui16>
          %low, %high = ascvec.intlv %100, %100 : <ui16>, <ui16> -> <ui16>, <ui16>
          %101 = arith.muli %arg22, %c16_i32 overflow<nsw> : i32
          %102 = arith.index_cast %101 : i32 to index
          ascvec.storealign %arg18[%102], %low, %75 : memref<1024xui16, 1>, index, <ui16>
        }
        %c0_29 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg22 = %c0_29 to %c1 step %c1 {
          ascvec.store_unalign_post %arg16[%c0_28] : memref<768xui8, 1>
        }
        ascvec.mem_bar vst_vld
        scf.for %arg22 = %c0_i32_26 to %c64_i32 step %c1_i32_25  : i32 {
          scf.for %arg23 = %c0_i32_26 to %c2_i32 step %c1_i32_25  : i32 {
            %83 = arith.muli %arg22, %c256_i32_27 overflow<nsw> : i32
            %84 = arith.muli %arg23, %c128_i32_24 overflow<nsw> : i32
            %85 = arith.addi %83, %84 overflow<nsw> : i32
            %86 = arith.index_cast %85 : i32 to index
            %87 = ascvec.loadalign %arg17[%86] : memref<16384xbf16, 1>, index -> <bf16>
            %88 = arith.muli %arg22, %c16_i32 overflow<nsw> : i32
            %89 = arith.muli %arg23, %c8_i32 overflow<nsw> : i32
            %90 = arith.addi %88, %89 overflow<nsw> : i32
            %91 = arith.index_cast %90 : i32 to index
            %92 = ascvec.loadalign_brc %arg18[%91] {brc_width = "e2b_b16"} : memref<1024xui16, 1>, index -> <ui16>
            %93 = ascvec.reinterpret %92 : <ui16> -> <bf16>
            %94 = ascvec.mul %87, %93, %72 : <bf16>
            %95 = ascvec.cast %94, %72 {reg_layout = 0 : i32, rounding = 6 : i32} : <bf16> -> <f32>
            %96 = ascvec.cast %94, %72 {reg_layout = 1 : i32, rounding = 6 : i32} : <bf16> -> <f32>
            %97 = ascvec.cast %95, %73 {reg_layout = 0 : i32, saturate = true} : <f32> -> <f8E4M3FN>
            %98 = ascvec.cast %96, %73 {reg_layout = 0 : i32, saturate = true} : <f32> -> <f8E4M3FN>
            %99 = arith.muli %arg22, %c128_i32_24 overflow<nsw> : i32
            %100 = arith.muli %arg23, %c64_i32 overflow<nsw> : i32
            %101 = arith.addi %99, %100 overflow<nsw> : i32
            %102 = arith.index_cast %101 : i32 to index
            ascvec.storealign_pack %arg19[%102], %97, %73 {mode = "b32_to_b8"} : memref<8192xui8, 1>, index, <f8E4M3FN>
            ascvec.storealign_pack %arg20[%102], %98, %73 {mode = "b32_to_b8"} : memref<8192xui8, 1>, index, <f8E4M3FN>
          }
        }
        ascvec.mem_bar vst_vld
        scf.for %arg22 = %c0_i32_26 to %c64_i32 step %c1_i32_25  : i32 {
          scf.for %arg23 = %c0_i32_26 to %c2_i32 step %c1_i32_25  : i32 {
            %83 = arith.muli %arg22, %c256_i32_27 overflow<nsw> : i32
            %84 = arith.muli %arg23, %c128_i32_24 overflow<nsw> : i32
            %85 = arith.addi %83, %84 overflow<nsw> : i32
            %86 = arith.muli %arg22, %c128_i32_24 overflow<nsw> : i32
            %87 = arith.muli %arg23, %c64_i32 overflow<nsw> : i32
            %88 = arith.addi %86, %87 overflow<nsw> : i32
            %89 = arith.index_cast %88 : i32 to index
            %90 = ascvec.loadalign %arg19[%89] : memref<8192xui8, 1>, index -> <ui8>
            %91 = ascvec.loadalign %arg20[%89] : memref<8192xui8, 1>, index -> <ui8>
            %92 = arith.index_cast %85 : i32 to index
            ascvec.storealign_intlv %arg21[%92], %90, %91 {intlv_width = "b8"} : memref<32768xui8, 1>, <ui8>, <ui8>
          }
        }
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%alloc_19, %alloc_20 : memref<768xui8, 1>, memref<2048xui8, 1>) {
      ^bb0(%arg16: memref<768xui8, 1>, %arg17: memref<2048xui8, 1>):
        %c1_i32_24 = arith.constant 1 : i32
        %c64_i32 = arith.constant 64 : i32
        %c0_i32_25 = arith.constant 0 : i32
        %c32_i32_26 = arith.constant 32 : i32
        %c8_i32 = arith.constant 8 : i32
        %mask, %remaining_out = ascvec.update_mask %c8_i32 elem_bits 8 : !ascvec.mask
        scf.for %arg18 = %c0_i32_25 to %c64_i32 step %c1_i32_24  : i32 {
          %72 = arith.muli %arg18, %c8_i32 overflow<nsw> : i32
          %73 = arith.index_cast %72 : i32 to index
          ascvec.loadunalign_init %arg16[%73] : memref<768xui8, 1>
          %74 = ascvec.loadunalign %arg16[%73] : memref<768xui8, 1> -> <ui8>
          %75 = arith.muli %arg18, %c32_i32_26 overflow<nsw> : i32
          %76 = arith.index_cast %75 : i32 to index
          ascvec.storealign %arg17[%76], %74, %mask : memref<2048xui8, 1>, index, <ui8>
        }
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
      %38 = arith.muli %3, %37 : i64
      %39 = arith.addi %20, %38 : i64
      %40 = arith.muli %39, %c8192_i64 : i64
      %41 = arith.addi %40, %30 : i64
      %42 = arith.muli %41, %c8_i64 : i64
      %43 = ascscalar.memref_address %alloc_4 : memref<8388608xui8>
      %44 = arith.divui %42, %c8_i64 : i64
      %45 = arith.addi %43, %44 : i64
      %46 = ascscalar.bind_gm_ptr %45 : memref<1xui8>
      %47 = ascscalar.bitcast_memref %alloc_18 byte_offset = %c0_i64 : memref<32768xui8, 1> -> memref<?xui8, 1>
      ascvec.get_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
      %48 = arith.trunci %37 : i64 to i32
      ascvec.copy_ub2gm %46[%c0], %47[%c0], %48, %c256_i32, %c8192_i64, %c256_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
      ascvec.release_buf mte3, %c5_i64, 0 {cannir.buffer.generated_sync}
      %49 = arith.remsi %30, %c32_i64 : i64
      %50 = arith.cmpi ne, %49, %c0_i64 : i64
      %51 = arith.cmpi slt, %49, %c0_i64 : i64
      %52 = arith.andi %50, %51 : i1
      %53 = arith.divsi %30, %c32_i64 : i64
      %54 = arith.extui %52 : i1 to i64
      %55 = arith.subi %53, %54 : i64
      %56 = arith.remsi %29, %c32_i64 : i64
      %57 = arith.cmpi ne, %56, %c0_i64 : i64
      %58 = arith.cmpi slt, %56, %c0_i64 : i64
      %59 = arith.andi %57, %58 : i1
      %60 = arith.divsi %29, %c32_i64 : i64
      %61 = arith.extui %59 : i1 to i64
      %62 = arith.subi %60, %61 : i64
      %63 = arith.muli %39, %c256_i64 : i64
      %64 = arith.addi %63, %55 : i64
      %65 = arith.muli %64, %c8_i64 : i64
      %66 = ascscalar.memref_address %alloc_5 : memref<262144xui8>
      %67 = arith.divui %65, %c8_i64 : i64
      %68 = arith.addi %66, %67 : i64
      %69 = ascscalar.bind_gm_ptr %68 : memref<1xui8>
      %70 = ascscalar.bitcast_memref %alloc_20 byte_offset = %c0_i64 : memref<2048xui8, 1> -> memref<?xui8, 1>
      ascvec.get_buf mte3, %c6_i64, 0 {cannir.buffer.generated_sync}
      %71 = arith.trunci %62 : i64 to i32
      ascvec.copy_ub2gm %69[%c0], %70[%c0], %48, %71, %c256_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<?xui8, 1>
      ascvec.release_buf mte3, %c6_i64, 0 {cannir.buffer.generated_sync}
      scf.yield %25#1, %24 : i64, i64
    }
    ascvec.sync_all
    ascvec.sync_block_arrive mte3, %c0_i64
    ascvec.sync_block_arrive mte3, %c1_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c1_i64 {mode = 0 : i32}
    ascvec.sync_block_wait s, %c2_i64
    return
  }
  func.func @kernel_AttnEpilogueTP1Kernel___call___0_aic(%arg0: !cannir.tensor<<gm, bf16>, <<(8192, 4096)>, <(4096, 1)>>, <<(8192, 4096)>, <(4096, 1)>>, identity>, %arg1: !cannir.tensor<<gm, f32>, <<(1024, 64)>, <(64, 1)>>, <<(1024, 64)>, <(64, 1)>>, identity>, %arg2: !cannir.tensor<<gm, f32>, <<(1024, 64)>, <(64, 1)>>, <<(1024, 64)>, <(64, 1)>>, identity>, %arg3: !cannir.tensor<<gm, f8E4M3FN>, <<(8192, 4096)>, <(4096, 1)>>, <<(8192, 4096)>, <(4096, 1)>>, identity>, %arg4: !cannir.tensor<<gm, ui8>, <<(8192, 64, 2)>, <(128, 2, 1)>>, <<(8192, 64, 2)>, <(128, 2, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f8E4M3FN>, <<(5120, 8192)>, <(8192, 1)>>, <<(5120, 8192)>, <(8192, 1)>>, identity>, %arg6: !cannir.tensor<<gm, ui8>, <<(5120, 128, 2)>, <(256, 2, 1)>>, <<(5120, 128, 2)>, <(256, 2, 1)>>, identity>, %arg7: !cannir.tensor<<gm, ui8>, <<(8192, 4096)>, <(4096, 1)>>, <<(8192, 4096)>, <(4096, 1)>>, identity>, %arg8: !cannir.tensor<<gm, ui8>, <<(8192, 128)>, <(128, 1)>>, <<(8192, 128)>, <(128, 1)>>, identity>, %arg9: !cannir.tensor<<gm, bf16>, <<(1024, 8192)>, <(8192, 1)>>, <<(1024, 8192)>, <(8192, 1)>>, identity>, %arg10: !cannir.tensor<<gm, ui8>, <<(1024, 8192)>, <(8192, 1)>>, <<(1024, 8192)>, <(8192, 1)>>, identity>, %arg11: !cannir.tensor<<gm, ui8>, <<(1024, 256)>, <(256, 1)>>, <<(1024, 256)>, <(256, 1)>>, identity>, %arg12: !cannir.tensor<<gm, bf16>, <<(1024, 5120)>, <(5120, 1)>>, <<(1024, 5120)>, <(5120, 1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_AttnEpilogueTP1Kernel___call___0", cannir.resource.ids_prepared} {
    %c5120_i32 = arith.constant 5120 : i32
    %c4_i32 = arith.constant 4 : i32
    %c32_i64 = arith.constant 32 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c0_i32 = arith.constant 0 : i32
    %c256_i64 = arith.constant 256 : i64
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %c512_i32 = arith.constant 512 : i32
    %c64_i32 = arith.constant 64 : i32
    %c0 = arith.constant 0 : index
    %c16_i32 = arith.constant 16 : i32
    %c256_i32 = arith.constant 256 : i32
    %c2_i32 = arith.constant 2 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1024_i64 = arith.constant 1024 : i64
    %true = arith.constant true
    %c512_i64 = arith.constant 512 : i64
    %c128_i64 = arith.constant 128 : i64
    %false = arith.constant false
    %c3_i32 = arith.constant 3 : i32
    %c5120_i64 = arith.constant 5120 : i64
    %c160_i64 = arith.constant 160 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c2 = arith.constant 2 : index
    %c32768_i64 = arith.constant 32768 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c20_i64 = arith.constant 20 : i64
    %c16_i64 = arith.constant 16 : i64
    %c15_i64 = arith.constant 15 : i64
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
    %alloc = memref.alloc() {cannir.arg_index = 3 : i64} : memref<33554432xf8E4M3FN>
    %alloc_0 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<1048576xui8>
    %alloc_1 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<41943040xf8E4M3FN>
    %alloc_2 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1310720xui8>
    %alloc_3 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<33554432xui8>
    %alloc_4 = memref.alloc() {cannir.arg_index = 8 : i64} : memref<1048576xui8>
    %alloc_5 = memref.alloc() {cannir.arg_index = 10 : i64} : memref<8388608xui8>
    %alloc_6 = memref.alloc() {cannir.arg_index = 11 : i64} : memref<262144xui8>
    %alloc_7 = memref.alloc() {cannir.arg_index = 12 : i64} : memref<5242880xbf16>
    %alloc_8 = memref.alloc() {cannir.byte_offset = 31872 : i64} : memref<16384xf32, 1>
    %alloc_9 = memref.alloc() {cannir.byte_offset = 425984 : i64} : memref<65536x!ascdtype.fp8_e8m0, 2>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 393216 : i64} : memref<32768x!ascdtype.fp8_e8m0, 2>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<32768xf8E4M3FN, 4>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<32768xf8E4M3FN, 4>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<16384xf8E4M3FN, 3>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xf8E4M3FN, 3>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 262144 : i64} : memref<131072xf8E4M3FN, 2>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 131072 : i64} : memref<131072xf8E4M3FN, 2>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<65536xf8E4M3FN, 2>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<65536xf8E4M3FN, 2>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 131072 : i64} : memref<32768xf32, 5>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<32768xf32, 5>
    %0 = ascscalar.get_block_idx
    %1 = arith.remsi %0, %c4_i64 : i64
    %2 = arith.muli %1, %c256_i64 overflow<nsw> : i64
    %3 = arith.addi %1, %c1_i64 overflow<nsw> : i64
    %4 = arith.muli %3, %c256_i64 overflow<nsw> : i64
    %5 = arith.minsi %4, %c1024_i64 : i64
    %6 = arith.subi %5, %2 overflow<nsw> : i64
    %7 = ascscalar.bitcast_memref %alloc_16 byte_offset = %c0_i64 : memref<131072xf8E4M3FN, 2> -> memref<?xf8E4M3FN, 2>
    %8 = arith.divui %2, %c8_i64 : i64
    %9 = arith.muli %8, %c32768_i64 : i64
    asccube.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
    %10 = arith.cmpi ugt, %6, %c0_i64 : i64
    scf.if %10 {
      %33 = arith.index_cast %9 : i64 to index
      %34 = arith.trunci %6 : i64 to i32
      asccube.copy_gm2l1 %7[%c0], %alloc[%33], %34, %c512_i32, %c4096_i64, %c1_i32, %c0_i64, %c0_i64, %c256_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<?xf8E4M3FN, 2>, memref<33554432xf8E4M3FN>
    }
    asccube.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
    %11 = arith.muli %1, %c32768_i64 overflow<nsw> : i64
    %12 = ascscalar.bitcast_memref %alloc_9 byte_offset = %c0_i64 : memref<65536x!ascdtype.fp8_e8m0, 2> -> memref<?x!ascdtype.fp8_e8m0, 2>
    scf.if %10 {
      %33 = arith.index_cast %11 : i64 to index
      %34 = arith.trunci %6 : i64 to i32
      asccube.copy_gm2l1 %12[%c0], %alloc_0[%33], %c64_i32, %34, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {dn2nz = true, mx_scale = true} : memref<?x!ascdtype.fp8_e8m0, 2>, memref<1048576xui8>
    }
    asccube.sync_all
    asccube.sync_block_wait s, %c0_i64
    asccube.sync_block_arrive fixpipe, %c1_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c1_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c2_i64
    %13:2 = scf.for %arg13 = %0 to %c256_i64 step %c32_i64 iter_args(%arg14 = %c1_i64, %arg15 = %c0_i64) -> (i64, i64)  : i64 {
      %33 = arith.divsi %arg13, %c32_i64 : i64
      %34 = arith.divsi %arg13, %c4_i64 : i64
      %35 = arith.remsi %34, %c8_i64 : i64
      %36 = arith.cmpi ne, %35, %c0_i64 : i64
      %37 = arith.cmpi slt, %35, %c0_i64 : i64
      %38 = arith.andi %36, %37 : i1
      %39 = arith.select %38, %c8_i64, %c0_i64 : i64
      %40 = arith.addi %35, %39 overflow<nsw> : i64
      %41 = arith.remsi %arg13, %c4_i64 : i64
      %42 = arith.muli %33, %c1024_i64 : i64
      %43 = arith.addi %33, %c1_i64 : i64
      %44 = arith.muli %43, %c1024_i64 : i64
      %45 = arith.subi %44, %42 : i64
      %46 = arith.cmpi slt, %arg13, %c32_i64 : i64
      %47 = arith.muli %40, %c128_i64 overflow<nsw> : i64
      %48 = arith.muli %41, %c256_i64 overflow<nsw> : i64
      %49 = arith.addi %47, %c128_i64 overflow<nsw> : i64
      %50 = arith.minsi %49, %45 : i64
      %51 = arith.addi %48, %c256_i64 overflow<nsw> : i64
      %52 = arith.minsi %51, %45 : i64
      %53 = arith.subi %50, %47 : i64
      %54 = arith.muli %40, %c16384_i64 overflow<nsw> : i64
      %55 = arith.muli %33, %c131072_i64 : i64
      %56 = arith.addi %54, %55 : i64
      %57 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<32768x!ascdtype.fp8_e8m0, 2> -> memref<?x!ascdtype.fp8_e8m0, 2>
      asccube.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
      %58 = arith.cmpi ugt, %53, %c0_i64 : i64
      scf.if %58 {
        %69 = arith.index_cast %56 : i64 to index
        %70 = arith.trunci %53 : i64 to i32
        asccube.copy_gm2l1 %57[%c0], %alloc_4[%69], %c64_i32, %70, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {dn2nz = true, mx_scale = true} : memref<?x!ascdtype.fp8_e8m0, 2>, memref<1048576xui8>
      }
      asccube.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
      %59 = arith.cmpi eq, %46, %false : i1
      scf.if %59 {
        %69 = arith.subi %52, %48 : i64
        %70 = arith.muli %41, %c32768_i64 overflow<nsw> : i64
        %71 = arith.addi %70, %55 : i64
        asccube.get_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
        %72 = arith.cmpi ugt, %69, %c0_i64 : i64
        scf.if %72 {
          %73 = arith.index_cast %71 : i64 to index
          %74 = arith.trunci %69 : i64 to i32
          asccube.copy_gm2l1 %12[%c0], %alloc_0[%73], %c64_i32, %74, %c128_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {dn2nz = true, mx_scale = true} : memref<?x!ascdtype.fp8_e8m0, 2>, memref<1048576xui8>
        }
        asccube.release_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
      }
      %60 = arith.cmpi eq, %arg15, %c0_i64 : i64
      %61 = ascscalar.memref_address %alloc_20 : memref<32768xf32, 5>
      %62 = ascscalar.memref_address %alloc_19 : memref<32768xf32, 5>
      %63 = arith.select %60, %61, %62 : i64
      %64 = arith.addi %arg15, %c1_i64 overflow<nsw> : i64
      %65 = arith.remui %64, %c2_i64 : i64
      %66:2 = scf.for %arg16 = %c0_i64 to %c8_i64 step %c1_i64 iter_args(%arg17 = %c0_i64, %arg18 = %arg14) -> (i64, i64)  : i64 {
        %69 = arith.muli %arg16, %c512_i64 overflow<nsw> : i64
        %70 = arith.cmpi eq, %arg17, %c0_i64 : i64
        %71 = ascscalar.memref_address %alloc_18 : memref<65536xf8E4M3FN, 2>
        %72 = ascscalar.memref_address %alloc_17 : memref<65536xf8E4M3FN, 2>
        %73 = arith.select %70, %71, %72 : i64
        %74 = arith.cmpi ne, %arg17, %c0_i64 : i64
        %75 = arith.extui %74 : i1 to i64
        %76 = arith.addi %arg17, %c1_i64 overflow<nsw> : i64
        %77 = arith.remui %76, %c2_i64 : i64
        %78 = ascscalar.bind_local_ptr %73 : memref<1xf8E4M3FN, 2>
        %79 = arith.divui %47, %c8_i64 : i64
        %80 = arith.divui %42, %c8_i64 : i64
        %81 = arith.addi %79, %80 : i64
        %82 = arith.muli %81, %c4096_i64 : i64
        %83 = arith.divui %69, %c8_i64 : i64
        %84 = arith.addi %82, %83 : i64
        %85 = arith.muli %84, %c8_i64 : i64
        asccube.get_buf mte2, %75, 0 {cannir.buffer.generated_sync}
        scf.if %58 {
          %94 = arith.index_cast %85 : i64 to index
          %95 = arith.trunci %53 : i64 to i32
          asccube.copy_gm2l1 %78[%c0], %alloc_3[%94], %95, %c512_i32, %c4096_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xf8E4M3FN, 2>, memref<33554432xui8>
        }
        asccube.release_buf mte2, %75, 0 {cannir.buffer.generated_sync}
        %86 = arith.cmpi sge, %arg16, %c1_i64 : i64
        %87 = arith.select %86, %true, %59 : i1
        %88 = scf.if %87 -> (i64) {
          %94 = arith.cmpi eq, %arg18, %c0_i64 : i64
          %95 = ascscalar.memref_address %alloc_16 : memref<131072xf8E4M3FN, 2>
          %96 = ascscalar.memref_address %alloc_15 : memref<131072xf8E4M3FN, 2>
          %97 = arith.select %94, %95, %96 : i64
          %98 = arith.select %94, %c2_i64, %c3_i64 : i64
          %99 = arith.addi %arg18, %c1_i64 : i64
          %100 = arith.remui %99, %c2_i64 : i64
          %101 = arith.subi %52, %48 : i64
          %102 = ascscalar.bind_local_ptr %97 : memref<1xf8E4M3FN, 2>
          %103 = arith.divui %48, %c8_i64 : i64
          %104 = arith.addi %103, %80 : i64
          %105 = arith.muli %104, %c4096_i64 : i64
          %106 = arith.addi %105, %83 : i64
          %107 = arith.muli %106, %c8_i64 : i64
          asccube.get_buf mte2, %98, 0 {cannir.buffer.generated_sync}
          %108 = arith.cmpi ugt, %101, %c0_i64 : i64
          scf.if %108 {
            %109 = arith.index_cast %107 : i64 to index
            %110 = arith.trunci %101 : i64 to i32
            asccube.copy_gm2l1 %102[%c0], %alloc[%109], %110, %c512_i32, %c4096_i64, %c1_i32, %c0_i64, %c0_i64, %c256_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xf8E4M3FN, 2>, memref<33554432xf8E4M3FN>
          }
          asccube.release_buf mte2, %98, 0 {cannir.buffer.generated_sync}
          scf.yield %100 : i64
        } else {
          scf.yield %arg18 : i64
        }
        %89 = ascscalar.memref_address %alloc_16 : memref<131072xf8E4M3FN, 2>
        %90 = ascscalar.memref_address %alloc_15 : memref<131072xf8E4M3FN, 2>
        %91 = arith.select %70, %89, %90 : i64
        %92 = arith.select %70, %c2_i64, %c3_i64 : i64
        %93 = scf.for %arg19 = %c0_i64 to %c4_i64 step %c1_i64 iter_args(%arg20 = %c0_i64) -> (i64)  : i64 {
          %94 = arith.muli %arg16, %c4_i64 overflow<nsw> : i64
          %95 = arith.addi %94, %arg19 overflow<nsw> : i64
          %96 = arith.cmpi eq, %arg20, %c0_i64 : i64
          %97 = ascscalar.memref_address %alloc_14 : memref<16384xf8E4M3FN, 3>
          %98 = ascscalar.memref_address %alloc_13 : memref<16384xf8E4M3FN, 3>
          %99 = arith.select %96, %97, %98 : i64
          %100 = arith.select %96, %c6_i64, %c7_i64 : i64
          %101 = arith.addi %arg20, %c1_i64 overflow<nsw> : i64
          %102 = arith.remui %101, %c2_i64 : i64
          %103 = ascscalar.memref_address %alloc_12 : memref<32768xf8E4M3FN, 4>
          %104 = ascscalar.memref_address %alloc_11 : memref<32768xf8E4M3FN, 4>
          %105 = arith.select %96, %103, %104 : i64
          %106 = arith.select %96, %c8_i64, %c9_i64 : i64
          %107 = ascscalar.bind_local_ptr %99 : memref<1xf8E4M3FN, 3>
          %108 = arith.muli %arg19, %c4_i64 overflow<nsw> : i64
          %109 = arith.muli %95, %c4_i64 overflow<nsw> : i64
          %110 = arith.index_cast %109 : i64 to index
          %111 = arith.divui %110, %c2 : index
          %112 = arith.index_cast %111 : index to i64
          %113 = ascscalar.memref_address %107 : memref<1xf8E4M3FN, 3>
          %114 = ascscalar.bind_local_ptr %113 : memref<?xi8, 6>
          asccube.get_buf mte1, %100, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %75, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
          %115 = arith.trunci %108 : i64 to i32
          asccube.copy_l12l0a %107[%c0], %78[%c0], %c0_i32, %115, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<1xf8E4M3FN, 3>, memref<1xf8E4M3FN, 2>
          %116 = arith.trunci %112 : i64 to i32
          asccube.copy_l12l0a %114[%c0], %57[%c0], %c0_i32, %116, %c8_i32, %c2_i32, %c128_i32, %c2_i32 : memref<?xi8, 6>, memref<?x!ascdtype.fp8_e8m0, 2>
          asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %75, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %100, 0 {cannir.buffer.generated_sync}
          %117 = ascscalar.bind_local_ptr %105 : memref<1xf8E4M3FN, 4>
          %118 = ascscalar.bind_local_ptr %91 : memref<1xf8E4M3FN, 2>
          %119 = ascscalar.memref_address %117 : memref<1xf8E4M3FN, 4>
          %120 = ascscalar.bind_local_ptr %119 : memref<?xi8, 7>
          asccube.get_buf mte1, %106, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %92, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %117[%c0], %118[%c0], %c0_i32, %115, %c16_i32, %c4_i32, %c16_i32, %c16_i32 : memref<1xf8E4M3FN, 4>, memref<1xf8E4M3FN, 2>
          asccube.copy_l12l0b %120[%c0], %12[%c0], %c0_i32, %116, %c16_i32, %c2_i32, %c128_i32, %c2_i32 : memref<?xi8, 7>, memref<?x!ascdtype.fp8_e8m0, 2>
          asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %92, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %106, 0 {cannir.buffer.generated_sync}
          %121 = arith.cmpi eq, %arg16, %c0_i64 : i64
          %122 = arith.cmpi eq, %arg19, %c0_i64 : i64
          %123 = arith.select %121, %122, %false : i1
          %124 = arith.cmpi eq, %arg16, %c7_i64 : i64
          %125 = arith.cmpi eq, %arg19, %c3_i64 : i64
          %126 = arith.select %124, %125, %false : i1
          %127 = arith.select %126, %c3_i64, %c2_i64 : i64
          %128 = arith.trunci %127 : i64 to i32
          %129 = ascscalar.bind_local_ptr %63 : memref<1xf32, 5>
          scf.if %123 {
            asccube.get_buf m, %100, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %106, 0 {cannir.buffer.generated_sync}
            asccube.mmad %129[%c0], %107, %117 scale(%114, %120), %c128_i32, %c128_i32, %c256_i32, %128 {c_matrix_init_val = true, mx_mode = true} : memref<1xf32, 5>, memref<1xf8E4M3FN, 3>, memref<1xf8E4M3FN, 4>, memref<?xi8, 6>, memref<?xi8, 7>
            asccube.release_buf m, %106, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %100, 0 {cannir.buffer.generated_sync}
          } else {
            asccube.get_buf m, %100, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %106, 0 {cannir.buffer.generated_sync}
            asccube.mmad %129[%c0], %107, %117 scale(%114, %120), %c128_i32, %c128_i32, %c256_i32, %128 {c_matrix_init_val = false, mx_mode = true} : memref<1xf32, 5>, memref<1xf8E4M3FN, 3>, memref<1xf8E4M3FN, 4>, memref<?xi8, 6>, memref<?xi8, 7>
            asccube.release_buf m, %106, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %100, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %102 : i64
        }
        scf.yield %77, %88 : i64, i64
      }
      %67 = ascscalar.bitcast_memref %alloc_8 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
      %68 = ascscalar.bind_local_ptr %63 : memref<1xf32, 5>
      asccube.sync_intra_wait fixpipe, %c0_i64 {cannir.sync.banks = 3 : i32}
      asccube.sync_intra_wait fixpipe, %c16_i64 {cannir.sync.banks = 3 : i32}
      asccube.copy_l0c2ub %67[%c0], %68[%c0], %c256_i32, %c128_i32, %c256_i32, %c128_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<1xf32, 5>
      asccube.sync_intra_arrive fixpipe, %c0_i64 {cannir.sync.banks = 3 : i32}
      asccube.sync_intra_arrive fixpipe, %c16_i64 {cannir.sync.banks = 3 : i32}
      scf.yield %66#1, %65 : i64, i64
    }
    asccube.sync_all
    %14 = arith.remsi %0, %c20_i64 : i64
    %15 = arith.cmpi eq, %13#0, %c0_i64 : i64
    %16 = ascscalar.memref_address %alloc_16 : memref<131072xf8E4M3FN, 2>
    %17 = ascscalar.memref_address %alloc_15 : memref<131072xf8E4M3FN, 2>
    %18 = arith.select %15, %16, %17 : i64
    %19 = arith.select %15, %c2_i64, %c3_i64 : i64
    %20 = arith.addi %13#0, %c1_i64 : i64
    %21 = arith.remui %20, %c2_i64 : i64
    %22 = arith.muli %14, %c256_i64 overflow<nsw> : i64
    %23 = arith.addi %14, %c1_i64 overflow<nsw> : i64
    %24 = arith.muli %23, %c256_i64 overflow<nsw> : i64
    %25 = arith.minsi %24, %c5120_i64 : i64
    %26 = arith.subi %25, %22 overflow<nsw> : i64
    %27 = ascscalar.bind_local_ptr %18 : memref<1xf8E4M3FN, 2>
    %28 = arith.divui %22, %c8_i64 : i64
    %29 = arith.muli %28, %c65536_i64 : i64
    asccube.get_buf mte2, %19, 0 {cannir.buffer.generated_sync}
    %30 = arith.cmpi ugt, %26, %c0_i64 : i64
    scf.if %30 {
      %33 = arith.index_cast %29 : i64 to index
      %34 = arith.trunci %26 : i64 to i32
      asccube.copy_gm2l1 %27[%c0], %alloc_1[%33], %34, %c512_i32, %c8192_i64, %c1_i32, %c0_i64, %c0_i64, %c256_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xf8E4M3FN, 2>, memref<41943040xf8E4M3FN>
    }
    asccube.release_buf mte2, %19, 0 {cannir.buffer.generated_sync}
    %31 = arith.muli %14, %c65536_i64 overflow<nsw> : i64
    scf.if %30 {
      %33 = arith.index_cast %31 : i64 to index
      %34 = arith.trunci %26 : i64 to i32
      asccube.copy_gm2l1 %12[%c0], %alloc_2[%33], %c128_i32, %34, %c256_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {dn2nz = true, mx_scale = true} : memref<?x!ascdtype.fp8_e8m0, 2>, memref<1310720xui8>
    }
    asccube.sync_intra_wait fixpipe, %c0_i64
    asccube.sync_intra_wait fixpipe, %c16_i64
    asccube.sync_all
    asccube.sync_block_wait s, %c0_i64
    asccube.sync_block_arrive fixpipe, %c1_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c1_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c2_i64
    %32:2 = scf.for %arg13 = %0 to %c160_i64 step %c32_i64 iter_args(%arg14 = %21, %arg15 = %13#1) -> (i64, i64)  : i64 {
      %33 = arith.divsi %arg13, %c20_i64 : i64
      %34 = arith.remsi %33, %c8_i64 : i64
      %35 = arith.cmpi ne, %34, %c0_i64 : i64
      %36 = arith.cmpi slt, %34, %c0_i64 : i64
      %37 = arith.andi %35, %36 : i1
      %38 = arith.select %37, %c8_i64, %c0_i64 : i64
      %39 = arith.addi %34, %38 overflow<nsw> : i64
      %40 = arith.remsi %arg13, %c20_i64 : i64
      %41 = arith.cmpi slt, %arg13, %c32_i64 : i64
      %42 = arith.muli %39, %c128_i64 overflow<nsw> : i64
      %43 = arith.muli %40, %c256_i64 overflow<nsw> : i64
      %44 = arith.addi %42, %c128_i64 overflow<nsw> : i64
      %45 = arith.minsi %44, %c1024_i64 : i64
      %46 = arith.addi %43, %c256_i64 overflow<nsw> : i64
      %47 = arith.minsi %46, %c5120_i64 : i64
      %48 = arith.subi %45, %42 overflow<nsw> : i64
      %49 = arith.muli %39, %c32768_i64 overflow<nsw> : i64
      %50 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<32768x!ascdtype.fp8_e8m0, 2> -> memref<?x!ascdtype.fp8_e8m0, 2>
      asccube.get_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
      %51 = arith.cmpi ugt, %48, %c0_i64 : i64
      scf.if %51 {
        %67 = arith.index_cast %49 : i64 to index
        %68 = arith.trunci %48 : i64 to i32
        asccube.copy_gm2l1 %50[%c0], %alloc_6[%67], %c128_i32, %68, %c256_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {dn2nz = true, mx_scale = true} : memref<?x!ascdtype.fp8_e8m0, 2>, memref<262144xui8>
      }
      asccube.release_buf mte2, %c4_i64, 0 {cannir.buffer.generated_sync}
      %52 = arith.cmpi eq, %41, %false : i1
      scf.if %52 {
        %67 = arith.subi %47, %43 overflow<nsw> : i64
        %68 = arith.muli %40, %c65536_i64 overflow<nsw> : i64
        asccube.get_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
        %69 = arith.cmpi ugt, %67, %c0_i64 : i64
        scf.if %69 {
          %70 = arith.index_cast %68 : i64 to index
          %71 = arith.trunci %67 : i64 to i32
          asccube.copy_gm2l1 %12[%c0], %alloc_2[%70], %c128_i32, %71, %c256_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {dn2nz = true, mx_scale = true} : memref<?x!ascdtype.fp8_e8m0, 2>, memref<1310720xui8>
        }
        asccube.release_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
      }
      %53 = arith.cmpi eq, %arg15, %c0_i64 : i64
      %54 = ascscalar.memref_address %alloc_20 : memref<32768xf32, 5>
      %55 = ascscalar.memref_address %alloc_19 : memref<32768xf32, 5>
      %56 = arith.select %53, %54, %55 : i64
      %57 = arith.addi %arg15, %c1_i64 overflow<nsw> : i64
      %58 = arith.remui %57, %c2_i64 : i64
      %59:2 = scf.for %arg16 = %c0_i64 to %c16_i64 step %c1_i64 iter_args(%arg17 = %c0_i64, %arg18 = %arg14) -> (i64, i64)  : i64 {
        %67 = arith.muli %arg16, %c512_i64 overflow<nsw> : i64
        %68 = arith.cmpi eq, %arg17, %c0_i64 : i64
        %69 = ascscalar.memref_address %alloc_18 : memref<65536xf8E4M3FN, 2>
        %70 = ascscalar.memref_address %alloc_17 : memref<65536xf8E4M3FN, 2>
        %71 = arith.select %68, %69, %70 : i64
        %72 = arith.cmpi ne, %arg17, %c0_i64 : i64
        %73 = arith.extui %72 : i1 to i64
        %74 = arith.addi %arg17, %c1_i64 overflow<nsw> : i64
        %75 = arith.remui %74, %c2_i64 : i64
        %76 = ascscalar.bind_local_ptr %71 : memref<1xf8E4M3FN, 2>
        %77 = arith.divui %42, %c8_i64 : i64
        %78 = arith.muli %77, %c8192_i64 overflow<nsw> : i64
        %79 = arith.divui %67, %c8_i64 : i64
        %80 = arith.addi %78, %79 overflow<nsw> : i64
        %81 = arith.muli %80, %c8_i64 overflow<nsw> : i64
        asccube.get_buf mte2, %73, 0 {cannir.buffer.generated_sync}
        scf.if %51 {
          %88 = arith.index_cast %81 : i64 to index
          %89 = arith.trunci %48 : i64 to i32
          asccube.copy_gm2l1 %76[%c0], %alloc_5[%88], %89, %c512_i32, %c8192_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xf8E4M3FN, 2>, memref<8388608xui8>
        }
        asccube.release_buf mte2, %73, 0 {cannir.buffer.generated_sync}
        %82 = arith.cmpi sge, %arg16, %c1_i64 : i64
        %83 = arith.select %82, %true, %52 : i1
        %84 = scf.if %83 -> (i64) {
          %88 = arith.cmpi eq, %arg18, %c0_i64 : i64
          %89 = arith.select %88, %16, %17 : i64
          %90 = arith.select %88, %c2_i64, %c3_i64 : i64
          %91 = arith.addi %arg18, %c1_i64 : i64
          %92 = arith.remui %91, %c2_i64 : i64
          %93 = arith.subi %47, %43 overflow<nsw> : i64
          %94 = ascscalar.bind_local_ptr %89 : memref<1xf8E4M3FN, 2>
          %95 = arith.divui %43, %c8_i64 : i64
          %96 = arith.muli %95, %c8192_i64 : i64
          %97 = arith.addi %96, %79 : i64
          %98 = arith.muli %97, %c8_i64 : i64
          asccube.get_buf mte2, %90, 0 {cannir.buffer.generated_sync}
          %99 = arith.cmpi ugt, %93, %c0_i64 : i64
          scf.if %99 {
            %100 = arith.index_cast %98 : i64 to index
            %101 = arith.trunci %93 : i64 to i32
            asccube.copy_gm2l1 %94[%c0], %alloc_1[%100], %101, %c512_i32, %c8192_i64, %c1_i32, %c0_i64, %c0_i64, %c256_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xf8E4M3FN, 2>, memref<41943040xf8E4M3FN>
          }
          asccube.release_buf mte2, %90, 0 {cannir.buffer.generated_sync}
          scf.yield %92 : i64
        } else {
          scf.yield %arg18 : i64
        }
        %85 = arith.select %68, %16, %17 : i64
        %86 = arith.select %68, %c2_i64, %c3_i64 : i64
        %87 = scf.for %arg19 = %c0_i64 to %c4_i64 step %c1_i64 iter_args(%arg20 = %c0_i64) -> (i64)  : i64 {
          %88 = arith.muli %arg16, %c4_i64 overflow<nsw> : i64
          %89 = arith.addi %88, %arg19 overflow<nsw> : i64
          %90 = arith.cmpi eq, %arg20, %c0_i64 : i64
          %91 = ascscalar.memref_address %alloc_14 : memref<16384xf8E4M3FN, 3>
          %92 = ascscalar.memref_address %alloc_13 : memref<16384xf8E4M3FN, 3>
          %93 = arith.select %90, %91, %92 : i64
          %94 = arith.select %90, %c6_i64, %c7_i64 : i64
          %95 = arith.addi %arg20, %c1_i64 overflow<nsw> : i64
          %96 = arith.remui %95, %c2_i64 : i64
          %97 = ascscalar.memref_address %alloc_12 : memref<32768xf8E4M3FN, 4>
          %98 = ascscalar.memref_address %alloc_11 : memref<32768xf8E4M3FN, 4>
          %99 = arith.select %90, %97, %98 : i64
          %100 = arith.select %90, %c8_i64, %c9_i64 : i64
          %101 = ascscalar.bind_local_ptr %93 : memref<1xf8E4M3FN, 3>
          %102 = arith.muli %arg19, %c4_i64 overflow<nsw> : i64
          %103 = arith.muli %89, %c4_i64 overflow<nsw> : i64
          %104 = arith.index_cast %103 : i64 to index
          %105 = arith.divui %104, %c2 : index
          %106 = arith.index_cast %105 : index to i64
          %107 = ascscalar.memref_address %101 : memref<1xf8E4M3FN, 3>
          %108 = ascscalar.bind_local_ptr %107 : memref<?xi8, 6>
          asccube.get_buf mte1, %94, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %73, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
          %109 = arith.trunci %102 : i64 to i32
          asccube.copy_l12l0a %101[%c0], %76[%c0], %c0_i32, %109, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<1xf8E4M3FN, 3>, memref<1xf8E4M3FN, 2>
          %110 = arith.trunci %106 : i64 to i32
          asccube.copy_l12l0a %108[%c0], %50[%c0], %c0_i32, %110, %c8_i32, %c2_i32, %c128_i32, %c2_i32 : memref<?xi8, 6>, memref<?x!ascdtype.fp8_e8m0, 2>
          asccube.release_buf mte1, %c4_i64, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %73, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %94, 0 {cannir.buffer.generated_sync}
          %111 = ascscalar.bind_local_ptr %99 : memref<1xf8E4M3FN, 4>
          %112 = ascscalar.bind_local_ptr %85 : memref<1xf8E4M3FN, 2>
          %113 = ascscalar.memref_address %111 : memref<1xf8E4M3FN, 4>
          %114 = ascscalar.bind_local_ptr %113 : memref<?xi8, 7>
          asccube.get_buf mte1, %100, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %86, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %111[%c0], %112[%c0], %c0_i32, %109, %c16_i32, %c4_i32, %c16_i32, %c16_i32 : memref<1xf8E4M3FN, 4>, memref<1xf8E4M3FN, 2>
          asccube.copy_l12l0b %114[%c0], %12[%c0], %c0_i32, %110, %c16_i32, %c2_i32, %c128_i32, %c2_i32 : memref<?xi8, 7>, memref<?x!ascdtype.fp8_e8m0, 2>
          asccube.release_buf mte1, %c5_i64, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %86, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %100, 0 {cannir.buffer.generated_sync}
          %115 = arith.cmpi eq, %arg16, %c0_i64 : i64
          %116 = arith.cmpi eq, %arg19, %c0_i64 : i64
          %117 = arith.select %115, %116, %false : i1
          %118 = arith.cmpi eq, %arg16, %c15_i64 : i64
          %119 = arith.cmpi eq, %arg19, %c3_i64 : i64
          %120 = arith.select %118, %119, %false : i1
          %121 = arith.select %120, %c3_i64, %c2_i64 : i64
          %122 = arith.trunci %121 : i64 to i32
          %123 = ascscalar.bind_local_ptr %56 : memref<1xf32, 5>
          scf.if %117 {
            asccube.get_buf m, %94, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %100, 0 {cannir.buffer.generated_sync}
            asccube.mmad %123[%c0], %101, %111 scale(%108, %114), %c128_i32, %c128_i32, %c256_i32, %122 {c_matrix_init_val = true, mx_mode = true} : memref<1xf32, 5>, memref<1xf8E4M3FN, 3>, memref<1xf8E4M3FN, 4>, memref<?xi8, 6>, memref<?xi8, 7>
            asccube.release_buf m, %100, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %94, 0 {cannir.buffer.generated_sync}
          } else {
            asccube.get_buf m, %94, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %100, 0 {cannir.buffer.generated_sync}
            asccube.mmad %123[%c0], %101, %111 scale(%108, %114), %c128_i32, %c128_i32, %c256_i32, %122 {c_matrix_init_val = false, mx_mode = true} : memref<1xf32, 5>, memref<1xf8E4M3FN, 3>, memref<1xf8E4M3FN, 4>, memref<?xi8, 6>, memref<?xi8, 7>
            asccube.release_buf m, %100, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %94, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %96 : i64
        }
        scf.yield %75, %84 : i64, i64
      }
      %60 = ascscalar.bind_local_ptr %56 : memref<1xf32, 5>
      %61 = arith.divui %42, %c16_i64 : i64
      %62 = arith.muli %61, %c5120_i64 overflow<nsw> : i64
      %63 = arith.divui %43, %c16_i64 : i64
      %64 = arith.addi %62, %63 : i64
      %65 = arith.muli %64, %c16_i64 : i64
      %66 = arith.index_cast %65 : i64 to index
      asccube.copy_l0c2gm %alloc_7[%66], %60[%c0], %c256_i32, %c128_i32, %c5120_i32, %c128_i32, %c3_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<5242880xbf16>, memref<1xf32, 5>
      scf.yield %59#1, %58 : i64, i64
    }
    return
  }
}
