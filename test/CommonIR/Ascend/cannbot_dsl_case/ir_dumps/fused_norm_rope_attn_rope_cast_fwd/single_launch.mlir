module {
  func.func @single_launch_6359b91d6e3959a3e4062a0650abdb0b(%arg0: !cannir.tensor<<gm, bf16>, <<(4, 32768)>, <(32768, 1)>>, <<(4, 32768)>, <(32768, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(32, 512)>, <(512, 1)>>, <<(32, 512)>, <(512, 1)>>, identity>, %arg2: !cannir.tensor<<gm, i32>, <<(4, 32)>, <(32, 1)>>, <<(4, 32)>, <(32, 1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(4)>, <(1)>>, <<(4)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i32>, <<(4)>, <(1)>>, <<(4)>, <(1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(37, 64)>, <(64, 1)>>, <<(37, 64)>, <(64, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(64)>, <(1)>>, <<(64)>, <(1)>>, identity>, %arg7: !cannir.tensor<<gm, f8E4M3FN>, <<(32, 4096)>, <(4096, 1)>>, <<(32, 4096)>, <(4096, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(1024)>, <(1)>>, <<(1024)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, f32>, <<(256)>, <(1)>>, <<(256)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, f32>, <<(256)>, <(1)>>, <<(256)>, <(1)>>, identity>, %arg11: f32, %arg12: f32) attributes {cannbotdsl.contract_digest = "6359b91d6e3959a3e4062a0650abdb0b52209605ef46ff6b7bc3cf20c934080b", cannbotdsl.public_slot_count = 13 : i64, cannbotdsl.slot_dtype_bits = [16, 16, 32, 32, 32, 32, 32, 8, 32, 32, 32, 32, 32], cannbotdsl.slot_dtype_codes = [4, 4, 0, 0, 0, 2, 2, 10, 0, 2, 2, 2, 2], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2], cannbotdsl.slot_type_digests = ["21d7143d3e6592b21080d5b617d72354e7b1f8e82c27f4b066aa762dab7e30c5", "ee69237cc8a54af28a4059f3bfadaeef457434319737e77a92cf712fc99c1cfe", "52c1544bb866c4be6a9b673f99b9dcfa9af44e35aa2f9a64e0653af8d815ced6", "ec5bf40ff868ab297a8582aafc213f00dc9214173b0931392ed5ab2c1a5e843f", "ec5bf40ff868ab297a8582aafc213f00dc9214173b0931392ed5ab2c1a5e843f", "ba32513815eaf7afa424b866bdcb700dddf6a0231106274a9b196ce9654b9b00", "3e5c3f7ef32e75fb1b4e3c39b6b6bf26ff018797c7a212af38e3c54bc1813413", "a01c9274ef9f7d790201ef382531ef31ea3591b6103f490a61ef2213271c1aab", "6282628c0e8481cc68f49c09abf09c87c594ce2be6de12604f51855d9322d600", "96a5c27dc4138cf0dd1cf506f06c22516bfcb9dd4755950302c0fbb0c8d0348f", "96a5c27dc4138cf0dd1cf506f06c22516bfcb9dd4755950302c0fbb0c8d0348f", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    %c4_i64 = arith.constant 4 : i64
    cannir.kernel_launch @kernel_fused_norm_rope_attn_rope_cast_fwd_kernel_0(%c4_i64) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10>, name = "fused_norm_rope_attn_rope_cast_fwd_kernel", op_type = "fused_norm_rope_attn_rope_cast_fwd_kernel", roles = array<i64: 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1>}} : !cannir.tensor<<gm, bf16>, <<(4, 32768)>, <(32768, 1)>>, <<(4, 32768)>, <(32768, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(32, 512)>, <(512, 1)>>, <<(32, 512)>, <(512, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(4, 32)>, <(32, 1)>>, <<(4, 32)>, <(32, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(4)>, <(1)>>, <<(4)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(4)>, <(1)>>, <<(4)>, <(1)>>, identity>, !cannir.tensor<<gm, f32>, <<(37, 64)>, <(64, 1)>>, <<(37, 64)>, <(64, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(64)>, <(1)>>, <<(64)>, <(1)>>, identity>, !cannir.tensor<<gm, f8E4M3FN>, <<(32, 4096)>, <(4096, 1)>>, <<(32, 4096)>, <(4096, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(1024)>, <(1)>>, <<(1024)>, <(1)>>, identity>, !cannir.tensor<<gm, f32>, <<(256)>, <(1)>>, <<(256)>, <(1)>>, identity>, !cannir.tensor<<gm, f32>, <<(256)>, <(1)>>, <<(256)>, <(1)>>, identity>, f32, f32
    return
  }
  func.func @kernel_fused_norm_rope_attn_rope_cast_fwd_kernel_0_aiv(%arg0: !cannir.tensor<<gm, bf16>, <<(4, 32768)>, <(32768, 1)>>, <<(4, 32768)>, <(32768, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(32, 512)>, <(512, 1)>>, <<(32, 512)>, <(512, 1)>>, identity>, %arg2: !cannir.tensor<<gm, i32>, <<(4, 32)>, <(32, 1)>>, <<(4, 32)>, <(32, 1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(4)>, <(1)>>, <<(4)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i32>, <<(4)>, <(1)>>, <<(4)>, <(1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(37, 64)>, <(64, 1)>>, <<(37, 64)>, <(64, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(64)>, <(1)>>, <<(64)>, <(1)>>, identity>, %arg7: !cannir.tensor<<gm, f8E4M3FN>, <<(32, 4096)>, <(4096, 1)>>, <<(32, 4096)>, <(4096, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(1024)>, <(1)>>, <<(1024)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, f32>, <<(256)>, <(1)>>, <<(256)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, f32>, <<(256)>, <(1)>>, <<(256)>, <(1)>>, identity>, %arg11: f32, %arg12: f32) attributes {cannir.core = "aiv", cannir.cube_state_mode = 1 : i32, cannir.cube_state_planned, cannir.device, cannir.kernel_group = "kernel_fused_norm_rope_attn_rope_cast_fwd_kernel_0"} {
    %c30864 = arith.constant 30864 : index
    %c30544 = arith.constant 30544 : index
    %c214016_i64 = arith.constant 214016 : i64
    %c219648_i64 = arith.constant 219648 : i64
    %c219392_i64 = arith.constant 219392 : i64
    %c219136_i64 = arith.constant 219136 : i64
    %c218880_i64 = arith.constant 218880 : i64
    %c197632_i64 = arith.constant 197632 : i64
    %c203264_i64 = arith.constant 203264 : i64
    %c203008_i64 = arith.constant 203008 : i64
    %c202752_i64 = arith.constant 202752 : i64
    %c202496_i64 = arith.constant 202496 : i64
    %c63008 = arith.constant 63008 : index
    %c1926144 = arith.constant 1926144 : index
    %c1917952 = arith.constant 1917952 : index
    %c115200 = arith.constant 115200 : index
    %c1913856 = arith.constant 1913856 : index
    %c1909760 = arith.constant 1909760 : index
    %c32 = arith.constant 32 : index
    %c1922048 = arith.constant 1922048 : index
    %c60064 = arith.constant 60064 : index
    %c260096 = arith.constant 260096 : index
    %c243712 = arith.constant 243712 : index
    %c30720 = arith.constant 30720 : index
    %c14336 = arith.constant 14336 : index
    %c49152 = arith.constant 49152 : index
    %c0_i64 = arith.constant 0 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c4_i64 = arith.constant 4 : i64
    %c10_i64 = arith.constant 10 : i64
    %c5_i64 = arith.constant 5 : i64
    %c11_i64 = arith.constant 11 : i64
    %c1_i64 = arith.constant 1 : i64
    %c6_i64 = arith.constant 6 : i64
    %c7_i64 = arith.constant 7 : i64
    %c8_i64 = arith.constant 8 : i64
    %c9_i64 = arith.constant 9 : i64
    %c63_i64 = arith.constant 63 : i64
    %c64_i64 = arith.constant 64 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c128_i64 = arith.constant 128 : i64
    %c256_i64 = arith.constant 256 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c16_i64 = arith.constant 16 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %c2_i64 = arith.constant 2 : i64
    %c32_i64 = arith.constant 32 : i64
    %c512_i64 = arith.constant 512 : i64
    %c786432_i64 = arith.constant 786432 : i64
    %c2621440_i64 = arith.constant 2621440 : i64
    %c524288_i64 = arith.constant 524288 : i64
    %c2359296_i64 = arith.constant 2359296 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c2883584_i64 = arith.constant 2883584 : i64
    %c3_i64 = arith.constant 3 : i64
    %c1310720_i64 = arith.constant 1310720 : i64
    %c3145728_i64 = arith.constant 3145728 : i64
    %true = arith.constant true
    %c12_i64 = arith.constant 12 : i64
    %c238592_i64 = arith.constant 238592 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c241792_i64 = arith.constant 241792 : i64
    %c57344_i64 = arith.constant 57344 : i64
    %c98304_i64 = arith.constant 98304 : i64
    %c102400_i64 = arith.constant 102400 : i64
    %c32_i32 = arith.constant 32 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %false = arith.constant false
    %c640_i64 = arith.constant 640 : i64
    %cst = arith.constant 6.000000e+00 : f32
    %c218624_i64 = arith.constant 218624 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c6144_i64 = arith.constant 6144 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c10240_i64 = arith.constant 10240 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %c14336_i64 = arith.constant 14336 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c18432_i64 = arith.constant 18432 : i64
    %c20480_i64 = arith.constant 20480 : i64
    %c22528_i64 = arith.constant 22528 : i64
    %c24576_i64 = arith.constant 24576 : i64
    %c13_i64 = arith.constant 13 : i64
    %c26624_i64 = arith.constant 26624 : i64
    %c14_i64 = arith.constant 14 : i64
    %c28672_i64 = arith.constant 28672 : i64
    %c15_i64 = arith.constant 15 : i64
    %c30720_i64 = arith.constant 30720 : i64
    %c16 = arith.constant 16 : index
    %c129024 = arith.constant 129024 : index
    %c1909760_i64 = arith.constant 1909760 : i64
    %c1917952_i64 = arith.constant 1917952 : i64
    %c3072_i64 = arith.constant 3072 : i64
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<131072xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<16384xbf16>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<128xi32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<4xi32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<4xi32>
    %alloc_4 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<2368xf32>
    %alloc_5 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<64xf32>
    %alloc_6 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<131072xf8E4M3FN>
    %alloc_7 = memref.alloc() {cannir.arg_index = 8 : i64} : memref<1024xi32>
    %alloc_8 = memref.alloc() {cannir.arg_index = 9 : i64} : memref<256xf32>
    %alloc_9 = memref.alloc() {cannir.arg_index = 10 : i64} : memref<256xf32>
    %0 = ascscalar.get_subblock_id
    %1 = ascscalar.bind_local_ptr %c0_i64 : memref<253952xui8, 1>
    %2 = ascscalar.bind_local_ptr %c0_i64 : memref<524288xui8, 2>
    %3 = cannir.datastruct_alloc : <"SparsePrefillCubeCursor", ["slot"], [ui32], [0], 4, 4>
    %4 = cannir.scalar_storage_cast %c0_i32 : i32 -> ui32
    cannir.datastruct_set %4, %3["slot"] : ui32, <"SparsePrefillCubeCursor", ["slot"], [ui32], [0], 4, 4>
    %5 = ascscalar.bind_local_ptr %c0_i64 : memref<3072xui8, 9>
    %6 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xbf16, 1>
    ascvec.vf(%6 : memref<?xbf16, 1>) {
    ^bb0(%arg13: memref<?xbf16, 1>):
      %c128_i32_10 = arith.constant 128 : i32
      %c16384_i32 = arith.constant 16384 : i32
      %c0_i32_11 = arith.constant 0 : i32
      %cst_12 = arith.constant 0.000000e+00 : bf16
      %108 = ascvec.create_mask elem_bits 16 : !ascvec.mask
      %109 = ascvec.duplicate_scalar_nomask %cst_12 : bf16 -> <bf16>
      scf.for %arg14 = %c0_i32_11 to %c16384_i32 step %c128_i32_10  : i32 {
        %110 = arith.index_cast %arg14 : i32 to index
        ascvec.storealign %arg13[%110], %109, %108 : memref<?xbf16, 1>, index, <bf16>
      }
    } {cannir.vf_mode = "simd"}
    ascvec.vf(%6 : memref<?xbf16, 1>) {
    ^bb0(%arg13: memref<?xbf16, 1>):
      %c128_i32_10 = arith.constant 128 : i32
      %c16384_i32 = arith.constant 16384 : i32
      %c0_i32_11 = arith.constant 0 : i32
      %c16384 = arith.constant 16384 : index
      %cst_12 = arith.constant 0.000000e+00 : bf16
      %108 = ascvec.create_mask elem_bits 16 : !ascvec.mask
      %109 = ascvec.duplicate_scalar_nomask %cst_12 : bf16 -> <bf16>
      scf.for %arg14 = %c0_i32_11 to %c16384_i32 step %c128_i32_10  : i32 {
        %110 = arith.index_cast %arg14 : i32 to index
        %111 = arith.addi %110, %c16384 : index
        ascvec.storealign %arg13[%111], %109, %108 : memref<?xbf16, 1>, index, <bf16>
      }
    } {cannir.vf_mode = "simd"}
    ascvec.vf(%6 : memref<?xbf16, 1>) {
    ^bb0(%arg13: memref<?xbf16, 1>):
      %c128_i32_10 = arith.constant 128 : i32
      %c16384_i32 = arith.constant 16384 : i32
      %c0_i32_11 = arith.constant 0 : i32
      %c32768 = arith.constant 32768 : index
      %cst_12 = arith.constant 0.000000e+00 : bf16
      %108 = ascvec.create_mask elem_bits 16 : !ascvec.mask
      %109 = ascvec.duplicate_scalar_nomask %cst_12 : bf16 -> <bf16>
      scf.for %arg14 = %c0_i32_11 to %c16384_i32 step %c128_i32_10  : i32 {
        %110 = arith.index_cast %arg14 : i32 to index
        %111 = arith.addi %110, %c32768 : index
        ascvec.storealign %arg13[%111], %109, %108 : memref<?xbf16, 1>, index, <bf16>
      }
    } {cannir.vf_mode = "simd"}
    ascvec.sync_all
    ascvec.sync_intra_arrive mte3, %c0_i64
    ascvec.sync_intra_arrive mte3, %c4_i64
    ascvec.sync_intra_arrive mte3, %c5_i64
    ascvec.get_buf mte2, %c12_i64, 0
    %7 = arith.muli %0, %c128_i64 : i64
    "ascvec.copy_gm2ub_plan"(%1, %alloc_5, %c0_i32, %c0_i32, %c238592_i64, %7, %c1_i64, %c128_i64, %c0_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 256 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 253952 : i64}> : (memref<253952xui8, 1>, memref<64xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
    ascvec.release_buf mte2, %c12_i64, 0
    ascvec.get_buf v, %c12_i64, 0
    ascvec.release_buf v, %c12_i64, 0
    %8 = cannir.datastruct_alloc : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.datastruct_set %c0_i64, %8["head"] : i64, <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %9 = cannir.datastruct_get %8["active"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %9[%c0] : <6xi64>
    cannir.array_store %c0_i64, %9[%c1] : <6xi64>
    cannir.array_store %c0_i64, %9[%c2] : <6xi64>
    cannir.array_store %c0_i64, %9[%c3] : <6xi64>
    cannir.array_store %c0_i64, %9[%c4] : <6xi64>
    cannir.array_store %c0_i64, %9[%c5] : <6xi64>
    %10 = cannir.datastruct_get %8["job"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %10[%c0] : <6xi64>
    cannir.array_store %c0_i64, %10[%c1] : <6xi64>
    cannir.array_store %c0_i64, %10[%c2] : <6xi64>
    cannir.array_store %c0_i64, %10[%c3] : <6xi64>
    cannir.array_store %c0_i64, %10[%c4] : <6xi64>
    cannir.array_store %c0_i64, %10[%c5] : <6xi64>
    %11 = cannir.datastruct_get %8["query"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %11[%c0] : <6xi64>
    cannir.array_store %c0_i64, %11[%c1] : <6xi64>
    cannir.array_store %c0_i64, %11[%c2] : <6xi64>
    cannir.array_store %c0_i64, %11[%c3] : <6xi64>
    cannir.array_store %c0_i64, %11[%c4] : <6xi64>
    cannir.array_store %c0_i64, %11[%c5] : <6xi64>
    %12 = cannir.datastruct_get %8["tile"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %12[%c0] : <6xi64>
    cannir.array_store %c0_i64, %12[%c1] : <6xi64>
    cannir.array_store %c0_i64, %12[%c2] : <6xi64>
    cannir.array_store %c0_i64, %12[%c3] : <6xi64>
    cannir.array_store %c0_i64, %12[%c4] : <6xi64>
    cannir.array_store %c0_i64, %12[%c5] : <6xi64>
    %13 = cannir.datastruct_get %8["blocks"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %13[%c0] : <6xi64>
    cannir.array_store %c0_i64, %13[%c1] : <6xi64>
    cannir.array_store %c0_i64, %13[%c2] : <6xi64>
    cannir.array_store %c0_i64, %13[%c3] : <6xi64>
    cannir.array_store %c0_i64, %13[%c4] : <6xi64>
    cannir.array_store %c0_i64, %13[%c5] : <6xi64>
    %14 = cannir.datastruct_get %8["length"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %14[%c0] : <6xi64>
    cannir.array_store %c0_i64, %14[%c1] : <6xi64>
    cannir.array_store %c0_i64, %14[%c2] : <6xi64>
    cannir.array_store %c0_i64, %14[%c3] : <6xi64>
    cannir.array_store %c0_i64, %14[%c4] : <6xi64>
    cannir.array_store %c0_i64, %14[%c5] : <6xi64>
    %15 = cannir.datastruct_get %8["global_block"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %15[%c0] : <6xi64>
    cannir.array_store %c0_i64, %15[%c1] : <6xi64>
    cannir.array_store %c0_i64, %15[%c2] : <6xi64>
    cannir.array_store %c0_i64, %15[%c3] : <6xi64>
    cannir.array_store %c0_i64, %15[%c4] : <6xi64>
    cannir.array_store %c0_i64, %15[%c5] : <6xi64>
    %16 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %17 = cannir.datastruct_get %8["active"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %18 = arith.index_cast %16 : i64 to index
    cannir.array_store %c0_i64, %17[%18] : <6xi64>
    %19 = cannir.datastruct_get %8["job"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %19[%18] : <6xi64>
    %20 = cannir.datastruct_get %8["query"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %20[%18] : <6xi64>
    %21 = cannir.datastruct_get %8["tile"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %21[%18] : <6xi64>
    %22 = cannir.datastruct_get %8["blocks"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %22[%18] : <6xi64>
    %23 = cannir.datastruct_get %8["length"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %23[%18] : <6xi64>
    %24 = cannir.datastruct_get %8["global_block"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %24[%18] : <6xi64>
    %25 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %26 = arith.addi %25, %c1_i64 : i64
    %27 = arith.cmpi eq, %26, %c6_i64 : i64
    %28 = arith.select %27, %c0_i64, %26 : i64
    cannir.datastruct_set %28, %8["head"] : i64, <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %29 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %30 = cannir.datastruct_get %8["active"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %31 = arith.index_cast %29 : i64 to index
    cannir.array_store %c0_i64, %30[%31] : <6xi64>
    %32 = cannir.datastruct_get %8["job"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %32[%31] : <6xi64>
    %33 = cannir.datastruct_get %8["query"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %33[%31] : <6xi64>
    %34 = cannir.datastruct_get %8["tile"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %34[%31] : <6xi64>
    %35 = cannir.datastruct_get %8["blocks"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %35[%31] : <6xi64>
    %36 = cannir.datastruct_get %8["length"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %36[%31] : <6xi64>
    %37 = cannir.datastruct_get %8["global_block"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %37[%31] : <6xi64>
    %38 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %39 = arith.addi %38, %c1_i64 : i64
    %40 = arith.cmpi eq, %39, %c6_i64 : i64
    %41 = arith.select %40, %c0_i64, %39 : i64
    cannir.datastruct_set %41, %8["head"] : i64, <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %42 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %43 = cannir.datastruct_get %8["active"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %44 = arith.index_cast %42 : i64 to index
    cannir.array_store %c0_i64, %43[%44] : <6xi64>
    %45 = cannir.datastruct_get %8["job"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %45[%44] : <6xi64>
    %46 = cannir.datastruct_get %8["query"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %46[%44] : <6xi64>
    %47 = cannir.datastruct_get %8["tile"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %47[%44] : <6xi64>
    %48 = cannir.datastruct_get %8["blocks"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %48[%44] : <6xi64>
    %49 = cannir.datastruct_get %8["length"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %49[%44] : <6xi64>
    %50 = cannir.datastruct_get %8["global_block"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %50[%44] : <6xi64>
    %51 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %52 = arith.addi %51, %c1_i64 : i64
    %53 = arith.cmpi eq, %52, %c6_i64 : i64
    %54 = arith.select %53, %c0_i64, %52 : i64
    cannir.datastruct_set %54, %8["head"] : i64, <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %55 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %56 = cannir.datastruct_get %8["active"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %57 = arith.index_cast %55 : i64 to index
    cannir.array_store %c0_i64, %56[%57] : <6xi64>
    %58 = cannir.datastruct_get %8["job"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %58[%57] : <6xi64>
    %59 = cannir.datastruct_get %8["query"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %59[%57] : <6xi64>
    %60 = cannir.datastruct_get %8["tile"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %60[%57] : <6xi64>
    %61 = cannir.datastruct_get %8["blocks"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %61[%57] : <6xi64>
    %62 = cannir.datastruct_get %8["length"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %62[%57] : <6xi64>
    %63 = cannir.datastruct_get %8["global_block"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %63[%57] : <6xi64>
    %64 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %65 = arith.addi %64, %c1_i64 : i64
    %66 = arith.cmpi eq, %65, %c6_i64 : i64
    %67 = arith.select %66, %c0_i64, %65 : i64
    cannir.datastruct_set %67, %8["head"] : i64, <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %68 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %69 = cannir.datastruct_get %8["active"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %70 = arith.index_cast %68 : i64 to index
    cannir.array_store %c0_i64, %69[%70] : <6xi64>
    %71 = cannir.datastruct_get %8["job"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %71[%70] : <6xi64>
    %72 = cannir.datastruct_get %8["query"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %72[%70] : <6xi64>
    %73 = cannir.datastruct_get %8["tile"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %73[%70] : <6xi64>
    %74 = cannir.datastruct_get %8["blocks"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %74[%70] : <6xi64>
    %75 = cannir.datastruct_get %8["length"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %75[%70] : <6xi64>
    %76 = cannir.datastruct_get %8["global_block"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %76[%70] : <6xi64>
    %77 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %78 = arith.addi %77, %c1_i64 : i64
    %79 = arith.cmpi eq, %78, %c6_i64 : i64
    %80 = arith.select %79, %c0_i64, %78 : i64
    cannir.datastruct_set %80, %8["head"] : i64, <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %81 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %82 = cannir.datastruct_get %8["active"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %83 = arith.index_cast %81 : i64 to index
    cannir.array_store %c0_i64, %82[%83] : <6xi64>
    %84 = cannir.datastruct_get %8["job"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %84[%83] : <6xi64>
    %85 = cannir.datastruct_get %8["query"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %85[%83] : <6xi64>
    %86 = cannir.datastruct_get %8["tile"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %86[%83] : <6xi64>
    %87 = cannir.datastruct_get %8["blocks"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %87[%83] : <6xi64>
    %88 = cannir.datastruct_get %8["length"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %88[%83] : <6xi64>
    %89 = cannir.datastruct_get %8["global_block"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    cannir.array_store %c0_i64, %89[%83] : <6xi64>
    %90 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %91 = arith.addi %90, %c1_i64 : i64
    %92 = arith.cmpi eq, %91, %c6_i64 : i64
    %93 = arith.select %92, %c0_i64, %91 : i64
    cannir.datastruct_set %93, %8["head"] : i64, <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
    %94 = ascscalar.get_block_idx
    %95 = arith.index_cast %94 : i64 to index
    %96 = memref.load %alloc_2[%95] : memref<4xi32>
    %97 = arith.extsi %96 : i32 to i64
    %98 = arith.addi %97, %c63_i64 : i64
    %99 = arith.remsi %98, %c64_i64 : i64
    %100 = arith.cmpi ne, %99, %c0_i64 : i64
    %101 = arith.cmpi slt, %99, %c0_i64 : i64
    %102 = arith.andi %100, %101 : i1
    %103 = arith.divsi %98, %c64_i64 : i64
    %104 = arith.extui %102 : i1 to i64
    %105 = arith.subi %103, %104 : i64
    %106 = arith.maxsi %105, %c1_i64 : i64
    %107:6 = scf.while (%arg13 = %c0_i64, %arg14 = %c0_i64, %arg15 = %94, %arg16 = %c0_i64, %arg17 = %97, %arg18 = %106) : (i64, i64, i64, i64, i64, i64) -> (i64, i64, i64, i64, i64, i64) {
      %108 = arith.cmpi slt, %arg14, %c2_i64 : i64
      scf.condition(%108) %arg13, %arg14, %arg15, %arg16, %arg17, %arg18 : i64, i64, i64, i64, i64, i64
    } do {
    ^bb0(%arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64, %arg17: i64, %arg18: i64):
      %108 = arith.cmpi eq, %arg14, %c1_i64 : i64
      %109 = arith.select %108, %c6_i64, %arg18 : i64
      %110 = arith.cmpi ne, %arg14, %c1_i64 : i64
      %111 = arith.extui %110 : i1 to i64
      %112 = scf.for %arg19 = %c0_i64 to %109 step %c1_i64 iter_args(%arg20 = %arg13) -> (i64)  : i64 {
        %114 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %115 = cannir.datastruct_get %8["active"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %116 = arith.index_cast %114 : i64 to index
        cannir.array_store %111, %115[%116] : <6xi64>
        %117 = cannir.datastruct_get %8["job"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        cannir.array_store %arg16, %117[%116] : <6xi64>
        %118 = cannir.datastruct_get %8["query"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        cannir.array_store %arg15, %118[%116] : <6xi64>
        %119 = cannir.datastruct_get %8["tile"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        cannir.array_store %arg19, %119[%116] : <6xi64>
        %120 = cannir.datastruct_get %8["blocks"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        cannir.array_store %arg18, %120[%116] : <6xi64>
        %121 = cannir.datastruct_get %8["length"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        cannir.array_store %arg17, %121[%116] : <6xi64>
        %122 = cannir.datastruct_get %8["global_block"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        cannir.array_store %arg20, %122[%116] : <6xi64>
        scf.if %110 {
          %202 = arith.cmpi eq, %arg19, %c0_i64 : i64
          scf.if %202 {
            ascvec.get_buf mte2, %c0_i64, 0
            %203 = arith.muli %arg15, %c128_i64 : i64
            "ascvec.copy_gm2ub_plan"(%1, %alloc_1, %c0_i32, %c0_i32, %c241792_i64, %203, %c1_i64, %c128_i64, %c0_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 512 : i64, pad_storage_bits = false, storage_dtype = i32, ub_base = 0 : i64, ub_capacity = 253952 : i64}> : (memref<253952xui8, 1>, memref<128xi32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %c0_i64, 0
          }
        }
        %123 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %124 = arith.addi %123, %c4_i64 : i64
        %125 = arith.cmpi sge, %124, %c6_i64 : i64
        %126 = arith.extui %125 : i1 to i64
        %127 = arith.muli %126, %c6_i64 : i64
        %128 = arith.subi %124, %127 : i64
        %129 = cannir.datastruct_get %8["active"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %130 = arith.index_cast %128 : i64 to index
        %131 = cannir.array_load %129[%130] : <6xi64>
        %132 = cannir.datastruct_get %8["job"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %133 = cannir.array_load %132[%130] : <6xi64>
        %134 = cannir.datastruct_get %8["query"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %135 = cannir.array_load %134[%130] : <6xi64>
        %136 = cannir.datastruct_get %8["tile"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %137 = cannir.array_load %136[%130] : <6xi64>
        %138 = cannir.datastruct_get %8["global_block"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %139 = cannir.array_load %138[%130] : <6xi64>
        %140 = arith.cmpi ne, %131, %c0_i64 : i64
        scf.if %140 {
          %202 = arith.cmpi eq, %137, %c0_i64 : i64
          scf.if %202 {
            %233 = arith.remsi %133, %c2_i64 : i64
            %234 = arith.cmpi ne, %233, %c0_i64 : i64
            %235 = arith.cmpi slt, %233, %c0_i64 : i64
            %236 = arith.andi %234, %235 : i1
            %237 = arith.select %236, %c2_i64, %c0_i64 : i64
            %238 = arith.addi %233, %237 overflow<nsw> : i64
            %239 = arith.addi %238, %c10_i64 overflow<nsw> : i64
            ascvec.sync_intra_wait mte2, %239
            ascvec.get_buf mte2, %c11_i64, 0
            %240 = arith.muli %135, %c65536_i64 : i64
            %241 = arith.muli %0, %c1024_i64 : i64
            %242 = arith.addi %240, %241 : i64
            %243 = arith.addi %242, %c57344_i64 : i64
            "ascvec.copy_gm2ub_plan"(%1, %alloc, %c0_i32, %c0_i32, %c98304_i64, %243, %c4_i64, %c1024_i64, %c2048_i64, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 262144 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 0 : i64, ub_capacity = 253952 : i64}> : (memref<253952xui8, 1>, memref<131072xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            %244 = arith.index_cast %135 : i64 to index
            %245 = memref.load %alloc_3[%244] : memref<4xi32>
            %246 = arith.extsi %245 : i32 to i64
            %247 = arith.muli %246, %c256_i64 : i64
            "ascvec.copy_gm2ub_plan"(%1, %alloc_4, %c0_i32, %c0_i32, %c102400_i64, %247, %c1_i64, %c256_i64, %c0_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 9472 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 253952 : i64}> : (memref<253952xui8, 1>, memref<2368xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %c11_i64, 0
            ascvec.get_buf v, %c11_i64, 0
            %248 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xf32, 1>
            ascvec.vf(%248, %6 : memref<?xf32, 1>, memref<?xbf16, 1>) {
            ^bb0(%arg21: memref<?xf32, 1>, %arg22: memref<?xbf16, 1>):
              %c1536_i32 = arith.constant 1536 : i32
              %c24_i32 = arith.constant 24 : i32
              %c1024_i32_10 = arith.constant 1024 : i32
              %c16_i32 = arith.constant 16 : i32
              %c512_i32 = arith.constant 512 : i32
              %c8_i32_11 = arith.constant 8 : i32
              %c49152_12 = arith.constant 49152 : index
              %c64_i32 = arith.constant 64 : i32
              %cst_13 = arith.constant -1.000000e+00 : f32
              %c32_i32_14 = arith.constant 32 : i32
              %c15_i32 = arith.constant 15 : i32
              %c1_i32_15 = arith.constant 1 : i32
              %c0_i32_16 = arith.constant 0 : i32
              %c25600 = arith.constant 25600 : index
              %250 = ascvec.create_mask : !ascvec.mask
              %251 = ascvec.loadalign %arg21[%c25600] : memref<?xf32, 1>, index -> <f32>
              %252 = ascvec.arange %c0_i32_16 : i32 -> <ui32>
              %253 = ascvec.shr %252, 1, %250 : <ui32> -> <ui32>
              %254 = ascvec.shl %253, 1, %250 : <ui32> -> <ui32>
              %255 = ascvec.add_scalar %254, %c1_i32_15, %250 : <ui32>, i32
              %256 = ascvec.duplicate_scalar_nomask %c1_i32_15 : i32 -> <ui32>
              %257 = ascvec.and %252, %256, %250 : <ui32>, <ui32> -> <ui32>
              %258 = ascvec.cmp_eq_scalar %257, %c0_i32_16, %250 : <ui32>, i32 -> !ascvec.mask
              %259 = ascvec.duplicate_scalar_nomask %c15_i32 : i32 -> <ui32>
              %260 = ascvec.and %252, %259, %250 : <ui32>, <ui32> -> <ui32>
              %261 = ascvec.shr %260, 1, %250 : <ui32> -> <ui32>
              %262 = ascvec.add_scalar %261, %c0_i32_16, %250 : <ui32>, i32
              %263 = ascvec.gather_reg %251, %262 : <f32>, <ui32> -> <f32>
              %264 = ascvec.add_scalar %262, %c32_i32_14, %250 : <ui32>, i32
              %265 = ascvec.gather_reg %251, %264 : <f32>, <ui32> -> <f32>
              %266 = ascvec.mul_scalar %265, %cst_13, %250 : <f32>, f32
              %267 = ascvec.select %263, %265, %258 : <f32>
              %268 = ascvec.select %266, %263, %258 : <f32>
              scf.for %arg23 = %c0_i32_16 to %c8_i32_11 step %c1_i32_15  : i32 {
                %290 = arith.muli %arg23, %c64_i32 overflow<nsw> : i32
                %291 = arith.index_cast %290 : i32 to index
                %292 = arith.addi %291, %c49152_12 : index
                %293 = ascvec.loadalign_unpack %arg22[%292] {mode = "b16_to_b32"} : memref<?xbf16, 1>, index -> <bf16>
                %294 = ascvec.cast %293, %250 {rounding = 6 : i32} : <bf16> -> <f32>
                %295 = ascvec.gather_reg %294, %254 : <f32>, <ui32> -> <f32>
                %296 = ascvec.gather_reg %294, %255 : <f32>, <ui32> -> <f32>
                %297 = ascvec.mul %296, %268, %250 : <f32>
                %298 = ascvec.madd_two_address %295, %267, %297, %250 : <f32>
                %299 = ascvec.cast %298, %250 : <f32> -> <bf16>
                ascvec.storealign_pack %arg22[%292], %299, %250 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
              }
              %269 = ascvec.add_scalar %261, %c8_i32_11, %250 : <ui32>, i32
              %270 = ascvec.gather_reg %251, %269 : <f32>, <ui32> -> <f32>
              %271 = ascvec.add_scalar %269, %c32_i32_14, %250 : <ui32>, i32
              %272 = ascvec.gather_reg %251, %271 : <f32>, <ui32> -> <f32>
              %273 = ascvec.mul_scalar %272, %cst_13, %250 : <f32>, f32
              %274 = ascvec.select %270, %272, %258 : <f32>
              %275 = ascvec.select %273, %270, %258 : <f32>
              scf.for %arg23 = %c0_i32_16 to %c8_i32_11 step %c1_i32_15  : i32 {
                %290 = arith.muli %arg23, %c64_i32 overflow<nsw> : i32
                %291 = arith.addi %290, %c512_i32 overflow<nsw> : i32
                %292 = arith.index_cast %291 : i32 to index
                %293 = arith.addi %292, %c49152_12 : index
                %294 = ascvec.loadalign_unpack %arg22[%293] {mode = "b16_to_b32"} : memref<?xbf16, 1>, index -> <bf16>
                %295 = ascvec.cast %294, %250 {rounding = 6 : i32} : <bf16> -> <f32>
                %296 = ascvec.gather_reg %295, %254 : <f32>, <ui32> -> <f32>
                %297 = ascvec.gather_reg %295, %255 : <f32>, <ui32> -> <f32>
                %298 = ascvec.mul %297, %275, %250 : <f32>
                %299 = ascvec.madd_two_address %296, %274, %298, %250 : <f32>
                %300 = ascvec.cast %299, %250 : <f32> -> <bf16>
                ascvec.storealign_pack %arg22[%293], %300, %250 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
              }
              %276 = ascvec.add_scalar %261, %c16_i32, %250 : <ui32>, i32
              %277 = ascvec.gather_reg %251, %276 : <f32>, <ui32> -> <f32>
              %278 = ascvec.add_scalar %276, %c32_i32_14, %250 : <ui32>, i32
              %279 = ascvec.gather_reg %251, %278 : <f32>, <ui32> -> <f32>
              %280 = ascvec.mul_scalar %279, %cst_13, %250 : <f32>, f32
              %281 = ascvec.select %277, %279, %258 : <f32>
              %282 = ascvec.select %280, %277, %258 : <f32>
              scf.for %arg23 = %c0_i32_16 to %c8_i32_11 step %c1_i32_15  : i32 {
                %290 = arith.muli %arg23, %c64_i32 overflow<nsw> : i32
                %291 = arith.addi %290, %c1024_i32_10 overflow<nsw> : i32
                %292 = arith.index_cast %291 : i32 to index
                %293 = arith.addi %292, %c49152_12 : index
                %294 = ascvec.loadalign_unpack %arg22[%293] {mode = "b16_to_b32"} : memref<?xbf16, 1>, index -> <bf16>
                %295 = ascvec.cast %294, %250 {rounding = 6 : i32} : <bf16> -> <f32>
                %296 = ascvec.gather_reg %295, %254 : <f32>, <ui32> -> <f32>
                %297 = ascvec.gather_reg %295, %255 : <f32>, <ui32> -> <f32>
                %298 = ascvec.mul %297, %282, %250 : <f32>
                %299 = ascvec.madd_two_address %296, %281, %298, %250 : <f32>
                %300 = ascvec.cast %299, %250 : <f32> -> <bf16>
                ascvec.storealign_pack %arg22[%293], %300, %250 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
              }
              %283 = ascvec.add_scalar %261, %c24_i32, %250 : <ui32>, i32
              %284 = ascvec.gather_reg %251, %283 : <f32>, <ui32> -> <f32>
              %285 = ascvec.add_scalar %283, %c32_i32_14, %250 : <ui32>, i32
              %286 = ascvec.gather_reg %251, %285 : <f32>, <ui32> -> <f32>
              %287 = ascvec.mul_scalar %286, %cst_13, %250 : <f32>, f32
              %288 = ascvec.select %284, %286, %258 : <f32>
              %289 = ascvec.select %287, %284, %258 : <f32>
              scf.for %arg23 = %c0_i32_16 to %c8_i32_11 step %c1_i32_15  : i32 {
                %290 = arith.muli %arg23, %c64_i32 overflow<nsw> : i32
                %291 = arith.addi %290, %c1536_i32 overflow<nsw> : i32
                %292 = arith.index_cast %291 : i32 to index
                %293 = arith.addi %292, %c49152_12 : index
                %294 = ascvec.loadalign_unpack %arg22[%293] {mode = "b16_to_b32"} : memref<?xbf16, 1>, index -> <bf16>
                %295 = ascvec.cast %294, %250 {rounding = 6 : i32} : <bf16> -> <f32>
                %296 = ascvec.gather_reg %295, %254 : <f32>, <ui32> -> <f32>
                %297 = ascvec.gather_reg %295, %255 : <f32>, <ui32> -> <f32>
                %298 = ascvec.mul %297, %289, %250 : <f32>
                %299 = ascvec.madd_two_address %296, %288, %298, %250 : <f32>
                %300 = ascvec.cast %299, %250 : <f32> -> <bf16>
                ascvec.storealign_pack %arg22[%293], %300, %250 {mode = "b32_to_b16"} : memref<?xbf16, 1>, index, <bf16>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c11_i64, 0
            ascvec.get_buf mte3, %c11_i64, 0
            %249 = arith.cmpi eq, %0, %c0_i64 : i64
            scf.if %249 {
              %250 = arith.cmpi eq, %238, %c0_i64 : i64
              scf.if %250 {
                %251 = ascscalar.bitcast_memref %2 byte_offset = %c0_i64 : memref<524288xui8, 2> -> memref<?xbf16, 2>
                ascvec.copy_ub2l1_strided %251[%c14336], %6[%c49152], %c1_i32, %c128_i32, %c0_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              } else {
                %251 = ascscalar.bitcast_memref %2 byte_offset = %c0_i64 : memref<524288xui8, 2> -> memref<?xbf16, 2>
                ascvec.copy_ub2l1_strided %251[%c30720], %6[%c49152], %c1_i32, %c128_i32, %c0_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              }
            } else {
              %250 = arith.cmpi eq, %238, %c0_i64 : i64
              scf.if %250 {
                %251 = ascscalar.bitcast_memref %2 byte_offset = %c0_i64 : memref<524288xui8, 2> -> memref<?xbf16, 2>
                ascvec.copy_ub2l1_strided %251[%c243712], %6[%c49152], %c1_i32, %c128_i32, %c0_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              } else {
                %251 = ascscalar.bitcast_memref %2 byte_offset = %c0_i64 : memref<524288xui8, 2> -> memref<?xbf16, 2>
                ascvec.copy_ub2l1_strided %251[%c260096], %6[%c49152], %c1_i32, %c128_i32, %c0_i32, %c0_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
              }
            }
            ascvec.sync_intra_arrive mte3, %239
            ascvec.release_buf mte3, %c11_i64, 0
          }
          %203 = arith.remsi %139, %c4_i64 : i64
          %204 = arith.cmpi ne, %203, %c0_i64 : i64
          %205 = arith.cmpi slt, %203, %c0_i64 : i64
          %206 = arith.andi %204, %205 : i1
          %207 = arith.select %206, %c4_i64, %c0_i64 : i64
          %208 = arith.addi %203, %207 overflow<nsw> : i64
          %209 = arith.cmpi eq, %208, %c1_i64 : i64
          %210 = arith.select %209, %c786432_i64, %c524288_i64 : i64
          %211 = arith.select %209, %c2621440_i64, %c2359296_i64 : i64
          %212 = arith.cmpi eq, %208, %c2_i64 : i64
          %213 = arith.select %212, %c1048576_i64, %210 : i64
          %214 = arith.select %212, %c2883584_i64, %211 : i64
          %215 = arith.cmpi eq, %208, %c3_i64 : i64
          %216 = arith.select %215, %c1310720_i64, %213 : i64
          %217 = arith.select %215, %c3145728_i64, %214 : i64
          %218 = arith.remsi %139, %c3_i64 : i64
          %219 = arith.cmpi ne, %218, %c0_i64 : i64
          %220 = arith.cmpi slt, %218, %c0_i64 : i64
          %221 = arith.andi %219, %220 : i1
          %222 = arith.select %221, %c3_i64, %c0_i64 : i64
          %223 = arith.addi %218, %222 overflow<nsw> : i64
          %224 = arith.cmpi eq, %223, %c1_i64 : i64
          %225 = arith.select %224, %c262144_i64, %c0_i64 : i64
          %226 = arith.cmpi eq, %223, %c2_i64 : i64
          %227 = arith.select %226, %c524288_i64, %225 : i64
          %228 = arith.divui %227, %c8_i64 : i64
          %229 = ascscalar.bind_local_ptr %228 : memref<1xbf16, 1>
          %230 = arith.addi %223, %c8_i64 overflow<nsw> : i64
          ascvec.get_buf v, %230, 0
          ascvec.get_buf v, %c11_i64, 0
          ascvec.vf(%229, %6 : memref<1xbf16, 1>, memref<?xbf16, 1>) {
          ^bb0(%arg21: memref<1xbf16, 1>, %arg22: memref<?xbf16, 1>):
            %c4224_i32 = arith.constant 4224 : i32
            %c2_i32 = arith.constant 2 : i32
            %c4_i32_10 = arith.constant 4 : i32
            %c32_i32_11 = arith.constant 32 : i32
            %c0_i32_12 = arith.constant 0 : i32
            %c33 = arith.constant 33 : index
            %c49152_13 = arith.constant 49152 : index
            %c16_i32 = arith.constant 16 : i32
            %c1_i32_14 = arith.constant 1 : i32
            %c0_15 = arith.constant 0 : index
            %c128_i32_16 = arith.constant 128 : i32
            %c512_i32 = arith.constant 512 : i32
            %233 = ascvec.create_mask elem_bits 16 : !ascvec.mask
            scf.for %arg23 = %c0_i32_12 to %c4_i32_10 step %c2_i32  : i32 {
              scf.for %arg24 = %c0_i32_12 to %c32_i32_11 step %c1_i32_14  : i32 {
                %234 = arith.muli %arg24, %c512_i32 overflow<nsw> : i32
                %235 = arith.muli %arg23, %c128_i32_16 overflow<nsw> : i32
                %236 = arith.addi %234, %235 overflow<nsw> : i32
                %237 = arith.index_cast %236 : i32 to index
                %238 = ascvec.loadalign %arg21[%237] : memref<1xbf16, 1>, index -> <bf16>
                %239 = arith.addi %arg23, %c1_i32_14 overflow<nsw> : i32
                %240 = arith.muli %239, %c128_i32_16 overflow<nsw> : i32
                %241 = arith.addi %234, %240 overflow<nsw> : i32
                %242 = arith.index_cast %241 : i32 to index
                %243 = ascvec.loadalign %arg21[%242] : memref<1xbf16, 1>, index -> <bf16>
                %244 = arith.muli %arg24, %c16_i32 overflow<nsw> : i32
                %245 = arith.muli %arg23, %c4224_i32 overflow<nsw> : i32
                %246 = arith.addi %244, %245 overflow<nsw> : i32
                %247 = arith.index_cast %246 : i32 to index
                %248 = arith.addi %247, %c49152_13 : index
                ascvec.storealign_strided %arg22[%248], %238, %233, %c33, %c0_15 : memref<?xbf16, 1>, <bf16>
                %249 = arith.muli %239, %c4224_i32 overflow<nsw> : i32
                %250 = arith.addi %244, %249 overflow<nsw> : i32
                %251 = arith.index_cast %250 : i32 to index
                %252 = arith.addi %251, %c49152_13 : index
                ascvec.storealign_strided %arg22[%252], %243, %233, %c33, %c0_15 : memref<?xbf16, 1>, <bf16>
              }
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c11_i64, 0
          ascvec.release_buf v, %230, 0
          ascvec.get_buf mte3, %c11_i64, 0
          %231 = arith.addi %208, %c6_i64 overflow<nsw> : i64
          ascvec.sync_intra_wait mte3, %231
          %232 = arith.cmpi eq, %0, %c0_i64 : i64
          scf.if %232 {
            %233 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 2>
            %234 = arith.index_cast %216 : i64 to index
            %235 = arith.divui %234, %c16 : index
            ascvec.copy_ub2l1_strided %233[%235], %6[%c49152], %c32_i32, %c32_i32, %c1_i32, %c0_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
          } else {
            %233 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 2>
            %234 = arith.index_cast %217 : i64 to index
            %235 = arith.divui %234, %c16 : index
            ascvec.copy_ub2l1_strided %233[%235], %6[%c49152], %c32_i32, %c32_i32, %c1_i32, %c0_i32 : memref<1xbf16, 2>, memref<?xbf16, 1>
          }
          ascvec.sync_intra_arrive mte3, %231
          ascvec.release_buf mte3, %c11_i64, 0
        }
        %141 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %142 = arith.addi %141, %c3_i64 : i64
        %143 = arith.cmpi sge, %142, %c6_i64 : i64
        %144 = arith.extui %143 : i1 to i64
        %145 = arith.muli %144, %c6_i64 : i64
        %146 = arith.subi %142, %145 : i64
        %147 = cannir.datastruct_get %8["active"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %148 = arith.index_cast %146 : i64 to index
        %149 = cannir.array_load %147[%148] : <6xi64>
        %150 = cannir.datastruct_get %8["job"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %151 = cannir.array_load %150[%148] : <6xi64>
        %152 = cannir.datastruct_get %8["query"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %153 = cannir.array_load %152[%148] : <6xi64>
        %154 = cannir.datastruct_get %8["tile"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %155 = cannir.array_load %154[%148] : <6xi64>
        %156 = cannir.datastruct_get %8["global_block"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %157 = cannir.array_load %156[%148] : <6xi64>
        %158 = arith.cmpi ne, %149, %c0_i64 : i64
        %159 = arith.cmpi eq, %155, %c0_i64 : i64
        %160 = arith.select %158, %159, %false : i1
        scf.if %160 {
          %202 = arith.addi %157, %c1_i64 : i64
          %203 = arith.remsi %202, %c3_i64 : i64
          %204 = arith.cmpi ne, %203, %c0_i64 : i64
          %205 = arith.cmpi slt, %203, %c0_i64 : i64
          %206 = arith.andi %204, %205 : i1
          %207 = arith.select %206, %c3_i64, %c0_i64 : i64
          %208 = arith.addi %203, %207 overflow<nsw> : i64
          %209 = arith.cmpi eq, %208, %c1_i64 : i64
          %210 = arith.select %209, %c262144_i64, %c0_i64 : i64
          %211 = arith.cmpi eq, %208, %c2_i64 : i64
          %212 = arith.select %211, %c524288_i64, %210 : i64
          %213 = arith.addi %208, %c8_i64 overflow<nsw> : i64
          ascvec.get_buf mte2, %213, 0
          %214 = arith.muli %153, %c65536_i64 : i64
          %215 = arith.muli %0, %c1024_i64 : i64
          %216 = arith.addi %214, %215 : i64
          %217 = arith.divui %212, %c8_i64 : i64
          %218 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
          "ascvec.copy_gm2ub_plan"(%218, %alloc, %c0_i32, %c0_i32, %217, %216, %c32_i64, %c1024_i64, %c2048_i64, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 262144 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<131072xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %213, 0
        }
        %161 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %162 = arith.addi %161, %c2_i64 : i64
        %163 = arith.cmpi sge, %162, %c6_i64 : i64
        %164 = arith.extui %163 : i1 to i64
        %165 = arith.muli %164, %c6_i64 : i64
        %166 = arith.subi %162, %165 : i64
        %167 = cannir.datastruct_get %8["active"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %168 = arith.index_cast %166 : i64 to index
        %169 = cannir.array_load %167[%168] : <6xi64>
        %170 = cannir.datastruct_get %8["job"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %171 = cannir.array_load %170[%168] : <6xi64>
        %172 = cannir.datastruct_get %8["query"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %173 = cannir.array_load %172[%168] : <6xi64>
        %174 = cannir.datastruct_get %8["tile"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %175 = cannir.array_load %174[%168] : <6xi64>
        %176 = cannir.datastruct_get %8["blocks"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %177 = cannir.array_load %176[%168] : <6xi64>
        %178 = arith.cmpi ne, %169, %c0_i64 : i64
        scf.if %178 {
          %202 = arith.cmpi eq, %175, %c0_i64 : i64
          scf.if %202 {
            %210 = arith.remsi %171, %c2_i64 : i64
            %211 = arith.cmpi ne, %210, %c0_i64 : i64
            %212 = arith.cmpi slt, %210, %c0_i64 : i64
            %213 = arith.andi %211, %212 : i1
            %214 = arith.select %213, %c2_i64, %c0_i64 : i64
            %215 = arith.addi %210, %214 overflow<nsw> : i64
            ascvec.sync_intra_wait v, %c0_i64
            %216 = arith.addi %215, %c2_i64 overflow<nsw> : i64
            ascvec.get_buf v, %216, 0
            %217 = arith.muli %215, %c64_i64 overflow<nsw> : i64
            %218 = arith.addi %217, %c32_i64 overflow<nsw> : i64
            %219 = arith.index_cast %218 : i64 to index
            %220 = arith.remsi %171, %c4_i64 : i64
            %221 = arith.cmpi ne, %220, %c0_i64 : i64
            %222 = arith.cmpi slt, %220, %c0_i64 : i64
            %223 = arith.andi %221, %222 : i1
            %224 = arith.select %223, %c4_i64, %c0_i64 : i64
            %225 = arith.addi %220, %224 overflow<nsw> : i64
            %226 = arith.muli %225, %c640_i64 overflow<nsw> : i64
            %227 = arith.index_cast %226 : i64 to index
            %228 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xui32, 1>
            %229 = arith.muli %215, %c2048_i64 overflow<nsw> : i64
            %230 = arith.index_cast %229 : i64 to index
            %231 = arith.addi %230, %c1922048 : index
            %232 = arith.divui %231, %c32 : index
            %233 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xf32, 1>
            %234 = arith.addi %230, %c1909760 : index
            %235 = arith.divui %234, %c32 : index
            %236 = arith.addi %230, %c1913856 : index
            %237 = arith.divui %236, %c32 : index
            %238 = arith.addi %230, %c1917952 : index
            %239 = arith.divui %238, %c32 : index
            ascvec.vf(%228, %219, %1, %227, %233, %232, %233, %233, %235, %233, %237, %6, %233, %239 : memref<?xui32, 1>, index, memref<253952xui8, 1>, index, memref<?xf32, 1>, index, memref<?xf32, 1>, memref<?xf32, 1>, index, memref<?xf32, 1>, index, memref<?xbf16, 1>, memref<?xf32, 1>, index) {
            ^bb0(%arg21: memref<?xui32, 1>, %arg22: index, %arg23: memref<253952xui8, 1>, %arg24: index, %arg25: memref<?xf32, 1>, %arg26: index, %arg27: memref<?xf32, 1>, %arg28: memref<?xf32, 1>, %arg29: index, %arg30: memref<?xf32, 1>, %arg31: index, %arg32: memref<?xbf16, 1>, %arg33: memref<?xf32, 1>, %arg34: index):
              %c3072_i32 = arith.constant 3072 : i32
              %c1600_i32 = arith.constant 1600 : i32
              %c1088_i32 = arith.constant 1088 : i32
              %c2048_i32 = arith.constant 2048 : i32
              %c115200_10 = arith.constant 115200 : index
              %c576_i32 = arith.constant 576 : i32
              %c2_i32 = arith.constant 2 : i32
              %c1_i32_11 = arith.constant 1 : i32
              %c8_i32_12 = arith.constant 8 : i32
              %c1536_i32 = arith.constant 1536 : i32
              %c1024_i32_13 = arith.constant 1024 : i32
              %c512_i32 = arith.constant 512 : i32
              %c57600 = arith.constant 57600 : index
              %c64_i32 = arith.constant 64 : i32
              %c4_i32_14 = arith.constant 4 : i32
              %c48_i32 = arith.constant 48 : i32
              %c32_i32_15 = arith.constant 32 : i32
              %c16_i32 = arith.constant 16 : i32
              %c15_i32 = arith.constant 15 : i32
              %c249472 = arith.constant 249472 : index
              %c60064_16 = arith.constant 60064 : index
              %c0_i32_17 = arith.constant 0 : i32
              %cst_18 = arith.constant -3.40282347E+38 : f32
              %240 = ascvec.create_mask elem_bits 16 : !ascvec.mask
              %241 = ascvec.create_mask pat_mode "PAT_VL8" : !ascvec.mask
              %242 = ascvec.duplicate_scalar_nomask %cst_18 : f32 -> <f32>
              %243 = ascvec.duplicate_scalar_nomask %c0_i32_17 : i32 -> <ui32>
              %244 = arith.addi %arg22, %c60064_16 : index
              %c0_19 = arith.constant 0 : index
              %c1_20 = arith.constant 1 : index
              scf.for %arg35 = %c0_19 to %c1_20 step %c1_20 {
                ascvec.storealign_1st %arg21[%244], %243 : memref<?xui32, 1>, index, <ui32>
              }
              %245 = arith.addi %arg24, %c249472 : index
              %246 = ascvec.loadalign_unpack %arg23[%245] {mode = "b8_to_b32"} : memref<253952xui8, 1>, index -> <ui32>
              %247 = ascvec.arange %c0_i32_17 : i32 -> <ui32>
              %248 = ascvec.duplicate_scalar_nomask %c15_i32 : i32 -> <ui32>
              %249 = ascvec.and %247, %248, %240 : <ui32>, <ui32> -> <ui32>
              %250 = ascvec.gather_reg %246, %249 : <ui32>, <ui32> -> <ui32>
              %251 = ascvec.cmp_gt_scalar %250, %c0_i32_17, %240 : <ui32>, i32 -> !ascvec.mask
              %252 = ascvec.add_scalar %249, %c16_i32, %240 : <ui32>, i32
              %253 = ascvec.gather_reg %246, %252 : <ui32>, <ui32> -> <ui32>
              %254 = ascvec.cmp_gt_scalar %253, %c0_i32_17, %240 : <ui32>, i32 -> !ascvec.mask
              %255 = ascvec.add_scalar %249, %c32_i32_15, %240 : <ui32>, i32
              %256 = ascvec.gather_reg %246, %255 : <ui32>, <ui32> -> <ui32>
              %257 = ascvec.cmp_gt_scalar %256, %c0_i32_17, %240 : <ui32>, i32 -> !ascvec.mask
              %258 = ascvec.add_scalar %249, %c48_i32, %240 : <ui32>, i32
              %259 = ascvec.gather_reg %246, %258 : <ui32>, <ui32> -> <ui32>
              %260 = ascvec.cmp_gt_scalar %259, %c0_i32_17, %240 : <ui32>, i32 -> !ascvec.mask
              %261 = ascvec.loadalign %arg25[%arg26] : memref<?xf32, 1>, index -> <f32>
              %262 = ascvec.shr %247, 4, %240 : <ui32> -> <ui32>
              scf.for %arg35 = %c0_i32_17 to %c8_i32_12 step %c1_i32_11  : i32 {
                %263 = arith.muli %arg35, %c4_i32_14 overflow<nsw> : i32
                %264 = ascvec.add_scalar %262, %263, %240 : <ui32>, i32
                %265 = ascvec.gather_reg %261, %264 : <f32>, <ui32> -> <f32>
                %266 = arith.muli %arg35, %c64_i32 overflow<nsw> : i32
                %267 = arith.index_cast %266 : i32 to index
                %268 = arith.addi %267, %c57600 : index
                %269 = ascvec.loadalign %arg27[%268] : memref<?xf32, 1>, index -> <f32>
                %270 = ascvec.mul %269, %265, %240 : <f32>
                %271 = arith.addi %266, %c512_i32 overflow<nsw> : i32
                %272 = arith.index_cast %271 : i32 to index
                %273 = arith.addi %272, %c57600 : index
                %274 = ascvec.loadalign %arg27[%273] : memref<?xf32, 1>, index -> <f32>
                %275 = ascvec.mul %274, %265, %240 : <f32>
                %276 = arith.addi %266, %c1024_i32_13 overflow<nsw> : i32
                %277 = arith.index_cast %276 : i32 to index
                %278 = arith.addi %277, %c57600 : index
                %279 = ascvec.loadalign %arg27[%278] : memref<?xf32, 1>, index -> <f32>
                %280 = ascvec.mul %279, %265, %240 : <f32>
                %281 = arith.addi %266, %c1536_i32 overflow<nsw> : i32
                %282 = arith.index_cast %281 : i32 to index
                %283 = arith.addi %282, %c57600 : index
                %284 = ascvec.loadalign %arg27[%283] : memref<?xf32, 1>, index -> <f32>
                %285 = ascvec.mul %284, %265, %240 : <f32>
                ascvec.storealign %arg27[%268], %270, %240 : memref<?xf32, 1>, index, <f32>
                ascvec.storealign %arg27[%273], %275, %240 : memref<?xf32, 1>, index, <f32>
                ascvec.storealign %arg27[%278], %280, %240 : memref<?xf32, 1>, index, <f32>
                ascvec.storealign %arg27[%283], %285, %240 : memref<?xf32, 1>, index, <f32>
                %286 = ascvec.select %270, %242, %251 : <f32>
                %287 = ascvec.select %275, %242, %254 : <f32>
                %288 = ascvec.select %280, %242, %257 : <f32>
                %289 = ascvec.select %285, %242, %260 : <f32>
                %290 = ascvec.max %242, %286, %240 : <f32>
                %291 = ascvec.max %290, %287, %240 : <f32>
                %292 = ascvec.max %291, %288, %240 : <f32>
                %293 = ascvec.max %292, %289, %240 : <f32>
                %294 = ascvec.reduce_max_datablock %293, %240 : <f32>
                %result0, %result1 = ascvec.deintlv %294, %294 : <f32>, <f32> -> <f32>, <f32>
                %295 = ascvec.max %result0, %result1, %240 : <f32>
                %low, %high = ascvec.intlv %295, %295 : <f32>, <f32> -> <f32>, <f32>
                %296 = arith.muli %arg35, %c8_i32_12 overflow<nsw> : i32
                %297 = arith.index_cast %296 : i32 to index
                %298 = arith.addi %arg29, %297 : index
                ascvec.storealign %arg28[%298], %low, %241 : memref<?xf32, 1>, index, <f32>
              }
              ascvec.mem_bar vst_vld
              %c0_21 = arith.constant 0 : index
              %c1_22 = arith.constant 1 : index
              scf.for %arg35 = %c0_21 to %c1_22 step %c1_22 {
                %263 = ascvec.loadalign %arg28[%arg29] : memref<?xf32, 1>, index -> <f32>
                ascvec.storealign %arg30[%arg31], %263, %240 : memref<?xf32, 1>, index, <f32>
              }
              ascvec.mem_bar vst_vld
              scf.for %arg35 = %c0_i32_17 to %c32_i32_15 step %c8_i32_12  : i32 {
                %263 = arith.muli %arg35, %c2_i32 overflow<nsw> : i32
                %264 = arith.index_cast %263 : i32 to index
                %265 = arith.addi %arg31, %264 : index
                %266 = ascvec.loadalign_brc %arg30[%265] {brc_width = "e2b_b32"} : memref<?xf32, 1>, index -> <ui32>
                %267 = ascvec.reinterpret %266 : <ui32> -> <f32>
                %268 = arith.addi %263, %c8_i32_12 overflow<nsw> : i32
                %269 = arith.index_cast %268 : i32 to index
                %270 = arith.addi %arg31, %269 : index
                %271 = ascvec.loadalign_brc %arg30[%270] {brc_width = "e2b_b32"} : memref<?xf32, 1>, index -> <ui32>
                %272 = ascvec.reinterpret %271 : <ui32> -> <f32>
                %273 = arith.muli %arg35, %c16_i32 overflow<nsw> : i32
                %274 = arith.index_cast %273 : i32 to index
                %275 = arith.addi %274, %c57600 : index
                %276 = ascvec.loadalign %arg27[%275] : memref<?xf32, 1>, index -> <f32>
                %277 = arith.addi %273, %c64_i32 overflow<nsw> : i32
                %278 = arith.index_cast %277 : i32 to index
                %279 = arith.addi %278, %c57600 : index
                %280 = ascvec.loadalign %arg27[%279] : memref<?xf32, 1>, index -> <f32>
                %281 = arith.addi %273, %c512_i32 overflow<nsw> : i32
                %282 = arith.index_cast %281 : i32 to index
                %283 = arith.addi %282, %c57600 : index
                %284 = ascvec.loadalign %arg27[%283] : memref<?xf32, 1>, index -> <f32>
                %285 = arith.addi %273, %c576_i32 overflow<nsw> : i32
                %286 = arith.index_cast %285 : i32 to index
                %287 = arith.addi %286, %c57600 : index
                %288 = ascvec.loadalign %arg27[%287] : memref<?xf32, 1>, index -> <f32>
                %289 = ascvec.exp_sub %276, %267, %251 : <f32>, <f32> -> <f32>
                %290 = ascvec.exp_sub %280, %272, %251 : <f32>, <f32> -> <f32>
                %291 = ascvec.exp_sub %284, %267, %254 : <f32>, <f32> -> <f32>
                %292 = ascvec.exp_sub %288, %272, %254 : <f32>, <f32> -> <f32>
                %293 = ascvec.add %289, %291, %240 : <f32>
                %294 = ascvec.add %290, %292, %240 : <f32>
                %295 = ascvec.cast %289, %240 {reg_layout = 0 : i32} : <f32> -> <bf16>
                %296 = ascvec.cast %290, %240 {reg_layout = 0 : i32} : <f32> -> <bf16>
                %297 = ascvec.cast %291, %240 {reg_layout = 1 : i32} : <f32> -> <bf16>
                %298 = ascvec.cast %292, %240 {reg_layout = 1 : i32} : <f32> -> <bf16>
                %299 = ascvec.add %295, %297, %240 : <bf16>
                %300 = ascvec.add %296, %298, %240 : <bf16>
                %result0, %result1 = ascvec.deintlv %299, %300 : <bf16>, <bf16> -> <bf16>, <bf16>
                %301 = arith.addi %274, %c115200_10 : index
                ascvec.storealign %arg32[%301], %result0, %240 : memref<?xbf16, 1>, index, <bf16>
                %302 = arith.addi %273, %c1024_i32_13 overflow<nsw> : i32
                %303 = arith.index_cast %302 : i32 to index
                %304 = arith.addi %303, %c115200_10 : index
                ascvec.storealign %arg32[%304], %result1, %240 : memref<?xbf16, 1>, index, <bf16>
                %305 = arith.addi %303, %c57600 : index
                %306 = ascvec.loadalign %arg27[%305] : memref<?xf32, 1>, index -> <f32>
                %307 = arith.addi %273, %c1088_i32 overflow<nsw> : i32
                %308 = arith.index_cast %307 : i32 to index
                %309 = arith.addi %308, %c57600 : index
                %310 = ascvec.loadalign %arg27[%309] : memref<?xf32, 1>, index -> <f32>
                %311 = arith.addi %273, %c1536_i32 overflow<nsw> : i32
                %312 = arith.index_cast %311 : i32 to index
                %313 = arith.addi %312, %c57600 : index
                %314 = ascvec.loadalign %arg27[%313] : memref<?xf32, 1>, index -> <f32>
                %315 = arith.addi %273, %c1600_i32 overflow<nsw> : i32
                %316 = arith.index_cast %315 : i32 to index
                %317 = arith.addi %316, %c57600 : index
                %318 = ascvec.loadalign %arg27[%317] : memref<?xf32, 1>, index -> <f32>
                %319 = ascvec.exp_sub %306, %267, %257 : <f32>, <f32> -> <f32>
                %320 = ascvec.exp_sub %310, %272, %257 : <f32>, <f32> -> <f32>
                %321 = ascvec.exp_sub %314, %267, %260 : <f32>, <f32> -> <f32>
                %322 = ascvec.exp_sub %318, %272, %260 : <f32>, <f32> -> <f32>
                %323 = ascvec.add %293, %319, %240 : <f32>
                %324 = ascvec.add %323, %321, %240 : <f32>
                %325 = ascvec.add %294, %320, %240 : <f32>
                %326 = ascvec.add %325, %322, %240 : <f32>
                %327 = ascvec.cast %319, %240 {reg_layout = 0 : i32} : <f32> -> <bf16>
                %328 = ascvec.cast %320, %240 {reg_layout = 0 : i32} : <f32> -> <bf16>
                %329 = ascvec.cast %321, %240 {reg_layout = 1 : i32} : <f32> -> <bf16>
                %330 = ascvec.cast %322, %240 {reg_layout = 1 : i32} : <f32> -> <bf16>
                %331 = ascvec.add %327, %329, %240 : <bf16>
                %332 = ascvec.add %328, %330, %240 : <bf16>
                %result0_23, %result1_24 = ascvec.deintlv %331, %332 : <bf16>, <bf16> -> <bf16>, <bf16>
                %333 = arith.addi %273, %c2048_i32 overflow<nsw> : i32
                %334 = arith.index_cast %333 : i32 to index
                %335 = arith.addi %334, %c115200_10 : index
                ascvec.storealign %arg32[%335], %result0_23, %240 : memref<?xbf16, 1>, index, <bf16>
                %336 = arith.addi %273, %c3072_i32 overflow<nsw> : i32
                %337 = arith.index_cast %336 : i32 to index
                %338 = arith.addi %337, %c115200_10 : index
                ascvec.storealign %arg32[%338], %result1_24, %240 : memref<?xbf16, 1>, index, <bf16>
                %339 = ascvec.reduce_sum_datablock %324, %240 : <f32> -> <f32>
                %340 = ascvec.pair_reduce_sum %339, %240 : <f32>
                %341 = ascvec.reduce_sum_datablock %326, %240 : <f32> -> <f32>
                %342 = ascvec.pair_reduce_sum %341, %240 : <f32>
                %low, %high = ascvec.intlv %340, %340 : <f32>, <f32> -> <f32>, <f32>
                %low_25, %high_26 = ascvec.intlv %342, %342 : <f32>, <f32> -> <f32>, <f32>
                %343 = arith.addi %arg34, %264 : index
                ascvec.storealign %arg33[%343], %low, %241 : memref<?xf32, 1>, index, <f32>
                %344 = arith.addi %arg34, %269 : index
                ascvec.storealign %arg33[%344], %low_25, %241 : memref<?xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %216, 0
            ascvec.sync_notify {dst_pipe = 6 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
          } else {
            %210 = arith.remsi %171, %c2_i64 : i64
            %211 = arith.cmpi ne, %210, %c0_i64 : i64
            %212 = arith.cmpi slt, %210, %c0_i64 : i64
            %213 = arith.andi %211, %212 : i1
            %214 = arith.select %213, %c2_i64, %c0_i64 : i64
            %215 = arith.addi %210, %214 overflow<nsw> : i64
            %216 = arith.muli %215, %c512_i64 overflow<nsw> : i64
            %217 = arith.remsi %175, %c2_i64 : i64
            %218 = arith.cmpi ne, %217, %c0_i64 : i64
            %219 = arith.cmpi slt, %217, %c0_i64 : i64
            %220 = arith.andi %218, %219 : i1
            %221 = arith.select %220, %c2_i64, %c0_i64 : i64
            %222 = arith.addi %217, %221 overflow<nsw> : i64
            %223 = arith.muli %222, %c256_i64 overflow<nsw> : i64
            %224 = arith.addi %216, %223 overflow<nsw> : i64
            ascvec.sync_intra_wait v, %c0_i64
            %225 = arith.addi %215, %c2_i64 overflow<nsw> : i64
            ascvec.get_buf v, %225, 0
            %226 = arith.remsi %171, %c4_i64 : i64
            %227 = arith.cmpi ne, %226, %c0_i64 : i64
            %228 = arith.cmpi slt, %226, %c0_i64 : i64
            %229 = arith.andi %227, %228 : i1
            %230 = arith.select %229, %c4_i64, %c0_i64 : i64
            %231 = arith.addi %226, %230 overflow<nsw> : i64
            %232 = arith.muli %231, %c640_i64 overflow<nsw> : i64
            %233 = arith.muli %175, %c64_i64 : i64
            %234 = arith.addi %232, %233 : i64
            %235 = arith.index_cast %234 : i64 to index
            %236 = arith.muli %215, %c2_i64 overflow<nsw> : i64
            %237 = arith.addi %236, %222 overflow<nsw> : i64
            %238 = arith.index_cast %237 : i64 to index
            %239 = arith.muli %215, %c2048_i64 overflow<nsw> : i64
            %240 = arith.index_cast %239 : i64 to index
            %241 = arith.addi %240, %c1922048 : index
            %242 = arith.divui %241, %c32 : index
            %243 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xf32, 1>
            %244 = arith.addi %240, %c1909760 : index
            %245 = arith.divui %244, %c32 : index
            %246 = arith.addi %240, %c1913856 : index
            %247 = arith.divui %246, %c32 : index
            %248 = arith.muli %224, %c8_i64 overflow<nsw> : i64
            %249 = arith.index_cast %248 : i64 to index
            %250 = arith.addi %249, %c1926144 : index
            %251 = arith.divui %250, %c32 : index
            %252 = arith.addi %240, %c1917952 : index
            %253 = arith.divui %252, %c32 : index
            ascvec.vf(%1, %235, %243, %242, %243, %245, %243, %243, %247, %243, %251, %243, %238, %243, %253, %6 : memref<253952xui8, 1>, index, memref<?xf32, 1>, index, memref<?xf32, 1>, index, memref<?xf32, 1>, memref<?xf32, 1>, index, memref<?xf32, 1>, index, memref<?xf32, 1>, index, memref<?xf32, 1>, index, memref<?xbf16, 1>) {
            ^bb0(%arg21: memref<253952xui8, 1>, %arg22: index, %arg23: memref<?xf32, 1>, %arg24: index, %arg25: memref<?xf32, 1>, %arg26: index, %arg27: memref<?xf32, 1>, %arg28: memref<?xf32, 1>, %arg29: index, %arg30: memref<?xf32, 1>, %arg31: index, %arg32: memref<?xf32, 1>, %arg33: index, %arg34: memref<?xf32, 1>, %arg35: index, %arg36: memref<?xbf16, 1>):
              %c3072_i32 = arith.constant 3072 : i32
              %c1600_i32 = arith.constant 1600 : i32
              %c1088_i32 = arith.constant 1088 : i32
              %c2048_i32 = arith.constant 2048 : i32
              %c115200_10 = arith.constant 115200 : index
              %c576_i32 = arith.constant 576 : i32
              %c2_i32 = arith.constant 2 : i32
              %c63008_11 = arith.constant 63008 : index
              %cst_12 = arith.constant 6.000000e+00 : f32
              %c1_i32_13 = arith.constant 1 : i32
              %c1536_i32 = arith.constant 1536 : i32
              %c1024_i32_14 = arith.constant 1024 : i32
              %c512_i32 = arith.constant 512 : i32
              %c57600 = arith.constant 57600 : index
              %c64_i32 = arith.constant 64 : i32
              %c4_i32_15 = arith.constant 4 : i32
              %c8_i32_16 = arith.constant 8 : i32
              %c48_i32 = arith.constant 48 : i32
              %c32_i32_17 = arith.constant 32 : i32
              %c16_i32 = arith.constant 16 : i32
              %c15_i32 = arith.constant 15 : i32
              %c0_i32_18 = arith.constant 0 : i32
              %c249472 = arith.constant 249472 : index
              %cst_19 = arith.constant -3.40282347E+38 : f32
              %255 = ascvec.create_mask elem_bits 16 : !ascvec.mask
              %256 = ascvec.create_mask pat_mode "PAT_VL8" : !ascvec.mask
              %257 = ascvec.duplicate_scalar_nomask %cst_19 : f32 -> <f32>
              %258 = arith.addi %arg22, %c249472 : index
              %259 = ascvec.loadalign_unpack %arg21[%258] {mode = "b8_to_b32"} : memref<253952xui8, 1>, index -> <ui32>
              %260 = ascvec.arange %c0_i32_18 : i32 -> <ui32>
              %261 = ascvec.duplicate_scalar_nomask %c15_i32 : i32 -> <ui32>
              %262 = ascvec.and %260, %261, %255 : <ui32>, <ui32> -> <ui32>
              %263 = ascvec.gather_reg %259, %262 : <ui32>, <ui32> -> <ui32>
              %264 = ascvec.cmp_gt_scalar %263, %c0_i32_18, %255 : <ui32>, i32 -> !ascvec.mask
              %265 = ascvec.add_scalar %262, %c16_i32, %255 : <ui32>, i32
              %266 = ascvec.gather_reg %259, %265 : <ui32>, <ui32> -> <ui32>
              %267 = ascvec.cmp_gt_scalar %266, %c0_i32_18, %255 : <ui32>, i32 -> !ascvec.mask
              %268 = ascvec.add_scalar %262, %c32_i32_17, %255 : <ui32>, i32
              %269 = ascvec.gather_reg %259, %268 : <ui32>, <ui32> -> <ui32>
              %270 = ascvec.cmp_gt_scalar %269, %c0_i32_18, %255 : <ui32>, i32 -> !ascvec.mask
              %271 = ascvec.add_scalar %262, %c48_i32, %255 : <ui32>, i32
              %272 = ascvec.gather_reg %259, %271 : <ui32>, <ui32> -> <ui32>
              %273 = ascvec.cmp_gt_scalar %272, %c0_i32_18, %255 : <ui32>, i32 -> !ascvec.mask
              %274 = ascvec.loadalign %arg23[%arg24] : memref<?xf32, 1>, index -> <f32>
              %275 = ascvec.shr %260, 4, %255 : <ui32> -> <ui32>
              scf.for %arg37 = %c0_i32_18 to %c8_i32_16 step %c1_i32_13  : i32 {
                %286 = arith.muli %arg37, %c8_i32_16 overflow<nsw> : i32
                %287 = arith.index_cast %286 : i32 to index
                %288 = arith.addi %arg26, %287 : index
                %289 = ascvec.loadalign_brc %arg25[%288] {brc_width = "e2b_b32"} : memref<?xf32, 1>, index -> <ui32>
                %290 = ascvec.reinterpret %289 : <ui32> -> <f32>
                %291 = arith.muli %arg37, %c4_i32_15 overflow<nsw> : i32
                %292 = ascvec.add_scalar %275, %291, %255 : <ui32>, i32
                %293 = ascvec.gather_reg %274, %292 : <f32>, <ui32> -> <f32>
                %294 = arith.muli %arg37, %c64_i32 overflow<nsw> : i32
                %295 = arith.index_cast %294 : i32 to index
                %296 = arith.addi %295, %c57600 : index
                %297 = ascvec.loadalign %arg27[%296] : memref<?xf32, 1>, index -> <f32>
                %298 = ascvec.mul %297, %293, %255 : <f32>
                %299 = arith.addi %294, %c512_i32 overflow<nsw> : i32
                %300 = arith.index_cast %299 : i32 to index
                %301 = arith.addi %300, %c57600 : index
                %302 = ascvec.loadalign %arg27[%301] : memref<?xf32, 1>, index -> <f32>
                %303 = ascvec.mul %302, %293, %255 : <f32>
                %304 = arith.addi %294, %c1024_i32_14 overflow<nsw> : i32
                %305 = arith.index_cast %304 : i32 to index
                %306 = arith.addi %305, %c57600 : index
                %307 = ascvec.loadalign %arg27[%306] : memref<?xf32, 1>, index -> <f32>
                %308 = ascvec.mul %307, %293, %255 : <f32>
                %309 = arith.addi %294, %c1536_i32 overflow<nsw> : i32
                %310 = arith.index_cast %309 : i32 to index
                %311 = arith.addi %310, %c57600 : index
                %312 = ascvec.loadalign %arg27[%311] : memref<?xf32, 1>, index -> <f32>
                %313 = ascvec.mul %312, %293, %255 : <f32>
                ascvec.storealign %arg27[%296], %298, %255 : memref<?xf32, 1>, index, <f32>
                ascvec.storealign %arg27[%301], %303, %255 : memref<?xf32, 1>, index, <f32>
                ascvec.storealign %arg27[%306], %308, %255 : memref<?xf32, 1>, index, <f32>
                ascvec.storealign %arg27[%311], %313, %255 : memref<?xf32, 1>, index, <f32>
                %314 = ascvec.select %298, %257, %264 : <f32>
                %315 = ascvec.select %303, %257, %267 : <f32>
                %316 = ascvec.select %308, %257, %270 : <f32>
                %317 = ascvec.select %313, %257, %273 : <f32>
                %318 = ascvec.max %290, %314, %255 : <f32>
                %319 = ascvec.max %318, %315, %255 : <f32>
                %320 = ascvec.max %319, %316, %255 : <f32>
                %321 = ascvec.max %320, %317, %255 : <f32>
                %322 = ascvec.reduce_max_datablock %321, %255 : <f32>
                %result0, %result1 = ascvec.deintlv %322, %322 : <f32>, <f32> -> <f32>, <f32>
                %323 = ascvec.max %result0, %result1, %255 : <f32>
                %low, %high = ascvec.intlv %323, %323 : <f32>, <f32> -> <f32>, <f32>
                ascvec.storealign %arg25[%288], %low, %256 : memref<?xf32, 1>, index, <f32>
              }
              ascvec.mem_bar vst_vld
              %276 = ascvec.loadalign %arg25[%arg26] : memref<?xf32, 1>, index -> <f32>
              %277 = ascvec.loadalign %arg28[%arg29] : memref<?xf32, 1>, index -> <f32>
              %278 = ascvec.sub %276, %277, %255 : <f32>
              %279 = ascvec.reduce_max %278, %255 : <f32>
              %280 = ascvec.duplicate %279, %255 : <f32> -> <f32>
              %281 = ascvec.cmp_gt_scalar %280, %cst_12, %255 : <f32>, f32 -> !ascvec.mask
              %282 = ascvec.select %276, %277, %281 : <f32>
              ascvec.storealign %arg28[%arg29], %282, %255 : memref<?xf32, 1>, index, <f32>
              %283 = ascvec.exp_sub %277, %282, %255 : <f32>, <f32> -> <f32>
              ascvec.storealign %arg30[%arg31], %283, %255 : memref<?xf32, 1>, index, <f32>
              %284 = arith.addi %arg33, %c63008_11 : index
              ascvec.storealign_1st %arg32[%284], %280 : memref<?xf32, 1>, index, <f32>
              %285 = ascvec.loadalign %arg34[%arg35] : memref<?xf32, 1>, index -> <f32>
              ascvec.mem_bar vst_vld
              scf.for %arg37 = %c0_i32_18 to %c32_i32_17 step %c8_i32_16  : i32 {
                %286 = arith.muli %arg37, %c2_i32 overflow<nsw> : i32
                %287 = arith.index_cast %286 : i32 to index
                %288 = arith.addi %arg29, %287 : index
                %289 = ascvec.loadalign_brc %arg28[%288] {brc_width = "e2b_b32"} : memref<?xf32, 1>, index -> <ui32>
                %290 = ascvec.reinterpret %289 : <ui32> -> <f32>
                %291 = arith.addi %286, %c8_i32_16 overflow<nsw> : i32
                %292 = arith.index_cast %291 : i32 to index
                %293 = arith.addi %arg29, %292 : index
                %294 = ascvec.loadalign_brc %arg28[%293] {brc_width = "e2b_b32"} : memref<?xf32, 1>, index -> <ui32>
                %295 = ascvec.reinterpret %294 : <ui32> -> <f32>
                %296 = arith.muli %arg37, %c16_i32 overflow<nsw> : i32
                %297 = arith.index_cast %296 : i32 to index
                %298 = arith.addi %297, %c57600 : index
                %299 = ascvec.loadalign %arg27[%298] : memref<?xf32, 1>, index -> <f32>
                %300 = arith.addi %296, %c64_i32 overflow<nsw> : i32
                %301 = arith.index_cast %300 : i32 to index
                %302 = arith.addi %301, %c57600 : index
                %303 = ascvec.loadalign %arg27[%302] : memref<?xf32, 1>, index -> <f32>
                %304 = arith.addi %296, %c512_i32 overflow<nsw> : i32
                %305 = arith.index_cast %304 : i32 to index
                %306 = arith.addi %305, %c57600 : index
                %307 = ascvec.loadalign %arg27[%306] : memref<?xf32, 1>, index -> <f32>
                %308 = arith.addi %296, %c576_i32 overflow<nsw> : i32
                %309 = arith.index_cast %308 : i32 to index
                %310 = arith.addi %309, %c57600 : index
                %311 = ascvec.loadalign %arg27[%310] : memref<?xf32, 1>, index -> <f32>
                %312 = ascvec.exp_sub %299, %290, %264 : <f32>, <f32> -> <f32>
                %313 = ascvec.exp_sub %303, %295, %264 : <f32>, <f32> -> <f32>
                %314 = ascvec.exp_sub %307, %290, %267 : <f32>, <f32> -> <f32>
                %315 = ascvec.exp_sub %311, %295, %267 : <f32>, <f32> -> <f32>
                %316 = ascvec.add %312, %314, %255 : <f32>
                %317 = ascvec.add %313, %315, %255 : <f32>
                %318 = ascvec.cast %312, %255 {reg_layout = 0 : i32} : <f32> -> <bf16>
                %319 = ascvec.cast %313, %255 {reg_layout = 0 : i32} : <f32> -> <bf16>
                %320 = ascvec.cast %314, %255 {reg_layout = 1 : i32} : <f32> -> <bf16>
                %321 = ascvec.cast %315, %255 {reg_layout = 1 : i32} : <f32> -> <bf16>
                %322 = ascvec.add %318, %320, %255 : <bf16>
                %323 = ascvec.add %319, %321, %255 : <bf16>
                %result0, %result1 = ascvec.deintlv %322, %323 : <bf16>, <bf16> -> <bf16>, <bf16>
                %324 = arith.addi %297, %c115200_10 : index
                ascvec.storealign %arg36[%324], %result0, %255 : memref<?xbf16, 1>, index, <bf16>
                %325 = arith.addi %296, %c1024_i32_14 overflow<nsw> : i32
                %326 = arith.index_cast %325 : i32 to index
                %327 = arith.addi %326, %c115200_10 : index
                ascvec.storealign %arg36[%327], %result1, %255 : memref<?xbf16, 1>, index, <bf16>
                %328 = arith.addi %326, %c57600 : index
                %329 = ascvec.loadalign %arg27[%328] : memref<?xf32, 1>, index -> <f32>
                %330 = arith.addi %296, %c1088_i32 overflow<nsw> : i32
                %331 = arith.index_cast %330 : i32 to index
                %332 = arith.addi %331, %c57600 : index
                %333 = ascvec.loadalign %arg27[%332] : memref<?xf32, 1>, index -> <f32>
                %334 = arith.addi %296, %c1536_i32 overflow<nsw> : i32
                %335 = arith.index_cast %334 : i32 to index
                %336 = arith.addi %335, %c57600 : index
                %337 = ascvec.loadalign %arg27[%336] : memref<?xf32, 1>, index -> <f32>
                %338 = arith.addi %296, %c1600_i32 overflow<nsw> : i32
                %339 = arith.index_cast %338 : i32 to index
                %340 = arith.addi %339, %c57600 : index
                %341 = ascvec.loadalign %arg27[%340] : memref<?xf32, 1>, index -> <f32>
                %342 = ascvec.exp_sub %329, %290, %270 : <f32>, <f32> -> <f32>
                %343 = ascvec.exp_sub %333, %295, %270 : <f32>, <f32> -> <f32>
                %344 = ascvec.exp_sub %337, %290, %273 : <f32>, <f32> -> <f32>
                %345 = ascvec.exp_sub %341, %295, %273 : <f32>, <f32> -> <f32>
                %346 = ascvec.add %316, %342, %255 : <f32>
                %347 = ascvec.add %346, %344, %255 : <f32>
                %348 = ascvec.add %317, %343, %255 : <f32>
                %349 = ascvec.add %348, %345, %255 : <f32>
                %350 = ascvec.cast %342, %255 {reg_layout = 0 : i32} : <f32> -> <bf16>
                %351 = ascvec.cast %343, %255 {reg_layout = 0 : i32} : <f32> -> <bf16>
                %352 = ascvec.cast %344, %255 {reg_layout = 1 : i32} : <f32> -> <bf16>
                %353 = ascvec.cast %345, %255 {reg_layout = 1 : i32} : <f32> -> <bf16>
                %354 = ascvec.add %350, %352, %255 : <bf16>
                %355 = ascvec.add %351, %353, %255 : <bf16>
                %result0_22, %result1_23 = ascvec.deintlv %354, %355 : <bf16>, <bf16> -> <bf16>, <bf16>
                %356 = arith.addi %296, %c2048_i32 overflow<nsw> : i32
                %357 = arith.index_cast %356 : i32 to index
                %358 = arith.addi %357, %c115200_10 : index
                ascvec.storealign %arg36[%358], %result0_22, %255 : memref<?xbf16, 1>, index, <bf16>
                %359 = arith.addi %296, %c3072_i32 overflow<nsw> : i32
                %360 = arith.index_cast %359 : i32 to index
                %361 = arith.addi %360, %c115200_10 : index
                ascvec.storealign %arg36[%361], %result1_23, %255 : memref<?xbf16, 1>, index, <bf16>
                %362 = ascvec.reduce_sum_datablock %347, %255 : <f32> -> <f32>
                %363 = ascvec.pair_reduce_sum %362, %255 : <f32>
                %364 = ascvec.reduce_sum_datablock %349, %255 : <f32> -> <f32>
                %365 = ascvec.pair_reduce_sum %364, %255 : <f32>
                %low, %high = ascvec.intlv %363, %363 : <f32>, <f32> -> <f32>, <f32>
                %low_24, %high_25 = ascvec.intlv %365, %365 : <f32>, <f32> -> <f32>, <f32>
                %366 = arith.addi %arg35, %287 : index
                ascvec.storealign %arg34[%366], %low, %256 : memref<?xf32, 1>, index, <f32>
                %367 = arith.addi %arg35, %292 : index
                ascvec.storealign %arg34[%367], %low_24, %256 : memref<?xf32, 1>, index, <f32>
              }
              ascvec.mem_bar vst_vld
              %c0_20 = arith.constant 0 : index
              %c1_21 = arith.constant 1 : index
              scf.for %arg37 = %c0_20 to %c1_21 step %c1_21 {
                %286 = ascvec.loadalign %arg34[%arg35] : memref<?xf32, 1>, index -> <f32>
                %287 = ascvec.madd_two_address %285, %283, %286, %255 : <f32>
                ascvec.storealign %arg34[%arg35], %287, %255 : memref<?xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %225, 0
            ascvec.sync_notify {dst_pipe = 6 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
            %254 = arith.cmpi ne, %175, %c0_i64 : i64
            scf.if %254 {
              ascvec.sync_notify {dst_pipe = 1 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
            }
          }
          ascvec.sync_wait {dst_pipe = 6 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
          ascvec.sync_intra_wait mte3, %c1_i64
          %203 = ascscalar.bitcast_memref %2 byte_offset = %c0_i64 : memref<524288xui8, 2> -> memref<?xbf16, 2>
          %204 = arith.muli %0, %c32_i64 : i64
          %205 = arith.index_cast %204 : i64 to index
          %206 = arith.muli %205, %c16 : index
          %207 = arith.addi %206, %c129024 : index
          ascvec.copy_ub2l1_strided %203[%207], %6[%c115200], %c4_i32, %c32_i32, %c32_i32, %c32_i32 : memref<?xbf16, 2>, memref<?xbf16, 1>
          ascvec.sync_intra_arrive mte3, %c1_i64
          ascvec.sync_intra_arrive mte3, %c0_i64
          %208 = arith.addi %175, %c1_i64 : i64
          %209 = arith.cmpi eq, %208, %177 : i64
          scf.if %209 {
            %210 = arith.remsi %171, %c2_i64 : i64
            %211 = arith.cmpi ne, %210, %c0_i64 : i64
            %212 = arith.cmpi slt, %210, %c0_i64 : i64
            %213 = arith.andi %211, %212 : i1
            %214 = arith.select %213, %c2_i64, %c0_i64 : i64
            %215 = arith.addi %210, %214 overflow<nsw> : i64
            %216 = arith.addi %215, %c2_i64 overflow<nsw> : i64
            ascvec.get_buf v, %216, 0
            %217 = arith.muli %215, %c2048_i64 overflow<nsw> : i64
            %218 = arith.index_cast %217 : i64 to index
            %219 = arith.addi %218, %c1909760 : index
            %220 = arith.divui %219, %c32 : index
            %221 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xf32, 1>
            %222 = arith.addi %218, %c1913856 : index
            %223 = arith.divui %222, %c32 : index
            %224 = arith.addi %218, %c1917952 : index
            %225 = arith.divui %224, %c32 : index
            %226 = arith.addi %218, %c1922048 : index
            %227 = arith.divui %226, %c32 : index
            ascvec.vf(%221, %220, %221, %223, %221, %225, %221, %221, %227 : memref<?xf32, 1>, index, memref<?xf32, 1>, index, memref<?xf32, 1>, index, memref<?xf32, 1>, memref<?xf32, 1>, index) {
            ^bb0(%arg21: memref<?xf32, 1>, %arg22: index, %arg23: memref<?xf32, 1>, %arg24: index, %arg25: memref<?xf32, 1>, %arg26: index, %arg27: memref<?xf32, 1>, %arg28: memref<?xf32, 1>, %arg29: index):
              %cst_10 = arith.constant 0.000000e+00 : f32
              %c59648 = arith.constant 59648 : index
              %c2139095040_i32 = arith.constant 2139095040 : i32
              %c-8388608_i32 = arith.constant -8388608 : i32
              %cst_11 = arith.constant 1.000000e+00 : f32
              %cst_12 = arith.constant -3.40282347E+38 : f32
              %c0_13 = arith.constant 0 : index
              %c1_14 = arith.constant 1 : index
              scf.for %arg30 = %c0_13 to %c1_14 step %c1_14 {
                %244 = ascvec.create_mask : !ascvec.mask
                %245 = ascvec.create_mask pat_mode "PAT_VL32" : !ascvec.mask
                %246 = ascvec.loadalign %arg21[%arg22] : memref<?xf32, 1>, index -> <f32>
                %result0, %result1 = ascvec.deintlv %246, %246 : <f32>, <f32> -> <f32>, <f32>
                %247 = ascvec.loadalign %arg23[%arg24] : memref<?xf32, 1>, index -> <f32>
                %result0_15, %result1_16 = ascvec.deintlv %247, %247 : <f32>, <f32> -> <f32>, <f32>
                %248 = ascvec.loadalign %arg25[%arg26] : memref<?xf32, 1>, index -> <f32>
                %result0_17, %result1_18 = ascvec.deintlv %248, %248 : <f32>, <f32> -> <f32>, <f32>
                %249 = ascvec.cmp_eq_scalar %result0, %cst_12, %244 : <f32>, f32 -> !ascvec.mask
                %250 = ascvec.mul_scalar %result0_15, %cst_11, %244 : <f32>, f32
                %251 = ascvec.duplicate_scalar_nomask %c-8388608_i32 : i32 -> <ui32>
                %252 = ascvec.reinterpret %251 : <ui32> -> <f32>
                %253 = ascvec.duplicate_scalar_nomask %c2139095040_i32 : i32 -> <ui32>
                %254 = ascvec.reinterpret %253 : <ui32> -> <f32>
                %255 = ascvec.mul_scalar %result0, %cst_11, %244 : <f32>, f32
                %256 = ascvec.select %252, %255, %249 : <f32>
                ascvec.storealign %arg21[%arg22], %256, %245 : memref<?xf32, 1>, index, <f32>
                %257 = ascvec.ln %result0_17, %244 : <f32>
                %258 = ascvec.add %257, %250, %244 : <f32>
                %259 = ascvec.select %254, %258, %249 : <f32>
                ascvec.storealign %arg25[%arg26], %259, %245 : memref<?xf32, 1>, index, <f32>
                %260 = ascvec.loadalign %arg27[%c59648] : memref<?xf32, 1>, index -> <f32>
                %261 = ascvec.exp_sub %260, %250, %244 : <f32>, <f32> -> <f32>
                %262 = ascvec.add %result0_17, %261, %244 : <f32>
                %263 = ascvec.duplicate_scalar_nomask %cst_11 : f32 -> <f32>
                %264 = ascvec.div %263, %262, %244 : <f32>
                %265 = ascvec.duplicate_scalar_nomask %cst_10 : f32 -> <f32>
                %266 = ascvec.select %265, %264, %249 : <f32>
                ascvec.storealign %arg28[%arg29], %266, %245 : memref<?xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %216, 0
            ascvec.get_buf mte3, %216, 0
            %228 = arith.muli %173, %c2_i64 : i64
            %229 = arith.addi %228, %0 : i64
            %230 = arith.muli %229, %c1024_i64 : i64
            %231 = ascscalar.memref_address %alloc_8 : memref<256xf32>
            %232 = arith.divui %230, %c8_i64 : i64
            %233 = arith.addi %231, %232 : i64
            %234 = ascscalar.bind_gm_ptr %233 : memref<1xf32>
            %235 = arith.addi %217, %c1909760_i64 overflow<nsw> : i64
            %236 = arith.divui %235, %c8_i64 : i64
            %237 = ascscalar.bind_local_ptr %236 : memref<1xf32, 1>
            ascvec.copy_ub2gm %234[%c0], %237[%c0], %c1_i32, %c128_i32, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
            %238 = ascscalar.memref_address %alloc_9 : memref<256xf32>
            %239 = arith.addi %238, %232 : i64
            %240 = ascscalar.bind_gm_ptr %239 : memref<1xf32>
            %241 = arith.addi %217, %c1917952_i64 overflow<nsw> : i64
            %242 = arith.divui %241, %c8_i64 : i64
            %243 = ascscalar.bind_local_ptr %242 : memref<1xf32, 1>
            ascvec.copy_ub2gm %240[%c0], %243[%c0], %c1_i32, %c128_i32, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
            ascvec.release_buf mte3, %216, 0
          }
        }
        scf.if %110 {
          %202 = arith.cmpi eq, %arg19, %c0_i64 : i64
          scf.if %202 {
            ascvec.get_buf v, %c0_i64, 0
            ascvec.get_buf v, %c1_i64, 0
            %203 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xi32, 1>
            %204 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xui32, 1>
            ascvec.vf(%203, %arg17, %arg16, %1, %204, %204 : memref<?xi32, 1>, i64, i64, memref<253952xui8, 1>, memref<?xui32, 1>, memref<?xui32, 1>) {
            ^bb0(%arg21: memref<?xi32, 1>, %arg22: i64, %arg23: i64, %arg24: memref<253952xui8, 1>, %arg25: memref<?xui32, 1>, %arg26: memref<?xui32, 1>):
              %c61792 = arith.constant 61792 : index
              %c61152 = arith.constant 61152 : index
              %c60512 = arith.constant 60512 : index
              %c64_i32 = arith.constant 64 : i32
              %c61728 = arith.constant 61728 : index
              %c61088 = arith.constant 61088 : index
              %c64_i64_10 = arith.constant 64 : i64
              %c249472 = arith.constant 249472 : index
              %c640_i64_11 = arith.constant 640 : i64
              %c0_i64_12 = arith.constant 0 : i64
              %c4_i64_13 = arith.constant 4 : i64
              %c0_i32_14 = arith.constant 0 : i32
              %c1_i32_15 = arith.constant 1 : i32
              %c60448 = arith.constant 60448 : index
              %c1024_i32_16 = arith.constant 1024 : i32
              %c32_i32_17 = arith.constant 32 : i32
              %205 = ascvec.create_mask : !ascvec.mask
              %206 = ascvec.duplicate_scalar_nomask %c32_i32_17 : i32 -> <ui32>
              %207 = ascvec.duplicate_scalar_nomask %c1024_i32_16 : i32 -> <ui32>
              %208 = ascvec.loadalign %arg21[%c60448] : memref<?xi32, 1>, index -> <i32>
              %209 = ascvec.reinterpret %208 : <i32> -> <ui32>
              %210 = ascvec.loadalign %arg21[%c60512] : memref<?xi32, 1>, index -> <i32>
              %211 = ascvec.reinterpret %210 : <i32> -> <ui32>
              %212 = ascvec.arange %c0_i32_14 : i32 -> <i32>
              %213 = arith.trunci %arg22 : i64 to i32
              %214 = ascvec.cmp_lt_scalar %212, %213, %205 : <i32>, i32 -> !ascvec.mask
              %215 = ascvec.arange %c64_i32 : i32 -> <i32>
              %216 = ascvec.cmp_lt_scalar %215, %213, %205 : <i32>, i32 -> !ascvec.mask
              %217 = ascvec.select %209, %206, %214 : <ui32>
              %218 = ascvec.select %211, %206, %216 : <ui32>
              %result0, %result1 = ascvec.deintlv %217, %218 : <ui32>, <ui32> -> <ui32>, <ui32>
              %219 = ascvec.min %result0, %result1, %205 : <ui32>
              %220 = ascvec.max %result0, %result1, %205 : <ui32>
              %221 = ascvec.min %219, %206, %205 : <ui32>
              %222 = ascvec.duplicate_scalar_nomask %c1_i32_15 : i32 -> <ui32>
              %223 = ascvec.duplicate_scalar_nomask %c0_i32_14 : i32 -> <ui32>
              %224 = ascvec.cmp_lt_scalar %221, %c32_i32_17, %205 : <ui32>, i32 -> !ascvec.mask
              %225 = ascvec.select %222, %223, %224 : <ui32>
              %226 = ascvec.cmp_lt_scalar %220, %c32_i32_17, %205 : <ui32>, i32 -> !ascvec.mask
              %227 = ascvec.select %222, %223, %226 : <ui32>
              %low, %high = ascvec.intlv %225, %227 : <ui32>, <ui32> -> <ui32>, <ui32>
              %228 = arith.remsi %arg23, %c4_i64_13 : i64
              %229 = arith.cmpi ne, %228, %c0_i64_12 : i64
              %230 = arith.cmpi slt, %228, %c0_i64_12 : i64
              %231 = arith.andi %229, %230 : i1
              %232 = arith.select %231, %c4_i64_13, %c0_i64_12 : i64
              %233 = arith.addi %228, %232 overflow<nsw> : i64
              %234 = arith.muli %233, %c640_i64_11 overflow<nsw> : i64
              %235 = arith.index_cast %234 : i64 to index
              %236 = arith.addi %235, %c249472 : index
              %c0_18 = arith.constant 0 : index
              %c1_19 = arith.constant 1 : index
              scf.for %arg27 = %c0_18 to %c1_19 step %c1_19 {
                ascvec.storealign_pack %arg24[%236], %low, %205 {mode = "b32_to_b8"} : memref<253952xui8, 1>, index, <ui32>
              }
              %237 = arith.addi %234, %c64_i64_10 overflow<nsw> : i64
              %238 = arith.index_cast %237 : i64 to index
              %239 = arith.addi %238, %c249472 : index
              %c0_20 = arith.constant 0 : index
              %c1_21 = arith.constant 1 : index
              scf.for %arg27 = %c0_20 to %c1_21 step %c1_21 {
                ascvec.storealign_pack %arg24[%239], %high, %205 {mode = "b32_to_b8"} : memref<253952xui8, 1>, index, <ui32>
                %240 = ascvec.select %220, %221, %226 : <ui32>
                %241 = ascvec.sub %240, %221, %205 : <ui32>
                %dst0, %dst1 = ascvec.mull %221, %207, %205 : <ui32>, <ui32> -> <ui32>, <ui32>
                %dst0_22, %dst1_23 = ascvec.mull %241, %207, %205 : <ui32>, <ui32> -> <ui32>, <ui32>
                %low_24, %high_25 = ascvec.intlv %dst0, %dst1 : <ui32>, <ui32> -> <ui32>, <ui32>
                %low_26, %high_27 = ascvec.intlv %dst0_22, %dst1_23 : <ui32>, <ui32> -> <ui32>, <ui32>
                ascvec.storealign %arg25[%c61088], %low_24, %205 : memref<?xui32, 1>, index, <ui32>
                ascvec.storealign %arg25[%c61152], %high_25, %205 : memref<?xui32, 1>, index, <ui32>
                ascvec.storealign %arg26[%c61728], %low_26, %205 : memref<?xui32, 1>, index, <ui32>
                ascvec.storealign %arg26[%c61792], %high_27, %205 : memref<?xui32, 1>, index, <ui32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c1_i64, 0
            ascvec.release_buf v, %c0_i64, 0
          }
        }
        %179 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %180 = arith.addi %179, %c1_i64 : i64
        %181 = arith.cmpi sge, %180, %c6_i64 : i64
        %182 = arith.extui %181 : i1 to i64
        %183 = arith.muli %182, %c6_i64 : i64
        %184 = arith.subi %180, %183 : i64
        %185 = cannir.datastruct_get %8["active"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %186 = arith.index_cast %184 : i64 to index
        %187 = cannir.array_load %185[%186] : <6xi64>
        %188 = cannir.datastruct_get %8["job"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %189 = cannir.array_load %188[%186] : <6xi64>
        %190 = cannir.datastruct_get %8["query"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %191 = cannir.array_load %190[%186] : <6xi64>
        %192 = cannir.datastruct_get %8["tile"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %193 = cannir.array_load %192[%186] : <6xi64>
        %194 = cannir.datastruct_get %8["blocks"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %195 = cannir.array_load %194[%186] : <6xi64>
        %196 = arith.cmpi ne, %187, %c0_i64 : i64
        scf.if %196 {
          %202 = arith.cmpi ne, %193, %c0_i64 : i64
          %203:2 = scf.if %202 -> (i32, i1) {
            ascvec.sync_intra_wait s, %c2_i64
            %224 = arith.remsi %189, %c8_i64 : i64
            %225 = arith.cmpi ne, %224, %c0_i64 : i64
            %226 = arith.cmpi slt, %224, %c0_i64 : i64
            %227 = arith.andi %225, %226 : i1
            %228 = arith.select %227, %c8_i64, %c0_i64 : i64
            %229 = arith.addi %224, %228 overflow<nsw> : i64
            %230 = arith.remsi %193, %c8_i64 : i64
            %231 = arith.cmpi ne, %230, %c0_i64 : i64
            %232 = arith.cmpi slt, %230, %c0_i64 : i64
            %233 = arith.andi %231, %232 : i1
            %234 = arith.select %233, %c8_i64, %c0_i64 : i64
            %235 = arith.addi %230, %234 overflow<nsw> : i64
            %236 = ascscalar.bitcast_memref %5 byte_offset = %c0_i64 : memref<3072xui8, 9> -> memref<?xui32, 9>
            %237 = arith.index_cast %229 : i64 to index
            %238 = arith.index_cast %235 : i64 to index
            %239 = arith.muli %237, %c16 : index
            %240 = arith.muli %238, %c2 : index
            %241 = arith.addi %239, %240 : index
            %242 = ascvec.ssbuf_load %236[%241] : memref<?xui32, 9> -> ui32
            %243 = cannir.scalar_storage_cast %242 : ui32 -> i32
            %244 = arith.addi %241, %c1 : index
            %245 = ascvec.ssbuf_load %236[%244] : memref<?xui32, 9> -> ui32
            %246 = cannir.scalar_storage_cast %245 : ui32 -> i32
            %247 = arith.addi %243, %246 : i32
            %248 = arith.cmpi ne, %247, %c0_i32 : i32
            %249 = arith.addi %193, %c1_i64 : i64
            %250 = arith.cmpi eq, %249, %195 : i64
            %251 = arith.select %248, %true, %250 : i1
            scf.yield %247, %251 : i32, i1
          } else {
            scf.yield %c0_i32, %false : i32, i1
          }
          %204 = scf.if %203#1 -> (i32) {
            %224 = arith.remsi %189, %c2_i64 : i64
            %225 = arith.cmpi ne, %224, %c0_i64 : i64
            %226 = arith.cmpi slt, %224, %c0_i64 : i64
            %227 = arith.andi %225, %226 : i1
            %228 = arith.select %227, %c2_i64, %c0_i64 : i64
            %229 = arith.addi %224, %228 overflow<nsw> : i64
            %230 = arith.addi %229, %c2_i64 overflow<nsw> : i64
            ascvec.get_buf s, %230, 0
            %231 = arith.muli %229, %c64_i64 overflow<nsw> : i64
            %232 = arith.addi %231, %c32_i64 overflow<nsw> : i64
            %233 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xui32, 1>
            %234 = arith.index_cast %232 : i64 to index
            %235 = arith.addi %234, %c60064 : index
            %236 = memref.load %233[%235] : memref<?xui32, 1>
            %237 = cannir.scalar_storage_cast %236 : ui32 -> i32
            ascvec.release_buf s, %230, 0
            scf.yield %237 : i32
          } else {
            scf.yield %c0_i32 : i32
          }
          %205 = arith.remsi %189, %c2_i64 : i64
          %206 = arith.cmpi ne, %205, %c0_i64 : i64
          %207 = arith.cmpi slt, %205, %c0_i64 : i64
          %208 = arith.andi %206, %207 : i1
          %209 = arith.select %208, %c2_i64, %c0_i64 : i64
          %210 = arith.addi %205, %209 overflow<nsw> : i64
          %211 = arith.muli %210, %c512_i64 overflow<nsw> : i64
          %212 = arith.remsi %193, %c2_i64 : i64
          %213 = arith.cmpi ne, %212, %c0_i64 : i64
          %214 = arith.cmpi slt, %212, %c0_i64 : i64
          %215 = arith.andi %213, %214 : i1
          %216 = arith.select %215, %c2_i64, %c0_i64 : i64
          %217 = arith.addi %212, %216 overflow<nsw> : i64
          %218 = arith.muli %217, %c256_i64 overflow<nsw> : i64
          %219 = arith.addi %211, %218 overflow<nsw> : i64
          %220 = arith.cmpi ne, %203#0, %c0_i32 : i32
          %221 = scf.if %220 -> (i32) {
            ascvec.get_buf v, %c4_i64, 0
            ascvec.sync_intra_wait v, %c4_i64
            %224 = arith.muli %219, %c8_i64 overflow<nsw> : i64
            %225 = arith.index_cast %224 : i64 to index
            %226 = arith.addi %225, %c1926144 : index
            %227 = arith.divui %226, %c32 : index
            %228 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xf32, 1>
            ascvec.vf(%228, %227, %228, %204, %228 : memref<?xf32, 1>, index, memref<?xf32, 1>, i32, memref<?xf32, 1>) {
            ^bb0(%arg21: memref<?xf32, 1>, %arg22: index, %arg23: memref<?xf32, 1>, %arg24: i32, %arg25: memref<?xf32, 1>):
              %c1_i32_10 = arith.constant 1 : i32
              %c32_i32_11 = arith.constant 32 : i32
              %c64_i32 = arith.constant 64 : i32
              %c33024 = arith.constant 33024 : index
              %c0_i32_12 = arith.constant 0 : i32
              %c49408 = arith.constant 49408 : index
              %c128_i32_13 = arith.constant 128 : i32
              %c512_i32 = arith.constant 512 : i32
              %c2_i32 = arith.constant 2 : i32
              %235 = ascvec.create_mask : !ascvec.mask
              scf.for %arg26 = %c0_i32_12 to %c32_i32_11 step %c1_i32_10  : i32 {
                %236 = arith.muli %arg26, %c2_i32 overflow<nsw> : i32
                %237 = arith.index_cast %236 : i32 to index
                %238 = arith.addi %arg22, %237 : index
                %239 = ascvec.loadalign_brc %arg21[%238] : memref<?xf32, 1>, index -> <f32>
                %240 = arith.muli %arg26, %c512_i32 overflow<nsw> : i32
                %241 = arith.muli %arg26, %c128_i32_13 overflow<nsw> : i32
                %242 = arith.index_cast %241 : i32 to index
                %243 = arith.addi %242, %c49408 : index
                %244 = ascvec.loadalign %arg23[%243] : memref<?xf32, 1>, index -> <f32>
                %245 = arith.cmpi ne, %arg24, %c0_i32_12 : i32
                %246 = scf.if %245 -> (!ascvec.vreg<f32>) {
                  %259 = arith.index_cast %240 : i32 to index
                  %260 = arith.addi %259, %c33024 : index
                  %261 = ascvec.loadalign %arg25[%260] : memref<?xf32, 1>, index -> <f32>
                  %262 = ascvec.add %244, %261, %235 : <f32>
                  scf.yield %262 : !ascvec.vreg<f32>
                } else {
                  scf.yield %244 : !ascvec.vreg<f32>
                }
                %247 = ascvec.mul %246, %239, %235 : <f32>
                %248 = arith.index_cast %240 : i32 to index
                %249 = arith.addi %248, %c33024 : index
                ascvec.storealign %arg25[%249], %247, %235 : memref<?xf32, 1>, index, <f32>
                %250 = arith.addi %240, %c64_i32 overflow<nsw> : i32
                %251 = arith.addi %241, %c64_i32 overflow<nsw> : i32
                %252 = arith.index_cast %251 : i32 to index
                %253 = arith.addi %252, %c49408 : index
                %254 = ascvec.loadalign %arg23[%253] : memref<?xf32, 1>, index -> <f32>
                %255 = scf.if %245 -> (!ascvec.vreg<f32>) {
                  %259 = arith.index_cast %250 : i32 to index
                  %260 = arith.addi %259, %c33024 : index
                  %261 = ascvec.loadalign %arg25[%260] : memref<?xf32, 1>, index -> <f32>
                  %262 = ascvec.add %254, %261, %235 : <f32>
                  scf.yield %262 : !ascvec.vreg<f32>
                } else {
                  scf.yield %254 : !ascvec.vreg<f32>
                }
                %256 = ascvec.mul %255, %239, %235 : <f32>
                %257 = arith.index_cast %250 : i32 to index
                %258 = arith.addi %257, %c33024 : index
                ascvec.storealign %arg25[%258], %256, %235 : memref<?xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.sync_intra_arrive v, %c4_i64
            ascvec.release_buf v, %c4_i64, 0
            ascvec.get_buf v, %c5_i64, 0
            ascvec.sync_intra_wait v, %c5_i64
            ascvec.vf(%228, %227, %228, %204, %228 : memref<?xf32, 1>, index, memref<?xf32, 1>, i32, memref<?xf32, 1>) {
            ^bb0(%arg21: memref<?xf32, 1>, %arg22: index, %arg23: memref<?xf32, 1>, %arg24: i32, %arg25: memref<?xf32, 1>):
              %c192_i32 = arith.constant 192 : i32
              %c1_i32_10 = arith.constant 1 : i32
              %c32_i32_11 = arith.constant 32 : i32
              %c64_i32 = arith.constant 64 : i32
              %c33024 = arith.constant 33024 : index
              %c0_i32_12 = arith.constant 0 : i32
              %c53504 = arith.constant 53504 : index
              %c128_i32_13 = arith.constant 128 : i32
              %c512_i32 = arith.constant 512 : i32
              %c2_i32 = arith.constant 2 : i32
              %235 = ascvec.create_mask : !ascvec.mask
              scf.for %arg26 = %c0_i32_12 to %c32_i32_11 step %c1_i32_10  : i32 {
                %236 = arith.muli %arg26, %c2_i32 overflow<nsw> : i32
                %237 = arith.index_cast %236 : i32 to index
                %238 = arith.addi %arg22, %237 : index
                %239 = ascvec.loadalign_brc %arg21[%238] : memref<?xf32, 1>, index -> <f32>
                %240 = arith.muli %arg26, %c512_i32 overflow<nsw> : i32
                %241 = arith.addi %240, %c128_i32_13 overflow<nsw> : i32
                %242 = arith.muli %arg26, %c128_i32_13 overflow<nsw> : i32
                %243 = arith.index_cast %242 : i32 to index
                %244 = arith.addi %243, %c53504 : index
                %245 = ascvec.loadalign %arg23[%244] : memref<?xf32, 1>, index -> <f32>
                %246 = arith.cmpi ne, %arg24, %c0_i32_12 : i32
                %247 = scf.if %246 -> (!ascvec.vreg<f32>) {
                  %260 = arith.index_cast %241 : i32 to index
                  %261 = arith.addi %260, %c33024 : index
                  %262 = ascvec.loadalign %arg25[%261] : memref<?xf32, 1>, index -> <f32>
                  %263 = ascvec.add %245, %262, %235 : <f32>
                  scf.yield %263 : !ascvec.vreg<f32>
                } else {
                  scf.yield %245 : !ascvec.vreg<f32>
                }
                %248 = ascvec.mul %247, %239, %235 : <f32>
                %249 = arith.index_cast %241 : i32 to index
                %250 = arith.addi %249, %c33024 : index
                ascvec.storealign %arg25[%250], %248, %235 : memref<?xf32, 1>, index, <f32>
                %251 = arith.addi %240, %c192_i32 overflow<nsw> : i32
                %252 = arith.addi %242, %c64_i32 overflow<nsw> : i32
                %253 = arith.index_cast %252 : i32 to index
                %254 = arith.addi %253, %c53504 : index
                %255 = ascvec.loadalign %arg23[%254] : memref<?xf32, 1>, index -> <f32>
                %256 = scf.if %246 -> (!ascvec.vreg<f32>) {
                  %260 = arith.index_cast %251 : i32 to index
                  %261 = arith.addi %260, %c33024 : index
                  %262 = ascvec.loadalign %arg25[%261] : memref<?xf32, 1>, index -> <f32>
                  %263 = ascvec.add %255, %262, %235 : <f32>
                  scf.yield %263 : !ascvec.vreg<f32>
                } else {
                  scf.yield %255 : !ascvec.vreg<f32>
                }
                %257 = ascvec.mul %256, %239, %235 : <f32>
                %258 = arith.index_cast %251 : i32 to index
                %259 = arith.addi %258, %c33024 : index
                ascvec.storealign %arg25[%259], %257, %235 : memref<?xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.sync_intra_arrive v, %c5_i64
            ascvec.release_buf v, %c5_i64, 0
            ascvec.get_buf v, %c6_i64, 0
            ascvec.sync_intra_wait v, %c4_i64
            ascvec.vf(%228, %227, %228, %204, %228 : memref<?xf32, 1>, index, memref<?xf32, 1>, i32, memref<?xf32, 1>) {
            ^bb0(%arg21: memref<?xf32, 1>, %arg22: index, %arg23: memref<?xf32, 1>, %arg24: i32, %arg25: memref<?xf32, 1>):
              %c320_i32 = arith.constant 320 : i32
              %c1_i32_10 = arith.constant 1 : i32
              %c32_i32_11 = arith.constant 32 : i32
              %c64_i32 = arith.constant 64 : i32
              %c33024 = arith.constant 33024 : index
              %c0_i32_12 = arith.constant 0 : i32
              %c49408 = arith.constant 49408 : index
              %c128_i32_13 = arith.constant 128 : i32
              %c256_i32 = arith.constant 256 : i32
              %c512_i32 = arith.constant 512 : i32
              %c2_i32 = arith.constant 2 : i32
              %235 = ascvec.create_mask : !ascvec.mask
              scf.for %arg26 = %c0_i32_12 to %c32_i32_11 step %c1_i32_10  : i32 {
                %236 = arith.muli %arg26, %c2_i32 overflow<nsw> : i32
                %237 = arith.index_cast %236 : i32 to index
                %238 = arith.addi %arg22, %237 : index
                %239 = ascvec.loadalign_brc %arg21[%238] : memref<?xf32, 1>, index -> <f32>
                %240 = arith.muli %arg26, %c512_i32 overflow<nsw> : i32
                %241 = arith.addi %240, %c256_i32 overflow<nsw> : i32
                %242 = arith.muli %arg26, %c128_i32_13 overflow<nsw> : i32
                %243 = arith.index_cast %242 : i32 to index
                %244 = arith.addi %243, %c49408 : index
                %245 = ascvec.loadalign %arg23[%244] : memref<?xf32, 1>, index -> <f32>
                %246 = arith.cmpi ne, %arg24, %c0_i32_12 : i32
                %247 = scf.if %246 -> (!ascvec.vreg<f32>) {
                  %260 = arith.index_cast %241 : i32 to index
                  %261 = arith.addi %260, %c33024 : index
                  %262 = ascvec.loadalign %arg25[%261] : memref<?xf32, 1>, index -> <f32>
                  %263 = ascvec.add %245, %262, %235 : <f32>
                  scf.yield %263 : !ascvec.vreg<f32>
                } else {
                  scf.yield %245 : !ascvec.vreg<f32>
                }
                %248 = ascvec.mul %247, %239, %235 : <f32>
                %249 = arith.index_cast %241 : i32 to index
                %250 = arith.addi %249, %c33024 : index
                ascvec.storealign %arg25[%250], %248, %235 : memref<?xf32, 1>, index, <f32>
                %251 = arith.addi %240, %c320_i32 overflow<nsw> : i32
                %252 = arith.addi %242, %c64_i32 overflow<nsw> : i32
                %253 = arith.index_cast %252 : i32 to index
                %254 = arith.addi %253, %c49408 : index
                %255 = ascvec.loadalign %arg23[%254] : memref<?xf32, 1>, index -> <f32>
                %256 = scf.if %246 -> (!ascvec.vreg<f32>) {
                  %260 = arith.index_cast %251 : i32 to index
                  %261 = arith.addi %260, %c33024 : index
                  %262 = ascvec.loadalign %arg25[%261] : memref<?xf32, 1>, index -> <f32>
                  %263 = ascvec.add %255, %262, %235 : <f32>
                  scf.yield %263 : !ascvec.vreg<f32>
                } else {
                  scf.yield %255 : !ascvec.vreg<f32>
                }
                %257 = ascvec.mul %256, %239, %235 : <f32>
                %258 = arith.index_cast %251 : i32 to index
                %259 = arith.addi %258, %c33024 : index
                ascvec.storealign %arg25[%259], %257, %235 : memref<?xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.sync_intra_arrive v, %c4_i64
            ascvec.release_buf v, %c6_i64, 0
            ascvec.get_buf v, %c7_i64, 0
            ascvec.sync_intra_wait v, %c5_i64
            %229 = arith.addi %210, %c2_i64 overflow<nsw> : i64
            ascvec.get_buf v, %229, 0
            %230 = arith.muli %210, %c64_i64 overflow<nsw> : i64
            %231 = arith.addi %230, %c32_i64 overflow<nsw> : i64
            %232 = arith.addi %204, %c1_i32 : i32
            %233 = arith.index_cast %231 : i64 to index
            %234 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xui32, 1>
            ascvec.vf(%228, %227, %228, %204, %228, %232, %234, %233 : memref<?xf32, 1>, index, memref<?xf32, 1>, i32, memref<?xf32, 1>, i32, memref<?xui32, 1>, index) {
            ^bb0(%arg21: memref<?xf32, 1>, %arg22: index, %arg23: memref<?xf32, 1>, %arg24: i32, %arg25: memref<?xf32, 1>, %arg26: i32, %arg27: memref<?xui32, 1>, %arg28: index):
              %c448_i32 = arith.constant 448 : i32
              %c60064_10 = arith.constant 60064 : index
              %c1_i32_11 = arith.constant 1 : i32
              %c32_i32_12 = arith.constant 32 : i32
              %c64_i32 = arith.constant 64 : i32
              %c33024 = arith.constant 33024 : index
              %c0_i32_13 = arith.constant 0 : i32
              %c53504 = arith.constant 53504 : index
              %c128_i32_14 = arith.constant 128 : i32
              %c384_i32 = arith.constant 384 : i32
              %c512_i32 = arith.constant 512 : i32
              %c2_i32 = arith.constant 2 : i32
              %235 = ascvec.create_mask : !ascvec.mask
              scf.for %arg29 = %c0_i32_13 to %c32_i32_12 step %c1_i32_11  : i32 {
                %238 = arith.muli %arg29, %c2_i32 overflow<nsw> : i32
                %239 = arith.index_cast %238 : i32 to index
                %240 = arith.addi %arg22, %239 : index
                %241 = ascvec.loadalign_brc %arg21[%240] : memref<?xf32, 1>, index -> <f32>
                %242 = arith.muli %arg29, %c512_i32 overflow<nsw> : i32
                %243 = arith.addi %242, %c384_i32 overflow<nsw> : i32
                %244 = arith.muli %arg29, %c128_i32_14 overflow<nsw> : i32
                %245 = arith.index_cast %244 : i32 to index
                %246 = arith.addi %245, %c53504 : index
                %247 = ascvec.loadalign %arg23[%246] : memref<?xf32, 1>, index -> <f32>
                %248 = arith.cmpi ne, %arg24, %c0_i32_13 : i32
                %249 = scf.if %248 -> (!ascvec.vreg<f32>) {
                  %262 = arith.index_cast %243 : i32 to index
                  %263 = arith.addi %262, %c33024 : index
                  %264 = ascvec.loadalign %arg25[%263] : memref<?xf32, 1>, index -> <f32>
                  %265 = ascvec.add %247, %264, %235 : <f32>
                  scf.yield %265 : !ascvec.vreg<f32>
                } else {
                  scf.yield %247 : !ascvec.vreg<f32>
                }
                %250 = ascvec.mul %249, %241, %235 : <f32>
                %251 = arith.index_cast %243 : i32 to index
                %252 = arith.addi %251, %c33024 : index
                ascvec.storealign %arg25[%252], %250, %235 : memref<?xf32, 1>, index, <f32>
                %253 = arith.addi %242, %c448_i32 overflow<nsw> : i32
                %254 = arith.addi %244, %c64_i32 overflow<nsw> : i32
                %255 = arith.index_cast %254 : i32 to index
                %256 = arith.addi %255, %c53504 : index
                %257 = ascvec.loadalign %arg23[%256] : memref<?xf32, 1>, index -> <f32>
                %258 = scf.if %248 -> (!ascvec.vreg<f32>) {
                  %262 = arith.index_cast %253 : i32 to index
                  %263 = arith.addi %262, %c33024 : index
                  %264 = ascvec.loadalign %arg25[%263] : memref<?xf32, 1>, index -> <f32>
                  %265 = ascvec.add %257, %264, %235 : <f32>
                  scf.yield %265 : !ascvec.vreg<f32>
                } else {
                  scf.yield %257 : !ascvec.vreg<f32>
                }
                %259 = ascvec.mul %258, %241, %235 : <f32>
                %260 = arith.index_cast %253 : i32 to index
                %261 = arith.addi %260, %c33024 : index
                ascvec.storealign %arg25[%261], %259, %235 : memref<?xf32, 1>, index, <f32>
              }
              %236 = ascvec.duplicate_scalar_nomask %arg26 : i32 -> <ui32>
              %237 = arith.addi %arg28, %c60064_10 : index
              %c0_15 = arith.constant 0 : index
              %c1_16 = arith.constant 1 : index
              scf.for %arg29 = %c0_15 to %c1_16 step %c1_16 {
                ascvec.storealign_1st %arg27[%237], %236 : memref<?xui32, 1>, index, <ui32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %229, 0
            ascvec.sync_intra_arrive v, %c5_i64
            ascvec.release_buf v, %c7_i64, 0
            scf.yield %232 : i32
          } else {
            scf.yield %204 : i32
          }
          %222 = arith.addi %193, %c1_i64 : i64
          %223 = arith.cmpi eq, %222, %195 : i64
          scf.if %223 {
            ascvec.get_buf v, %c4_i64, 0
            ascvec.sync_intra_wait v, %c4_i64
            %224 = arith.cmpi ne, %221, %c0_i32 : i32
            scf.if %224 {
              %389 = arith.muli %210, %c2048_i64 overflow<nsw> : i64
              %390 = arith.index_cast %389 : i64 to index
              %391 = arith.addi %390, %c1922048 : index
              %392 = arith.divui %391, %c32 : index
              %393 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xf32, 1>
              ascvec.vf(%393, %392, %393, %393 : memref<?xf32, 1>, index, memref<?xf32, 1>, memref<?xf32, 1>) {
              ^bb0(%arg21: memref<?xf32, 1>, %arg22: index, %arg23: memref<?xf32, 1>, %arg24: memref<?xf32, 1>):
                %c2_i32 = arith.constant 2 : i32
                %c32_i32_10 = arith.constant 32 : i32
                %c0_i32_11 = arith.constant 0 : i32
                %c33024 = arith.constant 33024 : index
                %c512_i32 = arith.constant 512 : i32
                %c64_i32 = arith.constant 64 : i32
                %c49408 = arith.constant 49408 : index
                %c128_i32_12 = arith.constant 128 : i32
                %c1_i32_13 = arith.constant 1 : i32
                %394 = ascvec.create_mask : !ascvec.mask
                scf.for %arg25 = %c0_i32_11 to %c32_i32_10 step %c2_i32  : i32 {
                  %395 = arith.index_cast %arg25 : i32 to index
                  %396 = arith.addi %arg22, %395 : index
                  %397 = ascvec.loadalign_brc %arg21[%396] : memref<?xf32, 1>, index -> <f32>
                  %398 = arith.addi %arg25, %c1_i32_13 overflow<nsw> : i32
                  %399 = arith.index_cast %398 : i32 to index
                  %400 = arith.addi %arg22, %399 : index
                  %401 = ascvec.loadalign_brc %arg21[%400] : memref<?xf32, 1>, index -> <f32>
                  %402 = arith.muli %arg25, %c128_i32_12 overflow<nsw> : i32
                  %403 = arith.index_cast %402 : i32 to index
                  %404 = arith.addi %403, %c49408 : index
                  %405 = ascvec.loadalign %arg23[%404] : memref<?xf32, 1>, index -> <f32>
                  %406 = arith.addi %402, %c64_i32 overflow<nsw> : i32
                  %407 = arith.index_cast %406 : i32 to index
                  %408 = arith.addi %407, %c49408 : index
                  %409 = ascvec.loadalign %arg23[%408] : memref<?xf32, 1>, index -> <f32>
                  %410 = arith.muli %398, %c128_i32_12 overflow<nsw> : i32
                  %411 = arith.index_cast %410 : i32 to index
                  %412 = arith.addi %411, %c49408 : index
                  %413 = ascvec.loadalign %arg23[%412] : memref<?xf32, 1>, index -> <f32>
                  %414 = arith.addi %410, %c64_i32 overflow<nsw> : i32
                  %415 = arith.index_cast %414 : i32 to index
                  %416 = arith.addi %415, %c49408 : index
                  %417 = ascvec.loadalign %arg23[%416] : memref<?xf32, 1>, index -> <f32>
                  %418 = arith.muli %arg25, %c512_i32 overflow<nsw> : i32
                  %419 = arith.index_cast %418 : i32 to index
                  %420 = arith.addi %419, %c33024 : index
                  %421 = ascvec.loadalign %arg24[%420] : memref<?xf32, 1>, index -> <f32>
                  %422 = ascvec.add %405, %421, %394 : <f32>
                  %423 = arith.addi %418, %c64_i32 overflow<nsw> : i32
                  %424 = arith.index_cast %423 : i32 to index
                  %425 = arith.addi %424, %c33024 : index
                  %426 = ascvec.loadalign %arg24[%425] : memref<?xf32, 1>, index -> <f32>
                  %427 = ascvec.add %409, %426, %394 : <f32>
                  %428 = arith.muli %398, %c512_i32 overflow<nsw> : i32
                  %429 = arith.index_cast %428 : i32 to index
                  %430 = arith.addi %429, %c33024 : index
                  %431 = ascvec.loadalign %arg24[%430] : memref<?xf32, 1>, index -> <f32>
                  %432 = ascvec.add %413, %431, %394 : <f32>
                  %433 = arith.addi %428, %c64_i32 overflow<nsw> : i32
                  %434 = arith.index_cast %433 : i32 to index
                  %435 = arith.addi %434, %c33024 : index
                  %436 = ascvec.loadalign %arg24[%435] : memref<?xf32, 1>, index -> <f32>
                  %437 = ascvec.add %417, %436, %394 : <f32>
                  %438 = ascvec.mul %422, %397, %394 : <f32>
                  ascvec.storealign %arg24[%420], %438, %394 : memref<?xf32, 1>, index, <f32>
                  %439 = ascvec.mul %427, %397, %394 : <f32>
                  ascvec.storealign %arg24[%425], %439, %394 : memref<?xf32, 1>, index, <f32>
                  %440 = ascvec.mul %432, %401, %394 : <f32>
                  ascvec.storealign %arg24[%430], %440, %394 : memref<?xf32, 1>, index, <f32>
                  %441 = ascvec.mul %437, %401, %394 : <f32>
                  ascvec.storealign %arg24[%435], %441, %394 : memref<?xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
            } else {
              %389 = arith.muli %210, %c2048_i64 overflow<nsw> : i64
              %390 = arith.index_cast %389 : i64 to index
              %391 = arith.addi %390, %c1922048 : index
              %392 = arith.divui %391, %c32 : index
              %393 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xf32, 1>
              ascvec.vf(%393, %392, %393, %393 : memref<?xf32, 1>, index, memref<?xf32, 1>, memref<?xf32, 1>) {
              ^bb0(%arg21: memref<?xf32, 1>, %arg22: index, %arg23: memref<?xf32, 1>, %arg24: memref<?xf32, 1>):
                %c2_i32 = arith.constant 2 : i32
                %c32_i32_10 = arith.constant 32 : i32
                %c0_i32_11 = arith.constant 0 : i32
                %c33024 = arith.constant 33024 : index
                %c512_i32 = arith.constant 512 : i32
                %c64_i32 = arith.constant 64 : i32
                %c49408 = arith.constant 49408 : index
                %c128_i32_12 = arith.constant 128 : i32
                %c1_i32_13 = arith.constant 1 : i32
                %394 = ascvec.create_mask : !ascvec.mask
                scf.for %arg25 = %c0_i32_11 to %c32_i32_10 step %c2_i32  : i32 {
                  %395 = arith.index_cast %arg25 : i32 to index
                  %396 = arith.addi %arg22, %395 : index
                  %397 = ascvec.loadalign_brc %arg21[%396] : memref<?xf32, 1>, index -> <f32>
                  %398 = arith.addi %arg25, %c1_i32_13 overflow<nsw> : i32
                  %399 = arith.index_cast %398 : i32 to index
                  %400 = arith.addi %arg22, %399 : index
                  %401 = ascvec.loadalign_brc %arg21[%400] : memref<?xf32, 1>, index -> <f32>
                  %402 = arith.muli %arg25, %c128_i32_12 overflow<nsw> : i32
                  %403 = arith.index_cast %402 : i32 to index
                  %404 = arith.addi %403, %c49408 : index
                  %405 = ascvec.loadalign %arg23[%404] : memref<?xf32, 1>, index -> <f32>
                  %406 = arith.addi %402, %c64_i32 overflow<nsw> : i32
                  %407 = arith.index_cast %406 : i32 to index
                  %408 = arith.addi %407, %c49408 : index
                  %409 = ascvec.loadalign %arg23[%408] : memref<?xf32, 1>, index -> <f32>
                  %410 = arith.muli %398, %c128_i32_12 overflow<nsw> : i32
                  %411 = arith.index_cast %410 : i32 to index
                  %412 = arith.addi %411, %c49408 : index
                  %413 = ascvec.loadalign %arg23[%412] : memref<?xf32, 1>, index -> <f32>
                  %414 = arith.addi %410, %c64_i32 overflow<nsw> : i32
                  %415 = arith.index_cast %414 : i32 to index
                  %416 = arith.addi %415, %c49408 : index
                  %417 = ascvec.loadalign %arg23[%416] : memref<?xf32, 1>, index -> <f32>
                  %418 = arith.muli %arg25, %c512_i32 overflow<nsw> : i32
                  %419 = ascvec.mul %405, %397, %394 : <f32>
                  %420 = arith.index_cast %418 : i32 to index
                  %421 = arith.addi %420, %c33024 : index
                  ascvec.storealign %arg24[%421], %419, %394 : memref<?xf32, 1>, index, <f32>
                  %422 = arith.addi %418, %c64_i32 overflow<nsw> : i32
                  %423 = ascvec.mul %409, %397, %394 : <f32>
                  %424 = arith.index_cast %422 : i32 to index
                  %425 = arith.addi %424, %c33024 : index
                  ascvec.storealign %arg24[%425], %423, %394 : memref<?xf32, 1>, index, <f32>
                  %426 = arith.muli %398, %c512_i32 overflow<nsw> : i32
                  %427 = ascvec.mul %413, %401, %394 : <f32>
                  %428 = arith.index_cast %426 : i32 to index
                  %429 = arith.addi %428, %c33024 : index
                  ascvec.storealign %arg24[%429], %427, %394 : memref<?xf32, 1>, index, <f32>
                  %430 = arith.addi %426, %c64_i32 overflow<nsw> : i32
                  %431 = ascvec.mul %417, %401, %394 : <f32>
                  %432 = arith.index_cast %430 : i32 to index
                  %433 = arith.addi %432, %c33024 : index
                  ascvec.storealign %arg24[%433], %431, %394 : memref<?xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
            }
            %225 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xf32, 1>
            %226 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xf8E4M3FN, 1>
            %227 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xi32, 1>
            ascvec.vf(%225, %226, %227 : memref<?xf32, 1>, memref<?xf8E4M3FN, 1>, memref<?xi32, 1>) {
            ^bb0(%arg21: memref<?xf32, 1>, %arg22: memref<?xf8E4M3FN, 1>, %arg23: memref<?xi32, 1>):
              %c3_i32 = arith.constant 3 : i32
              %c2_i32 = arith.constant 2 : i32
              %c64_i32 = arith.constant 64 : i32
              %c1_i32_10 = arith.constant 1 : i32
              %c50432 = arith.constant 50432 : index
              %c4_i32_11 = arith.constant 4 : i32
              %c256_i32 = arith.constant 256 : i32
              %c197632 = arith.constant 197632 : index
              %c1024_i32_12 = arith.constant 1024 : i32
              %c8_i32_13 = arith.constant 8 : i32
              %c254_i32 = arith.constant 254 : i32
              %c105_i32 = arith.constant 105 : i32
              %c8388607_i32 = arith.constant 8388607 : i32
              %cst_14 = arith.constant 0.00223214296 : f32
              %c33024 = arith.constant 33024 : index
              %c512_i32 = arith.constant 512 : i32
              %c32_i32_15 = arith.constant 32 : i32
              %c31_i32 = arith.constant 31 : i32
              %c0_i32_16 = arith.constant 0 : i32
              %389 = ascvec.create_mask : !ascvec.mask
              %390 = ascvec.create_mask pat_mode "PAT_VL32" : !ascvec.mask
              %391 = ascvec.arange %c0_i32_16 : i32 -> <ui32>
              %392 = ascvec.duplicate_scalar_nomask %c31_i32 : i32 -> <ui32>
              %393 = ascvec.and %391, %392, %389 : <ui32>, <ui32> -> <ui32>
              %394 = ascvec.add_scalar %393, %c32_i32_15, %389 : <ui32>, i32
              ascvec.mem_bar vst_vld
              scf.for %arg24 = %c0_i32_16 to %c32_i32_15 step %c1_i32_10  : i32 {
                %395 = arith.muli %arg24, %c512_i32 overflow<nsw> : i32
                %396 = arith.index_cast %395 : i32 to index
                %397 = arith.addi %396, %c33024 : index
                %398 = ascvec.loadalign %arg21[%397] : memref<?xf32, 1>, index -> <f32>
                %399 = ascvec.abs %398, %389 : <f32>
                %400 = ascvec.reduce_max %399, %390 : <f32>
                %401 = ascvec.duplicate %400, %389 : <f32> -> <f32>
                %402 = ascvec.gather_reg %399, %394 : <f32>, <ui32> -> <f32>
                %403 = ascvec.reduce_max %402, %390 : <f32>
                %404 = ascvec.duplicate %403, %389 : <f32> -> <f32>
                %405 = ascvec.select %401, %404, %390 : <f32>
                %406 = ascvec.mul_scalar %405, %cst_14, %389 : <f32>, f32
                %407 = ascvec.reinterpret %406 : <f32> -> <ui32>
                %408 = ascvec.add_scalar %407, %c8388607_i32, %389 : <ui32>, i32
                %409 = ascvec.shr %408, 23, %389 : <ui32> -> <ui32>
                %410 = ascvec.max_scalar %409, %c105_i32, %389 : <ui32>, i32
                %411 = ascvec.duplicate_scalar_nomask %c254_i32 : i32 -> <ui32>
                %412 = ascvec.sub %411, %410, %389 : <ui32>
                %413 = ascvec.shl %412, 23, %389 : <ui32> -> <ui32>
                %414 = ascvec.reinterpret %413 : <ui32> -> <f32>
                %415 = ascvec.mul %398, %414, %389 : <f32>
                %416 = ascvec.cast %415, %389 {reg_layout = 0 : i32, saturate = true} : <f32> -> <f8E4M3FN>
                %417 = ascvec.reinterpret %416 : <f8E4M3FN> -> <ui32>
                %418 = ascvec.gather_reg %417, %394 : <ui32>, <ui32> -> <ui32>
                %419 = ascvec.reinterpret %418 : <ui32> -> <f8E4M3FN>
                %420 = ascvec.gather_reg %410, %394 : <ui32>, <ui32> -> <ui32>
                %421 = arith.divsi %arg24, %c8_i32_13 : i32
                %422 = arith.muli %421, %c1024_i32_12 overflow<nsw> : i32
                %423 = arith.remsi %arg24, %c8_i32_13 : i32
                %424 = arith.muli %423, %c32_i32_15 overflow<nsw> : i32
                %425 = arith.addi %422, %424 overflow<nsw> : i32
                %426 = arith.index_cast %425 : i32 to index
                %427 = arith.addi %426, %c197632 : index
                ascvec.storealign_pack %arg22[%427], %416, %390 {mode = "b32_to_b8"} : memref<?xf8E4M3FN, 1>, index, <f8E4M3FN>
                %428 = arith.addi %425, %c256_i32 overflow<nsw> : i32
                %429 = arith.index_cast %428 : i32 to index
                %430 = arith.addi %429, %c197632 : index
                ascvec.storealign_pack %arg22[%430], %419, %390 {mode = "b32_to_b8"} : memref<?xf8E4M3FN, 1>, index, <f8E4M3FN>
                %431 = arith.muli %arg24, %c4_i32_11 overflow<nsw> : i32
                %432 = ascvec.reinterpret %410 : <ui32> -> <i32>
                %433 = arith.index_cast %431 : i32 to index
                %434 = arith.addi %433, %c50432 : index
                ascvec.storealign_1st %arg23[%434], %432 : memref<?xi32, 1>, index, <i32>
                %435 = arith.addi %431, %c1_i32_10 overflow<nsw> : i32
                %436 = ascvec.reinterpret %420 : <ui32> -> <i32>
                %437 = arith.index_cast %435 : i32 to index
                %438 = arith.addi %437, %c50432 : index
                ascvec.storealign_1st %arg23[%438], %436 : memref<?xi32, 1>, index, <i32>
                %439 = arith.addi %395, %c64_i32 overflow<nsw> : i32
                %440 = arith.index_cast %439 : i32 to index
                %441 = arith.addi %440, %c33024 : index
                %442 = ascvec.loadalign %arg21[%441] : memref<?xf32, 1>, index -> <f32>
                %443 = ascvec.abs %442, %389 : <f32>
                %444 = ascvec.reduce_max %443, %390 : <f32>
                %445 = ascvec.duplicate %444, %389 : <f32> -> <f32>
                %446 = ascvec.gather_reg %443, %394 : <f32>, <ui32> -> <f32>
                %447 = ascvec.reduce_max %446, %390 : <f32>
                %448 = ascvec.duplicate %447, %389 : <f32> -> <f32>
                %449 = ascvec.select %445, %448, %390 : <f32>
                %450 = ascvec.mul_scalar %449, %cst_14, %389 : <f32>, f32
                %451 = ascvec.reinterpret %450 : <f32> -> <ui32>
                %452 = ascvec.add_scalar %451, %c8388607_i32, %389 : <ui32>, i32
                %453 = ascvec.shr %452, 23, %389 : <ui32> -> <ui32>
                %454 = ascvec.max_scalar %453, %c105_i32, %389 : <ui32>, i32
                %455 = ascvec.sub %411, %454, %389 : <ui32>
                %456 = ascvec.shl %455, 23, %389 : <ui32> -> <ui32>
                %457 = ascvec.reinterpret %456 : <ui32> -> <f32>
                %458 = ascvec.mul %442, %457, %389 : <f32>
                %459 = ascvec.cast %458, %389 {reg_layout = 0 : i32, saturate = true} : <f32> -> <f8E4M3FN>
                %460 = ascvec.reinterpret %459 : <f8E4M3FN> -> <ui32>
                %461 = ascvec.gather_reg %460, %394 : <ui32>, <ui32> -> <ui32>
                %462 = ascvec.reinterpret %461 : <ui32> -> <f8E4M3FN>
                %463 = ascvec.gather_reg %454, %394 : <ui32>, <ui32> -> <ui32>
                %464 = arith.addi %422, %c512_i32 overflow<nsw> : i32
                %465 = arith.addi %464, %424 overflow<nsw> : i32
                %466 = arith.index_cast %465 : i32 to index
                %467 = arith.addi %466, %c197632 : index
                ascvec.storealign_pack %arg22[%467], %459, %390 {mode = "b32_to_b8"} : memref<?xf8E4M3FN, 1>, index, <f8E4M3FN>
                %468 = arith.addi %465, %c256_i32 overflow<nsw> : i32
                %469 = arith.index_cast %468 : i32 to index
                %470 = arith.addi %469, %c197632 : index
                ascvec.storealign_pack %arg22[%470], %462, %390 {mode = "b32_to_b8"} : memref<?xf8E4M3FN, 1>, index, <f8E4M3FN>
                %471 = arith.addi %431, %c2_i32 overflow<nsw> : i32
                %472 = ascvec.reinterpret %454 : <ui32> -> <i32>
                %473 = arith.index_cast %471 : i32 to index
                %474 = arith.addi %473, %c50432 : index
                ascvec.storealign_1st %arg23[%474], %472 : memref<?xi32, 1>, index, <i32>
                %475 = arith.addi %431, %c3_i32 overflow<nsw> : i32
                %476 = ascvec.reinterpret %463 : <ui32> -> <i32>
                %477 = arith.index_cast %475 : i32 to index
                %478 = arith.addi %477, %c50432 : index
                ascvec.storealign_1st %arg23[%478], %476 : memref<?xi32, 1>, index, <i32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.vf(%227, %227 : memref<?xi32, 1>, memref<?xi32, 1>) {
            ^bb0(%arg21: memref<?xi32, 1>, %arg22: memref<?xi32, 1>):
              %c50880 = arith.constant 50880 : index
              %c50624 = arith.constant 50624 : index
              %c12_i32 = arith.constant 12 : i32
              %c8_i32_10 = arith.constant 8 : i32
              %c4_i32_11 = arith.constant 4 : i32
              %c50432 = arith.constant 50432 : index
              %c1_i32_12 = arith.constant 1 : i32
              %c7_i32 = arith.constant 7 : i32
              %c0_i32_13 = arith.constant 0 : i32
              %c0_14 = arith.constant 0 : index
              %c1_15 = arith.constant 1 : index
              scf.for %arg23 = %c0_14 to %c1_15 step %c1_15 {
                %389 = ascvec.create_mask : !ascvec.mask
                %390 = ascvec.create_mask pat_mode "PAT_VL32" : !ascvec.mask
                %391 = ascvec.arange %c0_i32_13 : i32 -> <ui32>
                %392 = ascvec.shr %391, 3, %389 : <ui32> -> <ui32>
                %393 = ascvec.duplicate_scalar_nomask %c7_i32 : i32 -> <ui32>
                %394 = ascvec.and %391, %393, %389 : <ui32>, <ui32> -> <ui32>
                %395 = ascvec.shr %394, 1, %389 : <ui32> -> <ui32>
                %396 = ascvec.duplicate_scalar_nomask %c1_i32_12 : i32 -> <ui32>
                %397 = ascvec.and %394, %396, %389 : <ui32>, <ui32> -> <ui32>
                %398 = ascvec.shl %392, 5, %389 : <ui32> -> <ui32>
                %399 = ascvec.shl %397, 4, %389 : <ui32> -> <ui32>
                %400 = ascvec.add %398, %399, %389 : <ui32>
                %401 = ascvec.add %400, %395, %389 : <ui32>
                %402 = ascvec.gather %arg21[%c50432], %401, %390 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                %403 = ascvec.reinterpret %402 : <i32> -> <ui32>
                %404 = ascvec.add_scalar %401, %c4_i32_11, %389 : <ui32>, i32
                %405 = ascvec.gather %arg21[%c50432], %404, %390 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                %406 = ascvec.reinterpret %405 : <i32> -> <ui32>
                %407 = ascvec.shl %406, 8, %389 : <ui32> -> <ui32>
                %408 = ascvec.or %403, %407, %389 : <ui32>, <ui32> -> <ui32>
                %409 = ascvec.add_scalar %401, %c8_i32_10, %389 : <ui32>, i32
                %410 = ascvec.gather %arg21[%c50432], %409, %390 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                %411 = ascvec.reinterpret %410 : <i32> -> <ui32>
                %412 = ascvec.shl %411, 16, %389 : <ui32> -> <ui32>
                %413 = ascvec.or %408, %412, %389 : <ui32>, <ui32> -> <ui32>
                %414 = ascvec.add_scalar %401, %c12_i32, %389 : <ui32>, i32
                %415 = ascvec.gather %arg21[%c50432], %414, %390 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                %416 = ascvec.reinterpret %415 : <i32> -> <ui32>
                %417 = ascvec.shl %416, 24, %389 : <ui32> -> <ui32>
                %418 = ascvec.or %413, %417, %389 : <ui32>, <ui32> -> <ui32>
                %419 = ascvec.reinterpret %418 : <ui32> -> <i32>
                %420 = ascvec.shl %391, 3, %389 : <ui32> -> <ui32>
                ascvec.scatter %arg22[%c50624], %419, %420, %390 : memref<?xi32, 1>, <i32>, <ui32>, !ascvec.mask
                %421 = ascvec.duplicate_scalar_nomask %c0_i32_13 : i32 -> <i32>
                ascvec.storealign %arg22[%c50880], %421, %389 : memref<?xi32, 1>, index, <i32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c4_i64, 0
            ascvec.get_buf mte3, %c4_i64, 0
            %228 = arith.muli %0, %c4_i64 : i64
            %229 = arith.muli %0, %c16_i64 : i64
            %230 = arith.muli %0, %c512_i64 : i64
            %231 = arith.addi %230, %191 : i64
            %232 = arith.muli %231, %c32_i64 : i64
            %233 = ascscalar.memref_address %alloc_7 : memref<1024xi32>
            %234 = arith.divui %232, %c8_i64 : i64
            %235 = arith.addi %233, %234 : i64
            %236 = ascscalar.bind_gm_ptr %235 : memref<1xi32>
            %237 = ascscalar.bind_local_ptr %c202496_i64 : memref<1xi32, 1>
            ascvec.copy_ub2gm %236[%c0], %237[%c0], %c8_i32, %c4_i32, %c16_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
            %238 = arith.addi %228, %c1_i64 : i64
            %239 = arith.muli %238, %c4_i64 : i64
            %240 = arith.muli %238, %c128_i64 : i64
            %241 = arith.addi %240, %191 : i64
            %242 = arith.muli %241, %c32_i64 : i64
            %243 = arith.divui %242, %c8_i64 : i64
            %244 = arith.addi %233, %243 : i64
            %245 = ascscalar.bind_gm_ptr %244 : memref<1xi32>
            %246 = ascscalar.bind_local_ptr %c202752_i64 : memref<1xi32, 1>
            ascvec.copy_ub2gm %245[%c0], %246[%c0], %c8_i32, %c4_i32, %c16_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
            %247 = arith.addi %228, %c2_i64 : i64
            %248 = arith.muli %247, %c4_i64 : i64
            %249 = arith.muli %247, %c128_i64 : i64
            %250 = arith.addi %249, %191 : i64
            %251 = arith.muli %250, %c32_i64 : i64
            %252 = arith.divui %251, %c8_i64 : i64
            %253 = arith.addi %233, %252 : i64
            %254 = ascscalar.bind_gm_ptr %253 : memref<1xi32>
            %255 = ascscalar.bind_local_ptr %c203008_i64 : memref<1xi32, 1>
            ascvec.copy_ub2gm %254[%c0], %255[%c0], %c8_i32, %c4_i32, %c16_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
            %256 = arith.addi %228, %c3_i64 : i64
            %257 = arith.muli %256, %c4_i64 : i64
            %258 = arith.muli %256, %c128_i64 : i64
            %259 = arith.addi %258, %191 : i64
            %260 = arith.muli %259, %c32_i64 : i64
            %261 = arith.divui %260, %c8_i64 : i64
            %262 = arith.addi %233, %261 : i64
            %263 = ascscalar.bind_gm_ptr %262 : memref<1xi32>
            %264 = ascscalar.bind_local_ptr %c203264_i64 : memref<1xi32, 1>
            ascvec.copy_ub2gm %263[%c0], %264[%c0], %c8_i32, %c4_i32, %c16_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
            %265 = arith.muli %191, %c8_i64 : i64
            %266 = arith.addi %265, %228 : i64
            %267 = arith.remsi %266, %c4_i64 : i64
            %268 = arith.cmpi ne, %267, %c0_i64 : i64
            %269 = arith.cmpi slt, %267, %c0_i64 : i64
            %270 = arith.andi %268, %269 : i1
            %271 = arith.divsi %266, %c4_i64 : i64
            %272 = arith.extui %270 : i1 to i64
            %273 = arith.subi %271, %272 : i64
            %274 = arith.muli %273, %c131072_i64 : i64
            %275 = ascscalar.memref_address %alloc_6 : memref<131072xf8E4M3FN>
            %276 = arith.divui %274, %c8_i64 : i64
            %277 = arith.addi %275, %276 : i64
            %278 = ascscalar.bind_gm_ptr %277 : memref<1xf8E4M3FN>
            %279 = ascscalar.bind_local_ptr %c197632_i64 : memref<1xf8E4M3FN, 1>
            ascvec.copy_ub2gm %278[%c0], %279[%c0], %c4_i32, %c1024_i32, %c4096_i64, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xf8E4M3FN>, memref<1xf8E4M3FN, 1>
            ascvec.release_buf mte3, %c4_i64, 0
            ascvec.get_buf v, %c4_i64, 0
            ascvec.sync_intra_arrive v, %c4_i64
            ascvec.release_buf v, %c4_i64, 0
            ascvec.get_buf v, %c5_i64, 0
            ascvec.sync_intra_wait v, %c5_i64
            scf.if %224 {
              %389 = arith.muli %210, %c2048_i64 overflow<nsw> : i64
              %390 = arith.index_cast %389 : i64 to index
              %391 = arith.addi %390, %c1922048 : index
              %392 = arith.divui %391, %c32 : index
              ascvec.vf(%225, %392, %225, %225 : memref<?xf32, 1>, index, memref<?xf32, 1>, memref<?xf32, 1>) {
              ^bb0(%arg21: memref<?xf32, 1>, %arg22: index, %arg23: memref<?xf32, 1>, %arg24: memref<?xf32, 1>):
                %c2_i32 = arith.constant 2 : i32
                %c32_i32_10 = arith.constant 32 : i32
                %c0_i32_11 = arith.constant 0 : i32
                %c33152 = arith.constant 33152 : index
                %c512_i32 = arith.constant 512 : i32
                %c64_i32 = arith.constant 64 : i32
                %c53504 = arith.constant 53504 : index
                %c128_i32_12 = arith.constant 128 : i32
                %c1_i32_13 = arith.constant 1 : i32
                %393 = ascvec.create_mask : !ascvec.mask
                scf.for %arg25 = %c0_i32_11 to %c32_i32_10 step %c2_i32  : i32 {
                  %394 = arith.index_cast %arg25 : i32 to index
                  %395 = arith.addi %arg22, %394 : index
                  %396 = ascvec.loadalign_brc %arg21[%395] : memref<?xf32, 1>, index -> <f32>
                  %397 = arith.addi %arg25, %c1_i32_13 overflow<nsw> : i32
                  %398 = arith.index_cast %397 : i32 to index
                  %399 = arith.addi %arg22, %398 : index
                  %400 = ascvec.loadalign_brc %arg21[%399] : memref<?xf32, 1>, index -> <f32>
                  %401 = arith.muli %arg25, %c128_i32_12 overflow<nsw> : i32
                  %402 = arith.index_cast %401 : i32 to index
                  %403 = arith.addi %402, %c53504 : index
                  %404 = ascvec.loadalign %arg23[%403] : memref<?xf32, 1>, index -> <f32>
                  %405 = arith.addi %401, %c64_i32 overflow<nsw> : i32
                  %406 = arith.index_cast %405 : i32 to index
                  %407 = arith.addi %406, %c53504 : index
                  %408 = ascvec.loadalign %arg23[%407] : memref<?xf32, 1>, index -> <f32>
                  %409 = arith.muli %397, %c128_i32_12 overflow<nsw> : i32
                  %410 = arith.index_cast %409 : i32 to index
                  %411 = arith.addi %410, %c53504 : index
                  %412 = ascvec.loadalign %arg23[%411] : memref<?xf32, 1>, index -> <f32>
                  %413 = arith.addi %409, %c64_i32 overflow<nsw> : i32
                  %414 = arith.index_cast %413 : i32 to index
                  %415 = arith.addi %414, %c53504 : index
                  %416 = ascvec.loadalign %arg23[%415] : memref<?xf32, 1>, index -> <f32>
                  %417 = arith.muli %arg25, %c512_i32 overflow<nsw> : i32
                  %418 = arith.index_cast %417 : i32 to index
                  %419 = arith.addi %418, %c33152 : index
                  %420 = ascvec.loadalign %arg24[%419] : memref<?xf32, 1>, index -> <f32>
                  %421 = ascvec.add %404, %420, %393 : <f32>
                  %422 = arith.addi %417, %c64_i32 overflow<nsw> : i32
                  %423 = arith.index_cast %422 : i32 to index
                  %424 = arith.addi %423, %c33152 : index
                  %425 = ascvec.loadalign %arg24[%424] : memref<?xf32, 1>, index -> <f32>
                  %426 = ascvec.add %408, %425, %393 : <f32>
                  %427 = arith.muli %397, %c512_i32 overflow<nsw> : i32
                  %428 = arith.index_cast %427 : i32 to index
                  %429 = arith.addi %428, %c33152 : index
                  %430 = ascvec.loadalign %arg24[%429] : memref<?xf32, 1>, index -> <f32>
                  %431 = ascvec.add %412, %430, %393 : <f32>
                  %432 = arith.addi %427, %c64_i32 overflow<nsw> : i32
                  %433 = arith.index_cast %432 : i32 to index
                  %434 = arith.addi %433, %c33152 : index
                  %435 = ascvec.loadalign %arg24[%434] : memref<?xf32, 1>, index -> <f32>
                  %436 = ascvec.add %416, %435, %393 : <f32>
                  %437 = ascvec.mul %421, %396, %393 : <f32>
                  ascvec.storealign %arg24[%419], %437, %393 : memref<?xf32, 1>, index, <f32>
                  %438 = ascvec.mul %426, %396, %393 : <f32>
                  ascvec.storealign %arg24[%424], %438, %393 : memref<?xf32, 1>, index, <f32>
                  %439 = ascvec.mul %431, %400, %393 : <f32>
                  ascvec.storealign %arg24[%429], %439, %393 : memref<?xf32, 1>, index, <f32>
                  %440 = ascvec.mul %436, %400, %393 : <f32>
                  ascvec.storealign %arg24[%434], %440, %393 : memref<?xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
            } else {
              %389 = arith.muli %210, %c2048_i64 overflow<nsw> : i64
              %390 = arith.index_cast %389 : i64 to index
              %391 = arith.addi %390, %c1922048 : index
              %392 = arith.divui %391, %c32 : index
              ascvec.vf(%225, %392, %225, %225 : memref<?xf32, 1>, index, memref<?xf32, 1>, memref<?xf32, 1>) {
              ^bb0(%arg21: memref<?xf32, 1>, %arg22: index, %arg23: memref<?xf32, 1>, %arg24: memref<?xf32, 1>):
                %c2_i32 = arith.constant 2 : i32
                %c32_i32_10 = arith.constant 32 : i32
                %c0_i32_11 = arith.constant 0 : i32
                %c33152 = arith.constant 33152 : index
                %c512_i32 = arith.constant 512 : i32
                %c64_i32 = arith.constant 64 : i32
                %c53504 = arith.constant 53504 : index
                %c128_i32_12 = arith.constant 128 : i32
                %c1_i32_13 = arith.constant 1 : i32
                %393 = ascvec.create_mask : !ascvec.mask
                scf.for %arg25 = %c0_i32_11 to %c32_i32_10 step %c2_i32  : i32 {
                  %394 = arith.index_cast %arg25 : i32 to index
                  %395 = arith.addi %arg22, %394 : index
                  %396 = ascvec.loadalign_brc %arg21[%395] : memref<?xf32, 1>, index -> <f32>
                  %397 = arith.addi %arg25, %c1_i32_13 overflow<nsw> : i32
                  %398 = arith.index_cast %397 : i32 to index
                  %399 = arith.addi %arg22, %398 : index
                  %400 = ascvec.loadalign_brc %arg21[%399] : memref<?xf32, 1>, index -> <f32>
                  %401 = arith.muli %arg25, %c128_i32_12 overflow<nsw> : i32
                  %402 = arith.index_cast %401 : i32 to index
                  %403 = arith.addi %402, %c53504 : index
                  %404 = ascvec.loadalign %arg23[%403] : memref<?xf32, 1>, index -> <f32>
                  %405 = arith.addi %401, %c64_i32 overflow<nsw> : i32
                  %406 = arith.index_cast %405 : i32 to index
                  %407 = arith.addi %406, %c53504 : index
                  %408 = ascvec.loadalign %arg23[%407] : memref<?xf32, 1>, index -> <f32>
                  %409 = arith.muli %397, %c128_i32_12 overflow<nsw> : i32
                  %410 = arith.index_cast %409 : i32 to index
                  %411 = arith.addi %410, %c53504 : index
                  %412 = ascvec.loadalign %arg23[%411] : memref<?xf32, 1>, index -> <f32>
                  %413 = arith.addi %409, %c64_i32 overflow<nsw> : i32
                  %414 = arith.index_cast %413 : i32 to index
                  %415 = arith.addi %414, %c53504 : index
                  %416 = ascvec.loadalign %arg23[%415] : memref<?xf32, 1>, index -> <f32>
                  %417 = arith.muli %arg25, %c512_i32 overflow<nsw> : i32
                  %418 = ascvec.mul %404, %396, %393 : <f32>
                  %419 = arith.index_cast %417 : i32 to index
                  %420 = arith.addi %419, %c33152 : index
                  ascvec.storealign %arg24[%420], %418, %393 : memref<?xf32, 1>, index, <f32>
                  %421 = arith.addi %417, %c64_i32 overflow<nsw> : i32
                  %422 = ascvec.mul %408, %396, %393 : <f32>
                  %423 = arith.index_cast %421 : i32 to index
                  %424 = arith.addi %423, %c33152 : index
                  ascvec.storealign %arg24[%424], %422, %393 : memref<?xf32, 1>, index, <f32>
                  %425 = arith.muli %397, %c512_i32 overflow<nsw> : i32
                  %426 = ascvec.mul %412, %400, %393 : <f32>
                  %427 = arith.index_cast %425 : i32 to index
                  %428 = arith.addi %427, %c33152 : index
                  ascvec.storealign %arg24[%428], %426, %393 : memref<?xf32, 1>, index, <f32>
                  %429 = arith.addi %425, %c64_i32 overflow<nsw> : i32
                  %430 = ascvec.mul %416, %400, %393 : <f32>
                  %431 = arith.index_cast %429 : i32 to index
                  %432 = arith.addi %431, %c33152 : index
                  ascvec.storealign %arg24[%432], %430, %393 : memref<?xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
            }
            ascvec.vf(%225, %226, %227 : memref<?xf32, 1>, memref<?xf8E4M3FN, 1>, memref<?xi32, 1>) {
            ^bb0(%arg21: memref<?xf32, 1>, %arg22: memref<?xf8E4M3FN, 1>, %arg23: memref<?xi32, 1>):
              %c3_i32 = arith.constant 3 : i32
              %c192_i32 = arith.constant 192 : i32
              %c2_i32 = arith.constant 2 : i32
              %c1_i32_10 = arith.constant 1 : i32
              %c54528 = arith.constant 54528 : index
              %c4_i32_11 = arith.constant 4 : i32
              %c256_i32 = arith.constant 256 : i32
              %c214016 = arith.constant 214016 : index
              %c1024_i32_12 = arith.constant 1024 : i32
              %c8_i32_13 = arith.constant 8 : i32
              %c254_i32 = arith.constant 254 : i32
              %c105_i32 = arith.constant 105 : i32
              %c8388607_i32 = arith.constant 8388607 : i32
              %cst_14 = arith.constant 0.00223214296 : f32
              %c33024 = arith.constant 33024 : index
              %c128_i32_15 = arith.constant 128 : i32
              %c512_i32 = arith.constant 512 : i32
              %c32_i32_16 = arith.constant 32 : i32
              %c31_i32 = arith.constant 31 : i32
              %c0_i32_17 = arith.constant 0 : i32
              %389 = ascvec.create_mask : !ascvec.mask
              %390 = ascvec.create_mask pat_mode "PAT_VL32" : !ascvec.mask
              %391 = ascvec.arange %c0_i32_17 : i32 -> <ui32>
              %392 = ascvec.duplicate_scalar_nomask %c31_i32 : i32 -> <ui32>
              %393 = ascvec.and %391, %392, %389 : <ui32>, <ui32> -> <ui32>
              %394 = ascvec.add_scalar %393, %c32_i32_16, %389 : <ui32>, i32
              ascvec.mem_bar vst_vld
              scf.for %arg24 = %c0_i32_17 to %c32_i32_16 step %c1_i32_10  : i32 {
                %395 = arith.muli %arg24, %c512_i32 overflow<nsw> : i32
                %396 = arith.addi %395, %c128_i32_15 overflow<nsw> : i32
                %397 = arith.index_cast %396 : i32 to index
                %398 = arith.addi %397, %c33024 : index
                %399 = ascvec.loadalign %arg21[%398] : memref<?xf32, 1>, index -> <f32>
                %400 = ascvec.abs %399, %389 : <f32>
                %401 = ascvec.reduce_max %400, %390 : <f32>
                %402 = ascvec.duplicate %401, %389 : <f32> -> <f32>
                %403 = ascvec.gather_reg %400, %394 : <f32>, <ui32> -> <f32>
                %404 = ascvec.reduce_max %403, %390 : <f32>
                %405 = ascvec.duplicate %404, %389 : <f32> -> <f32>
                %406 = ascvec.select %402, %405, %390 : <f32>
                %407 = ascvec.mul_scalar %406, %cst_14, %389 : <f32>, f32
                %408 = ascvec.reinterpret %407 : <f32> -> <ui32>
                %409 = ascvec.add_scalar %408, %c8388607_i32, %389 : <ui32>, i32
                %410 = ascvec.shr %409, 23, %389 : <ui32> -> <ui32>
                %411 = ascvec.max_scalar %410, %c105_i32, %389 : <ui32>, i32
                %412 = ascvec.duplicate_scalar_nomask %c254_i32 : i32 -> <ui32>
                %413 = ascvec.sub %412, %411, %389 : <ui32>
                %414 = ascvec.shl %413, 23, %389 : <ui32> -> <ui32>
                %415 = ascvec.reinterpret %414 : <ui32> -> <f32>
                %416 = ascvec.mul %399, %415, %389 : <f32>
                %417 = ascvec.cast %416, %389 {reg_layout = 0 : i32, saturate = true} : <f32> -> <f8E4M3FN>
                %418 = ascvec.reinterpret %417 : <f8E4M3FN> -> <ui32>
                %419 = ascvec.gather_reg %418, %394 : <ui32>, <ui32> -> <ui32>
                %420 = ascvec.reinterpret %419 : <ui32> -> <f8E4M3FN>
                %421 = ascvec.gather_reg %411, %394 : <ui32>, <ui32> -> <ui32>
                %422 = arith.divsi %arg24, %c8_i32_13 : i32
                %423 = arith.muli %422, %c1024_i32_12 overflow<nsw> : i32
                %424 = arith.remsi %arg24, %c8_i32_13 : i32
                %425 = arith.muli %424, %c32_i32_16 overflow<nsw> : i32
                %426 = arith.addi %423, %425 overflow<nsw> : i32
                %427 = arith.index_cast %426 : i32 to index
                %428 = arith.addi %427, %c214016 : index
                ascvec.storealign_pack %arg22[%428], %417, %390 {mode = "b32_to_b8"} : memref<?xf8E4M3FN, 1>, index, <f8E4M3FN>
                %429 = arith.addi %426, %c256_i32 overflow<nsw> : i32
                %430 = arith.index_cast %429 : i32 to index
                %431 = arith.addi %430, %c214016 : index
                ascvec.storealign_pack %arg22[%431], %420, %390 {mode = "b32_to_b8"} : memref<?xf8E4M3FN, 1>, index, <f8E4M3FN>
                %432 = arith.muli %arg24, %c4_i32_11 overflow<nsw> : i32
                %433 = ascvec.reinterpret %411 : <ui32> -> <i32>
                %434 = arith.index_cast %432 : i32 to index
                %435 = arith.addi %434, %c54528 : index
                ascvec.storealign_1st %arg23[%435], %433 : memref<?xi32, 1>, index, <i32>
                %436 = arith.addi %432, %c1_i32_10 overflow<nsw> : i32
                %437 = ascvec.reinterpret %421 : <ui32> -> <i32>
                %438 = arith.index_cast %436 : i32 to index
                %439 = arith.addi %438, %c54528 : index
                ascvec.storealign_1st %arg23[%439], %437 : memref<?xi32, 1>, index, <i32>
                %440 = arith.addi %395, %c192_i32 overflow<nsw> : i32
                %441 = arith.index_cast %440 : i32 to index
                %442 = arith.addi %441, %c33024 : index
                %443 = ascvec.loadalign %arg21[%442] : memref<?xf32, 1>, index -> <f32>
                %444 = ascvec.abs %443, %389 : <f32>
                %445 = ascvec.reduce_max %444, %390 : <f32>
                %446 = ascvec.duplicate %445, %389 : <f32> -> <f32>
                %447 = ascvec.gather_reg %444, %394 : <f32>, <ui32> -> <f32>
                %448 = ascvec.reduce_max %447, %390 : <f32>
                %449 = ascvec.duplicate %448, %389 : <f32> -> <f32>
                %450 = ascvec.select %446, %449, %390 : <f32>
                %451 = ascvec.mul_scalar %450, %cst_14, %389 : <f32>, f32
                %452 = ascvec.reinterpret %451 : <f32> -> <ui32>
                %453 = ascvec.add_scalar %452, %c8388607_i32, %389 : <ui32>, i32
                %454 = ascvec.shr %453, 23, %389 : <ui32> -> <ui32>
                %455 = ascvec.max_scalar %454, %c105_i32, %389 : <ui32>, i32
                %456 = ascvec.sub %412, %455, %389 : <ui32>
                %457 = ascvec.shl %456, 23, %389 : <ui32> -> <ui32>
                %458 = ascvec.reinterpret %457 : <ui32> -> <f32>
                %459 = ascvec.mul %443, %458, %389 : <f32>
                %460 = ascvec.cast %459, %389 {reg_layout = 0 : i32, saturate = true} : <f32> -> <f8E4M3FN>
                %461 = ascvec.reinterpret %460 : <f8E4M3FN> -> <ui32>
                %462 = ascvec.gather_reg %461, %394 : <ui32>, <ui32> -> <ui32>
                %463 = ascvec.reinterpret %462 : <ui32> -> <f8E4M3FN>
                %464 = ascvec.gather_reg %455, %394 : <ui32>, <ui32> -> <ui32>
                %465 = arith.addi %423, %c512_i32 overflow<nsw> : i32
                %466 = arith.addi %465, %425 overflow<nsw> : i32
                %467 = arith.index_cast %466 : i32 to index
                %468 = arith.addi %467, %c214016 : index
                ascvec.storealign_pack %arg22[%468], %460, %390 {mode = "b32_to_b8"} : memref<?xf8E4M3FN, 1>, index, <f8E4M3FN>
                %469 = arith.addi %466, %c256_i32 overflow<nsw> : i32
                %470 = arith.index_cast %469 : i32 to index
                %471 = arith.addi %470, %c214016 : index
                ascvec.storealign_pack %arg22[%471], %463, %390 {mode = "b32_to_b8"} : memref<?xf8E4M3FN, 1>, index, <f8E4M3FN>
                %472 = arith.addi %432, %c2_i32 overflow<nsw> : i32
                %473 = ascvec.reinterpret %455 : <ui32> -> <i32>
                %474 = arith.index_cast %472 : i32 to index
                %475 = arith.addi %474, %c54528 : index
                ascvec.storealign_1st %arg23[%475], %473 : memref<?xi32, 1>, index, <i32>
                %476 = arith.addi %432, %c3_i32 overflow<nsw> : i32
                %477 = ascvec.reinterpret %464 : <ui32> -> <i32>
                %478 = arith.index_cast %476 : i32 to index
                %479 = arith.addi %478, %c54528 : index
                ascvec.storealign_1st %arg23[%479], %477 : memref<?xi32, 1>, index, <i32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.vf(%227, %227 : memref<?xi32, 1>, memref<?xi32, 1>) {
            ^bb0(%arg21: memref<?xi32, 1>, %arg22: memref<?xi32, 1>):
              %c54976 = arith.constant 54976 : index
              %c54720 = arith.constant 54720 : index
              %c12_i32 = arith.constant 12 : i32
              %c8_i32_10 = arith.constant 8 : i32
              %c4_i32_11 = arith.constant 4 : i32
              %c54528 = arith.constant 54528 : index
              %c1_i32_12 = arith.constant 1 : i32
              %c7_i32 = arith.constant 7 : i32
              %c0_i32_13 = arith.constant 0 : i32
              %c0_14 = arith.constant 0 : index
              %c1_15 = arith.constant 1 : index
              scf.for %arg23 = %c0_14 to %c1_15 step %c1_15 {
                %389 = ascvec.create_mask : !ascvec.mask
                %390 = ascvec.create_mask pat_mode "PAT_VL32" : !ascvec.mask
                %391 = ascvec.arange %c0_i32_13 : i32 -> <ui32>
                %392 = ascvec.shr %391, 3, %389 : <ui32> -> <ui32>
                %393 = ascvec.duplicate_scalar_nomask %c7_i32 : i32 -> <ui32>
                %394 = ascvec.and %391, %393, %389 : <ui32>, <ui32> -> <ui32>
                %395 = ascvec.shr %394, 1, %389 : <ui32> -> <ui32>
                %396 = ascvec.duplicate_scalar_nomask %c1_i32_12 : i32 -> <ui32>
                %397 = ascvec.and %394, %396, %389 : <ui32>, <ui32> -> <ui32>
                %398 = ascvec.shl %392, 5, %389 : <ui32> -> <ui32>
                %399 = ascvec.shl %397, 4, %389 : <ui32> -> <ui32>
                %400 = ascvec.add %398, %399, %389 : <ui32>
                %401 = ascvec.add %400, %395, %389 : <ui32>
                %402 = ascvec.gather %arg21[%c54528], %401, %390 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                %403 = ascvec.reinterpret %402 : <i32> -> <ui32>
                %404 = ascvec.add_scalar %401, %c4_i32_11, %389 : <ui32>, i32
                %405 = ascvec.gather %arg21[%c54528], %404, %390 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                %406 = ascvec.reinterpret %405 : <i32> -> <ui32>
                %407 = ascvec.shl %406, 8, %389 : <ui32> -> <ui32>
                %408 = ascvec.or %403, %407, %389 : <ui32>, <ui32> -> <ui32>
                %409 = ascvec.add_scalar %401, %c8_i32_10, %389 : <ui32>, i32
                %410 = ascvec.gather %arg21[%c54528], %409, %390 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                %411 = ascvec.reinterpret %410 : <i32> -> <ui32>
                %412 = ascvec.shl %411, 16, %389 : <ui32> -> <ui32>
                %413 = ascvec.or %408, %412, %389 : <ui32>, <ui32> -> <ui32>
                %414 = ascvec.add_scalar %401, %c12_i32, %389 : <ui32>, i32
                %415 = ascvec.gather %arg21[%c54528], %414, %390 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                %416 = ascvec.reinterpret %415 : <i32> -> <ui32>
                %417 = ascvec.shl %416, 24, %389 : <ui32> -> <ui32>
                %418 = ascvec.or %413, %417, %389 : <ui32>, <ui32> -> <ui32>
                %419 = ascvec.reinterpret %418 : <ui32> -> <i32>
                %420 = ascvec.shl %391, 3, %389 : <ui32> -> <ui32>
                ascvec.scatter %arg22[%c54720], %419, %420, %390 : memref<?xi32, 1>, <i32>, <ui32>, !ascvec.mask
                %421 = ascvec.duplicate_scalar_nomask %c0_i32_13 : i32 -> <i32>
                ascvec.storealign %arg22[%c54976], %421, %389 : memref<?xi32, 1>, index, <i32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c5_i64, 0
            ascvec.get_buf mte3, %c5_i64, 0
            %280 = arith.addi %229, %c1_i64 : i64
            %281 = arith.muli %280, %c32_i64 : i64
            %282 = arith.addi %281, %191 : i64
            %283 = arith.muli %282, %c32_i64 : i64
            %284 = arith.divui %283, %c8_i64 : i64
            %285 = arith.addi %233, %284 : i64
            %286 = ascscalar.bind_gm_ptr %285 : memref<1xi32>
            %287 = ascscalar.bind_local_ptr %c218880_i64 : memref<1xi32, 1>
            ascvec.copy_ub2gm %286[%c0], %287[%c0], %c8_i32, %c4_i32, %c16_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
            %288 = arith.addi %239, %c1_i64 : i64
            %289 = arith.muli %288, %c32_i64 : i64
            %290 = arith.addi %289, %191 : i64
            %291 = arith.muli %290, %c32_i64 : i64
            %292 = arith.divui %291, %c8_i64 : i64
            %293 = arith.addi %233, %292 : i64
            %294 = ascscalar.bind_gm_ptr %293 : memref<1xi32>
            %295 = ascscalar.bind_local_ptr %c219136_i64 : memref<1xi32, 1>
            ascvec.copy_ub2gm %294[%c0], %295[%c0], %c8_i32, %c4_i32, %c16_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
            %296 = arith.addi %248, %c1_i64 : i64
            %297 = arith.muli %296, %c32_i64 : i64
            %298 = arith.addi %297, %191 : i64
            %299 = arith.muli %298, %c32_i64 : i64
            %300 = arith.divui %299, %c8_i64 : i64
            %301 = arith.addi %233, %300 : i64
            %302 = ascscalar.bind_gm_ptr %301 : memref<1xi32>
            %303 = ascscalar.bind_local_ptr %c219392_i64 : memref<1xi32, 1>
            ascvec.copy_ub2gm %302[%c0], %303[%c0], %c8_i32, %c4_i32, %c16_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
            %304 = arith.addi %257, %c1_i64 : i64
            %305 = arith.muli %304, %c32_i64 : i64
            %306 = arith.addi %305, %191 : i64
            %307 = arith.muli %306, %c32_i64 : i64
            %308 = arith.divui %307, %c8_i64 : i64
            %309 = arith.addi %233, %308 : i64
            %310 = ascscalar.bind_gm_ptr %309 : memref<1xi32>
            %311 = ascscalar.bind_local_ptr %c219648_i64 : memref<1xi32, 1>
            ascvec.copy_ub2gm %310[%c0], %311[%c0], %c8_i32, %c4_i32, %c16_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
            %312 = arith.muli %273, %c16384_i64 : i64
            %313 = arith.addi %312, %c1024_i64 : i64
            %314 = arith.muli %313, %c8_i64 : i64
            %315 = arith.divui %314, %c8_i64 : i64
            %316 = arith.addi %275, %315 : i64
            %317 = ascscalar.bind_gm_ptr %316 : memref<1xf8E4M3FN>
            %318 = ascscalar.bind_local_ptr %c214016_i64 : memref<1xf8E4M3FN, 1>
            ascvec.copy_ub2gm %317[%c0], %318[%c0], %c4_i32, %c1024_i32, %c4096_i64, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xf8E4M3FN>, memref<1xf8E4M3FN, 1>
            ascvec.release_buf mte3, %c5_i64, 0
            ascvec.get_buf v, %c5_i64, 0
            ascvec.sync_intra_arrive v, %c5_i64
            ascvec.release_buf v, %c5_i64, 0
            ascvec.get_buf v, %c6_i64, 0
            ascvec.sync_intra_wait v, %c4_i64
            scf.if %224 {
              %389 = arith.muli %210, %c2048_i64 overflow<nsw> : i64
              %390 = arith.index_cast %389 : i64 to index
              %391 = arith.addi %390, %c1922048 : index
              %392 = arith.divui %391, %c32 : index
              ascvec.vf(%225, %392, %225, %225 : memref<?xf32, 1>, index, memref<?xf32, 1>, memref<?xf32, 1>) {
              ^bb0(%arg21: memref<?xf32, 1>, %arg22: index, %arg23: memref<?xf32, 1>, %arg24: memref<?xf32, 1>):
                %c2_i32 = arith.constant 2 : i32
                %c32_i32_10 = arith.constant 32 : i32
                %c0_i32_11 = arith.constant 0 : i32
                %c33280 = arith.constant 33280 : index
                %c512_i32 = arith.constant 512 : i32
                %c64_i32 = arith.constant 64 : i32
                %c49408 = arith.constant 49408 : index
                %c128_i32_12 = arith.constant 128 : i32
                %c1_i32_13 = arith.constant 1 : i32
                %393 = ascvec.create_mask : !ascvec.mask
                scf.for %arg25 = %c0_i32_11 to %c32_i32_10 step %c2_i32  : i32 {
                  %394 = arith.index_cast %arg25 : i32 to index
                  %395 = arith.addi %arg22, %394 : index
                  %396 = ascvec.loadalign_brc %arg21[%395] : memref<?xf32, 1>, index -> <f32>
                  %397 = arith.addi %arg25, %c1_i32_13 overflow<nsw> : i32
                  %398 = arith.index_cast %397 : i32 to index
                  %399 = arith.addi %arg22, %398 : index
                  %400 = ascvec.loadalign_brc %arg21[%399] : memref<?xf32, 1>, index -> <f32>
                  %401 = arith.muli %arg25, %c128_i32_12 overflow<nsw> : i32
                  %402 = arith.index_cast %401 : i32 to index
                  %403 = arith.addi %402, %c49408 : index
                  %404 = ascvec.loadalign %arg23[%403] : memref<?xf32, 1>, index -> <f32>
                  %405 = arith.addi %401, %c64_i32 overflow<nsw> : i32
                  %406 = arith.index_cast %405 : i32 to index
                  %407 = arith.addi %406, %c49408 : index
                  %408 = ascvec.loadalign %arg23[%407] : memref<?xf32, 1>, index -> <f32>
                  %409 = arith.muli %397, %c128_i32_12 overflow<nsw> : i32
                  %410 = arith.index_cast %409 : i32 to index
                  %411 = arith.addi %410, %c49408 : index
                  %412 = ascvec.loadalign %arg23[%411] : memref<?xf32, 1>, index -> <f32>
                  %413 = arith.addi %409, %c64_i32 overflow<nsw> : i32
                  %414 = arith.index_cast %413 : i32 to index
                  %415 = arith.addi %414, %c49408 : index
                  %416 = ascvec.loadalign %arg23[%415] : memref<?xf32, 1>, index -> <f32>
                  %417 = arith.muli %arg25, %c512_i32 overflow<nsw> : i32
                  %418 = arith.index_cast %417 : i32 to index
                  %419 = arith.addi %418, %c33280 : index
                  %420 = ascvec.loadalign %arg24[%419] : memref<?xf32, 1>, index -> <f32>
                  %421 = ascvec.add %404, %420, %393 : <f32>
                  %422 = arith.addi %417, %c64_i32 overflow<nsw> : i32
                  %423 = arith.index_cast %422 : i32 to index
                  %424 = arith.addi %423, %c33280 : index
                  %425 = ascvec.loadalign %arg24[%424] : memref<?xf32, 1>, index -> <f32>
                  %426 = ascvec.add %408, %425, %393 : <f32>
                  %427 = arith.muli %397, %c512_i32 overflow<nsw> : i32
                  %428 = arith.index_cast %427 : i32 to index
                  %429 = arith.addi %428, %c33280 : index
                  %430 = ascvec.loadalign %arg24[%429] : memref<?xf32, 1>, index -> <f32>
                  %431 = ascvec.add %412, %430, %393 : <f32>
                  %432 = arith.addi %427, %c64_i32 overflow<nsw> : i32
                  %433 = arith.index_cast %432 : i32 to index
                  %434 = arith.addi %433, %c33280 : index
                  %435 = ascvec.loadalign %arg24[%434] : memref<?xf32, 1>, index -> <f32>
                  %436 = ascvec.add %416, %435, %393 : <f32>
                  %437 = ascvec.mul %421, %396, %393 : <f32>
                  ascvec.storealign %arg24[%419], %437, %393 : memref<?xf32, 1>, index, <f32>
                  %438 = ascvec.mul %426, %396, %393 : <f32>
                  ascvec.storealign %arg24[%424], %438, %393 : memref<?xf32, 1>, index, <f32>
                  %439 = ascvec.mul %431, %400, %393 : <f32>
                  ascvec.storealign %arg24[%429], %439, %393 : memref<?xf32, 1>, index, <f32>
                  %440 = ascvec.mul %436, %400, %393 : <f32>
                  ascvec.storealign %arg24[%434], %440, %393 : memref<?xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
            } else {
              %389 = arith.muli %210, %c2048_i64 overflow<nsw> : i64
              %390 = arith.index_cast %389 : i64 to index
              %391 = arith.addi %390, %c1922048 : index
              %392 = arith.divui %391, %c32 : index
              ascvec.vf(%225, %392, %225, %225 : memref<?xf32, 1>, index, memref<?xf32, 1>, memref<?xf32, 1>) {
              ^bb0(%arg21: memref<?xf32, 1>, %arg22: index, %arg23: memref<?xf32, 1>, %arg24: memref<?xf32, 1>):
                %c2_i32 = arith.constant 2 : i32
                %c32_i32_10 = arith.constant 32 : i32
                %c0_i32_11 = arith.constant 0 : i32
                %c33280 = arith.constant 33280 : index
                %c512_i32 = arith.constant 512 : i32
                %c64_i32 = arith.constant 64 : i32
                %c49408 = arith.constant 49408 : index
                %c128_i32_12 = arith.constant 128 : i32
                %c1_i32_13 = arith.constant 1 : i32
                %393 = ascvec.create_mask : !ascvec.mask
                scf.for %arg25 = %c0_i32_11 to %c32_i32_10 step %c2_i32  : i32 {
                  %394 = arith.index_cast %arg25 : i32 to index
                  %395 = arith.addi %arg22, %394 : index
                  %396 = ascvec.loadalign_brc %arg21[%395] : memref<?xf32, 1>, index -> <f32>
                  %397 = arith.addi %arg25, %c1_i32_13 overflow<nsw> : i32
                  %398 = arith.index_cast %397 : i32 to index
                  %399 = arith.addi %arg22, %398 : index
                  %400 = ascvec.loadalign_brc %arg21[%399] : memref<?xf32, 1>, index -> <f32>
                  %401 = arith.muli %arg25, %c128_i32_12 overflow<nsw> : i32
                  %402 = arith.index_cast %401 : i32 to index
                  %403 = arith.addi %402, %c49408 : index
                  %404 = ascvec.loadalign %arg23[%403] : memref<?xf32, 1>, index -> <f32>
                  %405 = arith.addi %401, %c64_i32 overflow<nsw> : i32
                  %406 = arith.index_cast %405 : i32 to index
                  %407 = arith.addi %406, %c49408 : index
                  %408 = ascvec.loadalign %arg23[%407] : memref<?xf32, 1>, index -> <f32>
                  %409 = arith.muli %397, %c128_i32_12 overflow<nsw> : i32
                  %410 = arith.index_cast %409 : i32 to index
                  %411 = arith.addi %410, %c49408 : index
                  %412 = ascvec.loadalign %arg23[%411] : memref<?xf32, 1>, index -> <f32>
                  %413 = arith.addi %409, %c64_i32 overflow<nsw> : i32
                  %414 = arith.index_cast %413 : i32 to index
                  %415 = arith.addi %414, %c49408 : index
                  %416 = ascvec.loadalign %arg23[%415] : memref<?xf32, 1>, index -> <f32>
                  %417 = arith.muli %arg25, %c512_i32 overflow<nsw> : i32
                  %418 = ascvec.mul %404, %396, %393 : <f32>
                  %419 = arith.index_cast %417 : i32 to index
                  %420 = arith.addi %419, %c33280 : index
                  ascvec.storealign %arg24[%420], %418, %393 : memref<?xf32, 1>, index, <f32>
                  %421 = arith.addi %417, %c64_i32 overflow<nsw> : i32
                  %422 = ascvec.mul %408, %396, %393 : <f32>
                  %423 = arith.index_cast %421 : i32 to index
                  %424 = arith.addi %423, %c33280 : index
                  ascvec.storealign %arg24[%424], %422, %393 : memref<?xf32, 1>, index, <f32>
                  %425 = arith.muli %397, %c512_i32 overflow<nsw> : i32
                  %426 = ascvec.mul %412, %400, %393 : <f32>
                  %427 = arith.index_cast %425 : i32 to index
                  %428 = arith.addi %427, %c33280 : index
                  ascvec.storealign %arg24[%428], %426, %393 : memref<?xf32, 1>, index, <f32>
                  %429 = arith.addi %425, %c64_i32 overflow<nsw> : i32
                  %430 = ascvec.mul %416, %400, %393 : <f32>
                  %431 = arith.index_cast %429 : i32 to index
                  %432 = arith.addi %431, %c33280 : index
                  ascvec.storealign %arg24[%432], %430, %393 : memref<?xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
            }
            ascvec.vf(%225, %226, %227 : memref<?xf32, 1>, memref<?xf8E4M3FN, 1>, memref<?xi32, 1>) {
            ^bb0(%arg21: memref<?xf32, 1>, %arg22: memref<?xf8E4M3FN, 1>, %arg23: memref<?xi32, 1>):
              %c3_i32 = arith.constant 3 : i32
              %c320_i32 = arith.constant 320 : i32
              %c2_i32 = arith.constant 2 : i32
              %c1_i32_10 = arith.constant 1 : i32
              %c50432 = arith.constant 50432 : index
              %c4_i32_11 = arith.constant 4 : i32
              %c197632 = arith.constant 197632 : index
              %c1024_i32_12 = arith.constant 1024 : i32
              %c8_i32_13 = arith.constant 8 : i32
              %c254_i32 = arith.constant 254 : i32
              %c105_i32 = arith.constant 105 : i32
              %c8388607_i32 = arith.constant 8388607 : i32
              %cst_14 = arith.constant 0.00223214296 : f32
              %c33024 = arith.constant 33024 : index
              %c256_i32 = arith.constant 256 : i32
              %c512_i32 = arith.constant 512 : i32
              %c32_i32_15 = arith.constant 32 : i32
              %c31_i32 = arith.constant 31 : i32
              %c0_i32_16 = arith.constant 0 : i32
              %389 = ascvec.create_mask : !ascvec.mask
              %390 = ascvec.create_mask pat_mode "PAT_VL32" : !ascvec.mask
              %391 = ascvec.arange %c0_i32_16 : i32 -> <ui32>
              %392 = ascvec.duplicate_scalar_nomask %c31_i32 : i32 -> <ui32>
              %393 = ascvec.and %391, %392, %389 : <ui32>, <ui32> -> <ui32>
              %394 = ascvec.add_scalar %393, %c32_i32_15, %389 : <ui32>, i32
              ascvec.mem_bar vst_vld
              scf.for %arg24 = %c0_i32_16 to %c32_i32_15 step %c1_i32_10  : i32 {
                %395 = arith.muli %arg24, %c512_i32 overflow<nsw> : i32
                %396 = arith.addi %395, %c256_i32 overflow<nsw> : i32
                %397 = arith.index_cast %396 : i32 to index
                %398 = arith.addi %397, %c33024 : index
                %399 = ascvec.loadalign %arg21[%398] : memref<?xf32, 1>, index -> <f32>
                %400 = ascvec.abs %399, %389 : <f32>
                %401 = ascvec.reduce_max %400, %390 : <f32>
                %402 = ascvec.duplicate %401, %389 : <f32> -> <f32>
                %403 = ascvec.gather_reg %400, %394 : <f32>, <ui32> -> <f32>
                %404 = ascvec.reduce_max %403, %390 : <f32>
                %405 = ascvec.duplicate %404, %389 : <f32> -> <f32>
                %406 = ascvec.select %402, %405, %390 : <f32>
                %407 = ascvec.mul_scalar %406, %cst_14, %389 : <f32>, f32
                %408 = ascvec.reinterpret %407 : <f32> -> <ui32>
                %409 = ascvec.add_scalar %408, %c8388607_i32, %389 : <ui32>, i32
                %410 = ascvec.shr %409, 23, %389 : <ui32> -> <ui32>
                %411 = ascvec.max_scalar %410, %c105_i32, %389 : <ui32>, i32
                %412 = ascvec.duplicate_scalar_nomask %c254_i32 : i32 -> <ui32>
                %413 = ascvec.sub %412, %411, %389 : <ui32>
                %414 = ascvec.shl %413, 23, %389 : <ui32> -> <ui32>
                %415 = ascvec.reinterpret %414 : <ui32> -> <f32>
                %416 = ascvec.mul %399, %415, %389 : <f32>
                %417 = ascvec.cast %416, %389 {reg_layout = 0 : i32, saturate = true} : <f32> -> <f8E4M3FN>
                %418 = ascvec.reinterpret %417 : <f8E4M3FN> -> <ui32>
                %419 = ascvec.gather_reg %418, %394 : <ui32>, <ui32> -> <ui32>
                %420 = ascvec.reinterpret %419 : <ui32> -> <f8E4M3FN>
                %421 = ascvec.gather_reg %411, %394 : <ui32>, <ui32> -> <ui32>
                %422 = arith.divsi %arg24, %c8_i32_13 : i32
                %423 = arith.muli %422, %c1024_i32_12 overflow<nsw> : i32
                %424 = arith.remsi %arg24, %c8_i32_13 : i32
                %425 = arith.muli %424, %c32_i32_15 overflow<nsw> : i32
                %426 = arith.addi %423, %425 overflow<nsw> : i32
                %427 = arith.index_cast %426 : i32 to index
                %428 = arith.addi %427, %c197632 : index
                ascvec.storealign_pack %arg22[%428], %417, %390 {mode = "b32_to_b8"} : memref<?xf8E4M3FN, 1>, index, <f8E4M3FN>
                %429 = arith.addi %426, %c256_i32 overflow<nsw> : i32
                %430 = arith.index_cast %429 : i32 to index
                %431 = arith.addi %430, %c197632 : index
                ascvec.storealign_pack %arg22[%431], %420, %390 {mode = "b32_to_b8"} : memref<?xf8E4M3FN, 1>, index, <f8E4M3FN>
                %432 = arith.muli %arg24, %c4_i32_11 overflow<nsw> : i32
                %433 = ascvec.reinterpret %411 : <ui32> -> <i32>
                %434 = arith.index_cast %432 : i32 to index
                %435 = arith.addi %434, %c50432 : index
                ascvec.storealign_1st %arg23[%435], %433 : memref<?xi32, 1>, index, <i32>
                %436 = arith.addi %432, %c1_i32_10 overflow<nsw> : i32
                %437 = ascvec.reinterpret %421 : <ui32> -> <i32>
                %438 = arith.index_cast %436 : i32 to index
                %439 = arith.addi %438, %c50432 : index
                ascvec.storealign_1st %arg23[%439], %437 : memref<?xi32, 1>, index, <i32>
                %440 = arith.addi %395, %c320_i32 overflow<nsw> : i32
                %441 = arith.index_cast %440 : i32 to index
                %442 = arith.addi %441, %c33024 : index
                %443 = ascvec.loadalign %arg21[%442] : memref<?xf32, 1>, index -> <f32>
                %444 = ascvec.abs %443, %389 : <f32>
                %445 = ascvec.reduce_max %444, %390 : <f32>
                %446 = ascvec.duplicate %445, %389 : <f32> -> <f32>
                %447 = ascvec.gather_reg %444, %394 : <f32>, <ui32> -> <f32>
                %448 = ascvec.reduce_max %447, %390 : <f32>
                %449 = ascvec.duplicate %448, %389 : <f32> -> <f32>
                %450 = ascvec.select %446, %449, %390 : <f32>
                %451 = ascvec.mul_scalar %450, %cst_14, %389 : <f32>, f32
                %452 = ascvec.reinterpret %451 : <f32> -> <ui32>
                %453 = ascvec.add_scalar %452, %c8388607_i32, %389 : <ui32>, i32
                %454 = ascvec.shr %453, 23, %389 : <ui32> -> <ui32>
                %455 = ascvec.max_scalar %454, %c105_i32, %389 : <ui32>, i32
                %456 = ascvec.sub %412, %455, %389 : <ui32>
                %457 = ascvec.shl %456, 23, %389 : <ui32> -> <ui32>
                %458 = ascvec.reinterpret %457 : <ui32> -> <f32>
                %459 = ascvec.mul %443, %458, %389 : <f32>
                %460 = ascvec.cast %459, %389 {reg_layout = 0 : i32, saturate = true} : <f32> -> <f8E4M3FN>
                %461 = ascvec.reinterpret %460 : <f8E4M3FN> -> <ui32>
                %462 = ascvec.gather_reg %461, %394 : <ui32>, <ui32> -> <ui32>
                %463 = ascvec.reinterpret %462 : <ui32> -> <f8E4M3FN>
                %464 = ascvec.gather_reg %455, %394 : <ui32>, <ui32> -> <ui32>
                %465 = arith.addi %423, %c512_i32 overflow<nsw> : i32
                %466 = arith.addi %465, %425 overflow<nsw> : i32
                %467 = arith.index_cast %466 : i32 to index
                %468 = arith.addi %467, %c197632 : index
                ascvec.storealign_pack %arg22[%468], %460, %390 {mode = "b32_to_b8"} : memref<?xf8E4M3FN, 1>, index, <f8E4M3FN>
                %469 = arith.addi %466, %c256_i32 overflow<nsw> : i32
                %470 = arith.index_cast %469 : i32 to index
                %471 = arith.addi %470, %c197632 : index
                ascvec.storealign_pack %arg22[%471], %463, %390 {mode = "b32_to_b8"} : memref<?xf8E4M3FN, 1>, index, <f8E4M3FN>
                %472 = arith.addi %432, %c2_i32 overflow<nsw> : i32
                %473 = ascvec.reinterpret %455 : <ui32> -> <i32>
                %474 = arith.index_cast %472 : i32 to index
                %475 = arith.addi %474, %c50432 : index
                ascvec.storealign_1st %arg23[%475], %473 : memref<?xi32, 1>, index, <i32>
                %476 = arith.addi %432, %c3_i32 overflow<nsw> : i32
                %477 = ascvec.reinterpret %464 : <ui32> -> <i32>
                %478 = arith.index_cast %476 : i32 to index
                %479 = arith.addi %478, %c50432 : index
                ascvec.storealign_1st %arg23[%479], %477 : memref<?xi32, 1>, index, <i32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.vf(%227, %227 : memref<?xi32, 1>, memref<?xi32, 1>) {
            ^bb0(%arg21: memref<?xi32, 1>, %arg22: memref<?xi32, 1>):
              %c50880 = arith.constant 50880 : index
              %c50624 = arith.constant 50624 : index
              %c12_i32 = arith.constant 12 : i32
              %c8_i32_10 = arith.constant 8 : i32
              %c4_i32_11 = arith.constant 4 : i32
              %c50432 = arith.constant 50432 : index
              %c1_i32_12 = arith.constant 1 : i32
              %c7_i32 = arith.constant 7 : i32
              %c0_i32_13 = arith.constant 0 : i32
              %c0_14 = arith.constant 0 : index
              %c1_15 = arith.constant 1 : index
              scf.for %arg23 = %c0_14 to %c1_15 step %c1_15 {
                %389 = ascvec.create_mask : !ascvec.mask
                %390 = ascvec.create_mask pat_mode "PAT_VL32" : !ascvec.mask
                %391 = ascvec.arange %c0_i32_13 : i32 -> <ui32>
                %392 = ascvec.shr %391, 3, %389 : <ui32> -> <ui32>
                %393 = ascvec.duplicate_scalar_nomask %c7_i32 : i32 -> <ui32>
                %394 = ascvec.and %391, %393, %389 : <ui32>, <ui32> -> <ui32>
                %395 = ascvec.shr %394, 1, %389 : <ui32> -> <ui32>
                %396 = ascvec.duplicate_scalar_nomask %c1_i32_12 : i32 -> <ui32>
                %397 = ascvec.and %394, %396, %389 : <ui32>, <ui32> -> <ui32>
                %398 = ascvec.shl %392, 5, %389 : <ui32> -> <ui32>
                %399 = ascvec.shl %397, 4, %389 : <ui32> -> <ui32>
                %400 = ascvec.add %398, %399, %389 : <ui32>
                %401 = ascvec.add %400, %395, %389 : <ui32>
                %402 = ascvec.gather %arg21[%c50432], %401, %390 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                %403 = ascvec.reinterpret %402 : <i32> -> <ui32>
                %404 = ascvec.add_scalar %401, %c4_i32_11, %389 : <ui32>, i32
                %405 = ascvec.gather %arg21[%c50432], %404, %390 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                %406 = ascvec.reinterpret %405 : <i32> -> <ui32>
                %407 = ascvec.shl %406, 8, %389 : <ui32> -> <ui32>
                %408 = ascvec.or %403, %407, %389 : <ui32>, <ui32> -> <ui32>
                %409 = ascvec.add_scalar %401, %c8_i32_10, %389 : <ui32>, i32
                %410 = ascvec.gather %arg21[%c50432], %409, %390 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                %411 = ascvec.reinterpret %410 : <i32> -> <ui32>
                %412 = ascvec.shl %411, 16, %389 : <ui32> -> <ui32>
                %413 = ascvec.or %408, %412, %389 : <ui32>, <ui32> -> <ui32>
                %414 = ascvec.add_scalar %401, %c12_i32, %389 : <ui32>, i32
                %415 = ascvec.gather %arg21[%c50432], %414, %390 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                %416 = ascvec.reinterpret %415 : <i32> -> <ui32>
                %417 = ascvec.shl %416, 24, %389 : <ui32> -> <ui32>
                %418 = ascvec.or %413, %417, %389 : <ui32>, <ui32> -> <ui32>
                %419 = ascvec.reinterpret %418 : <ui32> -> <i32>
                %420 = ascvec.shl %391, 3, %389 : <ui32> -> <ui32>
                ascvec.scatter %arg22[%c50624], %419, %420, %390 : memref<?xi32, 1>, <i32>, <ui32>, !ascvec.mask
                %421 = ascvec.duplicate_scalar_nomask %c0_i32_13 : i32 -> <i32>
                ascvec.storealign %arg22[%c50880], %421, %389 : memref<?xi32, 1>, index, <i32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c6_i64, 0
            ascvec.get_buf mte3, %c6_i64, 0
            %319 = arith.addi %229, %c2_i64 : i64
            %320 = arith.muli %319, %c32_i64 : i64
            %321 = arith.addi %320, %191 : i64
            %322 = arith.muli %321, %c32_i64 : i64
            %323 = arith.divui %322, %c8_i64 : i64
            %324 = arith.addi %233, %323 : i64
            %325 = ascscalar.bind_gm_ptr %324 : memref<1xi32>
            ascvec.copy_ub2gm %325[%c0], %237[%c0], %c8_i32, %c4_i32, %c16_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
            %326 = arith.addi %239, %c2_i64 : i64
            %327 = arith.muli %326, %c32_i64 : i64
            %328 = arith.addi %327, %191 : i64
            %329 = arith.muli %328, %c32_i64 : i64
            %330 = arith.divui %329, %c8_i64 : i64
            %331 = arith.addi %233, %330 : i64
            %332 = ascscalar.bind_gm_ptr %331 : memref<1xi32>
            ascvec.copy_ub2gm %332[%c0], %246[%c0], %c8_i32, %c4_i32, %c16_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
            %333 = arith.addi %248, %c2_i64 : i64
            %334 = arith.muli %333, %c32_i64 : i64
            %335 = arith.addi %334, %191 : i64
            %336 = arith.muli %335, %c32_i64 : i64
            %337 = arith.divui %336, %c8_i64 : i64
            %338 = arith.addi %233, %337 : i64
            %339 = ascscalar.bind_gm_ptr %338 : memref<1xi32>
            ascvec.copy_ub2gm %339[%c0], %255[%c0], %c8_i32, %c4_i32, %c16_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
            %340 = arith.addi %257, %c2_i64 : i64
            %341 = arith.muli %340, %c32_i64 : i64
            %342 = arith.addi %341, %191 : i64
            %343 = arith.muli %342, %c32_i64 : i64
            %344 = arith.divui %343, %c8_i64 : i64
            %345 = arith.addi %233, %344 : i64
            %346 = ascscalar.bind_gm_ptr %345 : memref<1xi32>
            ascvec.copy_ub2gm %346[%c0], %264[%c0], %c8_i32, %c4_i32, %c16_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
            %347 = arith.addi %312, %c2048_i64 : i64
            %348 = arith.muli %347, %c8_i64 : i64
            %349 = arith.divui %348, %c8_i64 : i64
            %350 = arith.addi %275, %349 : i64
            %351 = ascscalar.bind_gm_ptr %350 : memref<1xf8E4M3FN>
            ascvec.copy_ub2gm %351[%c0], %279[%c0], %c4_i32, %c1024_i32, %c4096_i64, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xf8E4M3FN>, memref<1xf8E4M3FN, 1>
            ascvec.release_buf mte3, %c6_i64, 0
            ascvec.get_buf v, %c6_i64, 0
            ascvec.sync_intra_arrive v, %c4_i64
            ascvec.release_buf v, %c6_i64, 0
            ascvec.get_buf v, %c7_i64, 0
            ascvec.sync_intra_wait v, %c5_i64
            scf.if %224 {
              %389 = arith.muli %210, %c2048_i64 overflow<nsw> : i64
              %390 = arith.index_cast %389 : i64 to index
              %391 = arith.addi %390, %c1922048 : index
              %392 = arith.divui %391, %c32 : index
              ascvec.vf(%225, %392, %225, %225 : memref<?xf32, 1>, index, memref<?xf32, 1>, memref<?xf32, 1>) {
              ^bb0(%arg21: memref<?xf32, 1>, %arg22: index, %arg23: memref<?xf32, 1>, %arg24: memref<?xf32, 1>):
                %c2_i32 = arith.constant 2 : i32
                %c32_i32_10 = arith.constant 32 : i32
                %c0_i32_11 = arith.constant 0 : i32
                %c33408 = arith.constant 33408 : index
                %c512_i32 = arith.constant 512 : i32
                %c64_i32 = arith.constant 64 : i32
                %c53504 = arith.constant 53504 : index
                %c128_i32_12 = arith.constant 128 : i32
                %c1_i32_13 = arith.constant 1 : i32
                %393 = ascvec.create_mask : !ascvec.mask
                scf.for %arg25 = %c0_i32_11 to %c32_i32_10 step %c2_i32  : i32 {
                  %394 = arith.index_cast %arg25 : i32 to index
                  %395 = arith.addi %arg22, %394 : index
                  %396 = ascvec.loadalign_brc %arg21[%395] : memref<?xf32, 1>, index -> <f32>
                  %397 = arith.addi %arg25, %c1_i32_13 overflow<nsw> : i32
                  %398 = arith.index_cast %397 : i32 to index
                  %399 = arith.addi %arg22, %398 : index
                  %400 = ascvec.loadalign_brc %arg21[%399] : memref<?xf32, 1>, index -> <f32>
                  %401 = arith.muli %arg25, %c128_i32_12 overflow<nsw> : i32
                  %402 = arith.index_cast %401 : i32 to index
                  %403 = arith.addi %402, %c53504 : index
                  %404 = ascvec.loadalign %arg23[%403] : memref<?xf32, 1>, index -> <f32>
                  %405 = arith.addi %401, %c64_i32 overflow<nsw> : i32
                  %406 = arith.index_cast %405 : i32 to index
                  %407 = arith.addi %406, %c53504 : index
                  %408 = ascvec.loadalign %arg23[%407] : memref<?xf32, 1>, index -> <f32>
                  %409 = arith.muli %397, %c128_i32_12 overflow<nsw> : i32
                  %410 = arith.index_cast %409 : i32 to index
                  %411 = arith.addi %410, %c53504 : index
                  %412 = ascvec.loadalign %arg23[%411] : memref<?xf32, 1>, index -> <f32>
                  %413 = arith.addi %409, %c64_i32 overflow<nsw> : i32
                  %414 = arith.index_cast %413 : i32 to index
                  %415 = arith.addi %414, %c53504 : index
                  %416 = ascvec.loadalign %arg23[%415] : memref<?xf32, 1>, index -> <f32>
                  %417 = arith.muli %arg25, %c512_i32 overflow<nsw> : i32
                  %418 = arith.index_cast %417 : i32 to index
                  %419 = arith.addi %418, %c33408 : index
                  %420 = ascvec.loadalign %arg24[%419] : memref<?xf32, 1>, index -> <f32>
                  %421 = ascvec.add %404, %420, %393 : <f32>
                  %422 = arith.addi %417, %c64_i32 overflow<nsw> : i32
                  %423 = arith.index_cast %422 : i32 to index
                  %424 = arith.addi %423, %c33408 : index
                  %425 = ascvec.loadalign %arg24[%424] : memref<?xf32, 1>, index -> <f32>
                  %426 = ascvec.add %408, %425, %393 : <f32>
                  %427 = arith.muli %397, %c512_i32 overflow<nsw> : i32
                  %428 = arith.index_cast %427 : i32 to index
                  %429 = arith.addi %428, %c33408 : index
                  %430 = ascvec.loadalign %arg24[%429] : memref<?xf32, 1>, index -> <f32>
                  %431 = ascvec.add %412, %430, %393 : <f32>
                  %432 = arith.addi %427, %c64_i32 overflow<nsw> : i32
                  %433 = arith.index_cast %432 : i32 to index
                  %434 = arith.addi %433, %c33408 : index
                  %435 = ascvec.loadalign %arg24[%434] : memref<?xf32, 1>, index -> <f32>
                  %436 = ascvec.add %416, %435, %393 : <f32>
                  %437 = ascvec.mul %421, %396, %393 : <f32>
                  ascvec.storealign %arg24[%419], %437, %393 : memref<?xf32, 1>, index, <f32>
                  %438 = ascvec.mul %426, %396, %393 : <f32>
                  ascvec.storealign %arg24[%424], %438, %393 : memref<?xf32, 1>, index, <f32>
                  %439 = ascvec.mul %431, %400, %393 : <f32>
                  ascvec.storealign %arg24[%429], %439, %393 : memref<?xf32, 1>, index, <f32>
                  %440 = ascvec.mul %436, %400, %393 : <f32>
                  ascvec.storealign %arg24[%434], %440, %393 : memref<?xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
            } else {
              %389 = arith.muli %210, %c2048_i64 overflow<nsw> : i64
              %390 = arith.index_cast %389 : i64 to index
              %391 = arith.addi %390, %c1922048 : index
              %392 = arith.divui %391, %c32 : index
              ascvec.vf(%225, %392, %225, %225 : memref<?xf32, 1>, index, memref<?xf32, 1>, memref<?xf32, 1>) {
              ^bb0(%arg21: memref<?xf32, 1>, %arg22: index, %arg23: memref<?xf32, 1>, %arg24: memref<?xf32, 1>):
                %c2_i32 = arith.constant 2 : i32
                %c32_i32_10 = arith.constant 32 : i32
                %c0_i32_11 = arith.constant 0 : i32
                %c33408 = arith.constant 33408 : index
                %c512_i32 = arith.constant 512 : i32
                %c64_i32 = arith.constant 64 : i32
                %c53504 = arith.constant 53504 : index
                %c128_i32_12 = arith.constant 128 : i32
                %c1_i32_13 = arith.constant 1 : i32
                %393 = ascvec.create_mask : !ascvec.mask
                scf.for %arg25 = %c0_i32_11 to %c32_i32_10 step %c2_i32  : i32 {
                  %394 = arith.index_cast %arg25 : i32 to index
                  %395 = arith.addi %arg22, %394 : index
                  %396 = ascvec.loadalign_brc %arg21[%395] : memref<?xf32, 1>, index -> <f32>
                  %397 = arith.addi %arg25, %c1_i32_13 overflow<nsw> : i32
                  %398 = arith.index_cast %397 : i32 to index
                  %399 = arith.addi %arg22, %398 : index
                  %400 = ascvec.loadalign_brc %arg21[%399] : memref<?xf32, 1>, index -> <f32>
                  %401 = arith.muli %arg25, %c128_i32_12 overflow<nsw> : i32
                  %402 = arith.index_cast %401 : i32 to index
                  %403 = arith.addi %402, %c53504 : index
                  %404 = ascvec.loadalign %arg23[%403] : memref<?xf32, 1>, index -> <f32>
                  %405 = arith.addi %401, %c64_i32 overflow<nsw> : i32
                  %406 = arith.index_cast %405 : i32 to index
                  %407 = arith.addi %406, %c53504 : index
                  %408 = ascvec.loadalign %arg23[%407] : memref<?xf32, 1>, index -> <f32>
                  %409 = arith.muli %397, %c128_i32_12 overflow<nsw> : i32
                  %410 = arith.index_cast %409 : i32 to index
                  %411 = arith.addi %410, %c53504 : index
                  %412 = ascvec.loadalign %arg23[%411] : memref<?xf32, 1>, index -> <f32>
                  %413 = arith.addi %409, %c64_i32 overflow<nsw> : i32
                  %414 = arith.index_cast %413 : i32 to index
                  %415 = arith.addi %414, %c53504 : index
                  %416 = ascvec.loadalign %arg23[%415] : memref<?xf32, 1>, index -> <f32>
                  %417 = arith.muli %arg25, %c512_i32 overflow<nsw> : i32
                  %418 = ascvec.mul %404, %396, %393 : <f32>
                  %419 = arith.index_cast %417 : i32 to index
                  %420 = arith.addi %419, %c33408 : index
                  ascvec.storealign %arg24[%420], %418, %393 : memref<?xf32, 1>, index, <f32>
                  %421 = arith.addi %417, %c64_i32 overflow<nsw> : i32
                  %422 = ascvec.mul %408, %396, %393 : <f32>
                  %423 = arith.index_cast %421 : i32 to index
                  %424 = arith.addi %423, %c33408 : index
                  ascvec.storealign %arg24[%424], %422, %393 : memref<?xf32, 1>, index, <f32>
                  %425 = arith.muli %397, %c512_i32 overflow<nsw> : i32
                  %426 = ascvec.mul %412, %400, %393 : <f32>
                  %427 = arith.index_cast %425 : i32 to index
                  %428 = arith.addi %427, %c33408 : index
                  ascvec.storealign %arg24[%428], %426, %393 : memref<?xf32, 1>, index, <f32>
                  %429 = arith.addi %425, %c64_i32 overflow<nsw> : i32
                  %430 = ascvec.mul %416, %400, %393 : <f32>
                  %431 = arith.index_cast %429 : i32 to index
                  %432 = arith.addi %431, %c33408 : index
                  ascvec.storealign %arg24[%432], %430, %393 : memref<?xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
            }
            %352 = arith.index_cast %191 : i64 to index
            %353 = memref.load %alloc_3[%352] : memref<4xi32>
            ascvec.release_buf v, %c7_i64, 0
            ascvec.get_buf mte2, %c7_i64, 0
            %354 = arith.extsi %353 : i32 to i64
            %355 = arith.muli %354, %c256_i64 : i64
            "ascvec.copy_gm2ub_plan"(%1, %alloc_4, %c0_i32, %c0_i32, %c218624_i64, %355, %c1_i64, %c256_i64, %c0_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 9472 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 253952 : i64}> : (memref<253952xui8, 1>, memref<2368xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %c7_i64, 0
            ascvec.get_buf v, %c7_i64, 0
            ascvec.vf(%225, %225, %226, %227 : memref<?xf32, 1>, memref<?xf32, 1>, memref<?xf8E4M3FN, 1>, memref<?xi32, 1>) {
            ^bb0(%arg21: memref<?xf32, 1>, %arg22: memref<?xf32, 1>, %arg23: memref<?xf8E4M3FN, 1>, %arg24: memref<?xi32, 1>):
              %c3_i32 = arith.constant 3 : i32
              %c2_i32 = arith.constant 2 : i32
              %c54528 = arith.constant 54528 : index
              %c4_i32_10 = arith.constant 4 : i32
              %c256_i32 = arith.constant 256 : i32
              %c214016 = arith.constant 214016 : index
              %c1024_i32_11 = arith.constant 1024 : i32
              %c8_i32_12 = arith.constant 8 : i32
              %c254_i32 = arith.constant 254 : i32
              %c105_i32 = arith.constant 105 : i32
              %c8388607_i32 = arith.constant 8388607 : i32
              %cst_13 = arith.constant 0.00223214296 : f32
              %c384_i32 = arith.constant 384 : i32
              %c33024 = arith.constant 33024 : index
              %c448_i32 = arith.constant 448 : i32
              %c512_i32 = arith.constant 512 : i32
              %cst_14 = arith.constant -1.000000e+00 : f32
              %c1_i32_15 = arith.constant 1 : i32
              %c54656 = arith.constant 54656 : index
              %c32_i32_16 = arith.constant 32 : i32
              %c31_i32 = arith.constant 31 : i32
              %c0_i32_17 = arith.constant 0 : i32
              %389 = ascvec.create_mask : !ascvec.mask
              %390 = ascvec.create_mask pat_mode "PAT_VL32" : !ascvec.mask
              %391 = ascvec.arange %c0_i32_17 : i32 -> <ui32>
              %392 = ascvec.duplicate_scalar_nomask %c31_i32 : i32 -> <ui32>
              %393 = ascvec.and %391, %392, %389 : <ui32>, <ui32> -> <ui32>
              %394 = ascvec.add_scalar %393, %c32_i32_16, %389 : <ui32>, i32
              %395 = ascvec.loadalign %arg21[%c54656] : memref<?xf32, 1>, index -> <f32>
              %396 = ascvec.shr %391, 1, %389 : <ui32> -> <ui32>
              %397 = ascvec.shl %396, 1, %389 : <ui32> -> <ui32>
              %398 = ascvec.add_scalar %397, %c1_i32_15, %389 : <ui32>, i32
              %399 = ascvec.duplicate_scalar_nomask %c1_i32_15 : i32 -> <ui32>
              %400 = ascvec.and %391, %399, %389 : <ui32>, <ui32> -> <ui32>
              %401 = ascvec.cmp_eq_scalar %400, %c0_i32_17, %389 : <ui32>, i32 -> !ascvec.mask
              %402 = ascvec.gather_reg %395, %396 : <f32>, <ui32> -> <f32>
              %403 = ascvec.add_scalar %396, %c32_i32_16, %389 : <ui32>, i32
              %404 = ascvec.gather_reg %395, %403 : <f32>, <ui32> -> <f32>
              %405 = ascvec.mul_scalar %404, %cst_14, %389 : <f32>, f32
              %406 = ascvec.mul_scalar %405, %cst_14, %389 : <f32>, f32
              scf.for %arg25 = %c0_i32_17 to %c32_i32_16 step %c1_i32_15  : i32 {
                %407 = arith.muli %arg25, %c512_i32 overflow<nsw> : i32
                %408 = arith.addi %407, %c448_i32 overflow<nsw> : i32
                %409 = arith.index_cast %408 : i32 to index
                %410 = arith.addi %409, %c33024 : index
                %411 = ascvec.loadalign %arg22[%410] : memref<?xf32, 1>, index -> <f32>
                %412 = ascvec.gather_reg %411, %397 : <f32>, <ui32> -> <f32>
                %413 = ascvec.gather_reg %411, %398 : <f32>, <ui32> -> <f32>
                %414 = ascvec.mul %413, %406, %389 : <f32>
                %415 = ascvec.mul %413, %402, %389 : <f32>
                %416 = ascvec.copy %412, %389 : <f32>
                %417 = ascvec.madd_two_address %416, %402, %414, %389 : <f32>
                %418 = ascvec.madd_two_address %412, %405, %415, %389 : <f32>
                %419 = ascvec.select %417, %418, %401 : <f32>
                ascvec.storealign %arg22[%410], %419, %389 : memref<?xf32, 1>, index, <f32>
              }
              ascvec.mem_bar vst_vld
              scf.for %arg25 = %c0_i32_17 to %c32_i32_16 step %c1_i32_15  : i32 {
                %407 = arith.muli %arg25, %c512_i32 overflow<nsw> : i32
                %408 = arith.addi %407, %c384_i32 overflow<nsw> : i32
                %409 = arith.index_cast %408 : i32 to index
                %410 = arith.addi %409, %c33024 : index
                %411 = ascvec.loadalign %arg22[%410] : memref<?xf32, 1>, index -> <f32>
                %412 = ascvec.abs %411, %389 : <f32>
                %413 = ascvec.reduce_max %412, %390 : <f32>
                %414 = ascvec.duplicate %413, %389 : <f32> -> <f32>
                %415 = ascvec.gather_reg %412, %394 : <f32>, <ui32> -> <f32>
                %416 = ascvec.reduce_max %415, %390 : <f32>
                %417 = ascvec.duplicate %416, %389 : <f32> -> <f32>
                %418 = ascvec.select %414, %417, %390 : <f32>
                %419 = ascvec.mul_scalar %418, %cst_13, %389 : <f32>, f32
                %420 = ascvec.reinterpret %419 : <f32> -> <ui32>
                %421 = ascvec.add_scalar %420, %c8388607_i32, %389 : <ui32>, i32
                %422 = ascvec.shr %421, 23, %389 : <ui32> -> <ui32>
                %423 = ascvec.max_scalar %422, %c105_i32, %389 : <ui32>, i32
                %424 = ascvec.duplicate_scalar_nomask %c254_i32 : i32 -> <ui32>
                %425 = ascvec.sub %424, %423, %389 : <ui32>
                %426 = ascvec.shl %425, 23, %389 : <ui32> -> <ui32>
                %427 = ascvec.reinterpret %426 : <ui32> -> <f32>
                %428 = ascvec.mul %411, %427, %389 : <f32>
                %429 = ascvec.cast %428, %389 {reg_layout = 0 : i32, saturate = true} : <f32> -> <f8E4M3FN>
                %430 = ascvec.reinterpret %429 : <f8E4M3FN> -> <ui32>
                %431 = ascvec.gather_reg %430, %394 : <ui32>, <ui32> -> <ui32>
                %432 = ascvec.reinterpret %431 : <ui32> -> <f8E4M3FN>
                %433 = ascvec.gather_reg %423, %394 : <ui32>, <ui32> -> <ui32>
                %434 = arith.divsi %arg25, %c8_i32_12 : i32
                %435 = arith.muli %434, %c1024_i32_11 overflow<nsw> : i32
                %436 = arith.remsi %arg25, %c8_i32_12 : i32
                %437 = arith.muli %436, %c32_i32_16 overflow<nsw> : i32
                %438 = arith.addi %435, %437 overflow<nsw> : i32
                %439 = arith.index_cast %438 : i32 to index
                %440 = arith.addi %439, %c214016 : index
                ascvec.storealign_pack %arg23[%440], %429, %390 {mode = "b32_to_b8"} : memref<?xf8E4M3FN, 1>, index, <f8E4M3FN>
                %441 = arith.addi %438, %c256_i32 overflow<nsw> : i32
                %442 = arith.index_cast %441 : i32 to index
                %443 = arith.addi %442, %c214016 : index
                ascvec.storealign_pack %arg23[%443], %432, %390 {mode = "b32_to_b8"} : memref<?xf8E4M3FN, 1>, index, <f8E4M3FN>
                %444 = arith.muli %arg25, %c4_i32_10 overflow<nsw> : i32
                %445 = ascvec.reinterpret %423 : <ui32> -> <i32>
                %446 = arith.index_cast %444 : i32 to index
                %447 = arith.addi %446, %c54528 : index
                ascvec.storealign_1st %arg24[%447], %445 : memref<?xi32, 1>, index, <i32>
                %448 = arith.addi %444, %c1_i32_15 overflow<nsw> : i32
                %449 = ascvec.reinterpret %433 : <ui32> -> <i32>
                %450 = arith.index_cast %448 : i32 to index
                %451 = arith.addi %450, %c54528 : index
                ascvec.storealign_1st %arg24[%451], %449 : memref<?xi32, 1>, index, <i32>
                %452 = arith.addi %407, %c448_i32 overflow<nsw> : i32
                %453 = arith.index_cast %452 : i32 to index
                %454 = arith.addi %453, %c33024 : index
                %455 = ascvec.loadalign %arg22[%454] : memref<?xf32, 1>, index -> <f32>
                %456 = ascvec.abs %455, %389 : <f32>
                %457 = ascvec.reduce_max %456, %390 : <f32>
                %458 = ascvec.duplicate %457, %389 : <f32> -> <f32>
                %459 = ascvec.gather_reg %456, %394 : <f32>, <ui32> -> <f32>
                %460 = ascvec.reduce_max %459, %390 : <f32>
                %461 = ascvec.duplicate %460, %389 : <f32> -> <f32>
                %462 = ascvec.select %458, %461, %390 : <f32>
                %463 = ascvec.mul_scalar %462, %cst_13, %389 : <f32>, f32
                %464 = ascvec.reinterpret %463 : <f32> -> <ui32>
                %465 = ascvec.add_scalar %464, %c8388607_i32, %389 : <ui32>, i32
                %466 = ascvec.shr %465, 23, %389 : <ui32> -> <ui32>
                %467 = ascvec.max_scalar %466, %c105_i32, %389 : <ui32>, i32
                %468 = ascvec.sub %424, %467, %389 : <ui32>
                %469 = ascvec.shl %468, 23, %389 : <ui32> -> <ui32>
                %470 = ascvec.reinterpret %469 : <ui32> -> <f32>
                %471 = ascvec.mul %455, %470, %389 : <f32>
                %472 = ascvec.cast %471, %389 {reg_layout = 0 : i32, saturate = true} : <f32> -> <f8E4M3FN>
                %473 = ascvec.reinterpret %472 : <f8E4M3FN> -> <ui32>
                %474 = ascvec.gather_reg %473, %394 : <ui32>, <ui32> -> <ui32>
                %475 = ascvec.reinterpret %474 : <ui32> -> <f8E4M3FN>
                %476 = ascvec.gather_reg %467, %394 : <ui32>, <ui32> -> <ui32>
                %477 = arith.addi %435, %c512_i32 overflow<nsw> : i32
                %478 = arith.addi %477, %437 overflow<nsw> : i32
                %479 = arith.index_cast %478 : i32 to index
                %480 = arith.addi %479, %c214016 : index
                ascvec.storealign_pack %arg23[%480], %472, %390 {mode = "b32_to_b8"} : memref<?xf8E4M3FN, 1>, index, <f8E4M3FN>
                %481 = arith.addi %478, %c256_i32 overflow<nsw> : i32
                %482 = arith.index_cast %481 : i32 to index
                %483 = arith.addi %482, %c214016 : index
                ascvec.storealign_pack %arg23[%483], %475, %390 {mode = "b32_to_b8"} : memref<?xf8E4M3FN, 1>, index, <f8E4M3FN>
                %484 = arith.addi %444, %c2_i32 overflow<nsw> : i32
                %485 = ascvec.reinterpret %467 : <ui32> -> <i32>
                %486 = arith.index_cast %484 : i32 to index
                %487 = arith.addi %486, %c54528 : index
                ascvec.storealign_1st %arg24[%487], %485 : memref<?xi32, 1>, index, <i32>
                %488 = arith.addi %444, %c3_i32 overflow<nsw> : i32
                %489 = ascvec.reinterpret %476 : <ui32> -> <i32>
                %490 = arith.index_cast %488 : i32 to index
                %491 = arith.addi %490, %c54528 : index
                ascvec.storealign_1st %arg24[%491], %489 : memref<?xi32, 1>, index, <i32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.vf(%227, %227 : memref<?xi32, 1>, memref<?xi32, 1>) {
            ^bb0(%arg21: memref<?xi32, 1>, %arg22: memref<?xi32, 1>):
              %c54976 = arith.constant 54976 : index
              %c54720 = arith.constant 54720 : index
              %c12_i32 = arith.constant 12 : i32
              %c8_i32_10 = arith.constant 8 : i32
              %c4_i32_11 = arith.constant 4 : i32
              %c54528 = arith.constant 54528 : index
              %c1_i32_12 = arith.constant 1 : i32
              %c7_i32 = arith.constant 7 : i32
              %c0_i32_13 = arith.constant 0 : i32
              %c0_14 = arith.constant 0 : index
              %c1_15 = arith.constant 1 : index
              scf.for %arg23 = %c0_14 to %c1_15 step %c1_15 {
                %389 = ascvec.create_mask : !ascvec.mask
                %390 = ascvec.create_mask pat_mode "PAT_VL32" : !ascvec.mask
                %391 = ascvec.arange %c0_i32_13 : i32 -> <ui32>
                %392 = ascvec.shr %391, 3, %389 : <ui32> -> <ui32>
                %393 = ascvec.duplicate_scalar_nomask %c7_i32 : i32 -> <ui32>
                %394 = ascvec.and %391, %393, %389 : <ui32>, <ui32> -> <ui32>
                %395 = ascvec.shr %394, 1, %389 : <ui32> -> <ui32>
                %396 = ascvec.duplicate_scalar_nomask %c1_i32_12 : i32 -> <ui32>
                %397 = ascvec.and %394, %396, %389 : <ui32>, <ui32> -> <ui32>
                %398 = ascvec.shl %392, 5, %389 : <ui32> -> <ui32>
                %399 = ascvec.shl %397, 4, %389 : <ui32> -> <ui32>
                %400 = ascvec.add %398, %399, %389 : <ui32>
                %401 = ascvec.add %400, %395, %389 : <ui32>
                %402 = ascvec.gather %arg21[%c54528], %401, %390 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                %403 = ascvec.reinterpret %402 : <i32> -> <ui32>
                %404 = ascvec.add_scalar %401, %c4_i32_11, %389 : <ui32>, i32
                %405 = ascvec.gather %arg21[%c54528], %404, %390 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                %406 = ascvec.reinterpret %405 : <i32> -> <ui32>
                %407 = ascvec.shl %406, 8, %389 : <ui32> -> <ui32>
                %408 = ascvec.or %403, %407, %389 : <ui32>, <ui32> -> <ui32>
                %409 = ascvec.add_scalar %401, %c8_i32_10, %389 : <ui32>, i32
                %410 = ascvec.gather %arg21[%c54528], %409, %390 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                %411 = ascvec.reinterpret %410 : <i32> -> <ui32>
                %412 = ascvec.shl %411, 16, %389 : <ui32> -> <ui32>
                %413 = ascvec.or %408, %412, %389 : <ui32>, <ui32> -> <ui32>
                %414 = ascvec.add_scalar %401, %c12_i32, %389 : <ui32>, i32
                %415 = ascvec.gather %arg21[%c54528], %414, %390 : memref<?xi32, 1>, <ui32>, !ascvec.mask -> <i32>
                %416 = ascvec.reinterpret %415 : <i32> -> <ui32>
                %417 = ascvec.shl %416, 24, %389 : <ui32> -> <ui32>
                %418 = ascvec.or %413, %417, %389 : <ui32>, <ui32> -> <ui32>
                %419 = ascvec.reinterpret %418 : <ui32> -> <i32>
                %420 = ascvec.shl %391, 3, %389 : <ui32> -> <ui32>
                ascvec.scatter %arg22[%c54720], %419, %420, %390 : memref<?xi32, 1>, <i32>, <ui32>, !ascvec.mask
                %421 = ascvec.duplicate_scalar_nomask %c0_i32_13 : i32 -> <i32>
                ascvec.storealign %arg22[%c54976], %421, %389 : memref<?xi32, 1>, index, <i32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c7_i64, 0
            ascvec.get_buf mte3, %c7_i64, 0
            %356 = arith.addi %229, %c3_i64 : i64
            %357 = arith.muli %356, %c32_i64 : i64
            %358 = arith.addi %357, %191 : i64
            %359 = arith.muli %358, %c32_i64 : i64
            %360 = arith.divui %359, %c8_i64 : i64
            %361 = arith.addi %233, %360 : i64
            %362 = ascscalar.bind_gm_ptr %361 : memref<1xi32>
            ascvec.copy_ub2gm %362[%c0], %287[%c0], %c8_i32, %c4_i32, %c16_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
            %363 = arith.addi %239, %c3_i64 : i64
            %364 = arith.muli %363, %c32_i64 : i64
            %365 = arith.addi %364, %191 : i64
            %366 = arith.muli %365, %c32_i64 : i64
            %367 = arith.divui %366, %c8_i64 : i64
            %368 = arith.addi %233, %367 : i64
            %369 = ascscalar.bind_gm_ptr %368 : memref<1xi32>
            ascvec.copy_ub2gm %369[%c0], %295[%c0], %c8_i32, %c4_i32, %c16_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
            %370 = arith.addi %248, %c3_i64 : i64
            %371 = arith.muli %370, %c32_i64 : i64
            %372 = arith.addi %371, %191 : i64
            %373 = arith.muli %372, %c32_i64 : i64
            %374 = arith.divui %373, %c8_i64 : i64
            %375 = arith.addi %233, %374 : i64
            %376 = ascscalar.bind_gm_ptr %375 : memref<1xi32>
            ascvec.copy_ub2gm %376[%c0], %303[%c0], %c8_i32, %c4_i32, %c16_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
            %377 = arith.addi %257, %c3_i64 : i64
            %378 = arith.muli %377, %c32_i64 : i64
            %379 = arith.addi %378, %191 : i64
            %380 = arith.muli %379, %c32_i64 : i64
            %381 = arith.divui %380, %c8_i64 : i64
            %382 = arith.addi %233, %381 : i64
            %383 = ascscalar.bind_gm_ptr %382 : memref<1xi32>
            ascvec.copy_ub2gm %383[%c0], %311[%c0], %c8_i32, %c4_i32, %c16_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xi32>, memref<1xi32, 1>
            %384 = arith.addi %312, %c3072_i64 : i64
            %385 = arith.muli %384, %c8_i64 : i64
            %386 = arith.divui %385, %c8_i64 : i64
            %387 = arith.addi %275, %386 : i64
            %388 = ascscalar.bind_gm_ptr %387 : memref<1xf8E4M3FN>
            ascvec.copy_ub2gm %388[%c0], %318[%c0], %c4_i32, %c1024_i32, %c4096_i64, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xf8E4M3FN>, memref<1xf8E4M3FN, 1>
            ascvec.release_buf mte3, %c7_i64, 0
            ascvec.get_buf v, %c7_i64, 0
            ascvec.sync_intra_arrive v, %c5_i64
            ascvec.release_buf v, %c7_i64, 0
          }
        }
        scf.if %110 {
          %202 = arith.cmpi eq, %arg19, %c0_i64 : i64
          scf.if %202 {
            scf.if %178 {
              %359 = arith.cmpi ne, %175, %c0_i64 : i64
              scf.if %359 {
                ascvec.sync_wait {dst_pipe = 1 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
                %360 = arith.remsi %171, %c2_i64 : i64
                %361 = arith.cmpi ne, %360, %c0_i64 : i64
                %362 = arith.cmpi slt, %360, %c0_i64 : i64
                %363 = arith.andi %361, %362 : i1
                %364 = arith.select %363, %c2_i64, %c0_i64 : i64
                %365 = arith.addi %360, %364 overflow<nsw> : i64
                %366 = arith.muli %365, %c2_i64 overflow<nsw> : i64
                %367 = arith.remsi %175, %c2_i64 : i64
                %368 = arith.cmpi ne, %367, %c0_i64 : i64
                %369 = arith.cmpi slt, %367, %c0_i64 : i64
                %370 = arith.andi %368, %369 : i1
                %371 = arith.select %370, %c2_i64, %c0_i64 : i64
                %372 = arith.addi %367, %371 overflow<nsw> : i64
                %373 = arith.addi %366, %372 overflow<nsw> : i64
                %374 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xf32, 1>
                %375 = arith.index_cast %373 : i64 to index
                %376 = arith.addi %375, %c63008 : index
                %377 = memref.load %374[%376] : memref<?xf32, 1>
                %378 = arith.cmpf ogt, %377, %cst : f32
                %379 = arith.extui %378 : i1 to i32
                %380 = arith.remsi %171, %c8_i64 : i64
                %381 = arith.cmpi ne, %380, %c0_i64 : i64
                %382 = arith.cmpi slt, %380, %c0_i64 : i64
                %383 = arith.andi %381, %382 : i1
                %384 = arith.select %383, %c8_i64, %c0_i64 : i64
                %385 = arith.addi %380, %384 overflow<nsw> : i64
                %386 = arith.remsi %175, %c8_i64 : i64
                %387 = arith.cmpi ne, %386, %c0_i64 : i64
                %388 = arith.cmpi slt, %386, %c0_i64 : i64
                %389 = arith.andi %387, %388 : i1
                %390 = arith.select %389, %c8_i64, %c0_i64 : i64
                %391 = arith.addi %386, %390 overflow<nsw> : i64
                %392 = cannir.scalar_storage_cast %379 : i32 -> ui32
                %393 = ascscalar.bitcast_memref %5 byte_offset = %c0_i64 : memref<3072xui8, 9> -> memref<?xui32, 9>
                %394 = arith.index_cast %385 : i64 to index
                %395 = arith.index_cast %391 : i64 to index
                %396 = arith.index_cast %0 : i64 to index
                %397 = arith.muli %394, %c16 : index
                %398 = arith.muli %395, %c2 : index
                %399 = arith.addi %397, %398 : index
                %400 = arith.addi %399, %396 : index
                %c0_10 = arith.constant 0 : index
                %c1_11 = arith.constant 1 : index
                scf.for %arg21 = %c0_10 to %c1_11 step %c1_11 {
                  ascvec.ssbuf_store %392, %393[%400] : ui32, memref<?xui32, 9>
                }
                ascvec.sync_intra_arrive s, %c2_i64
              }
            }
            %203 = arith.remsi %arg20, %c3_i64 : i64
            %204 = arith.cmpi ne, %203, %c0_i64 : i64
            %205 = arith.cmpi slt, %203, %c0_i64 : i64
            %206 = arith.andi %204, %205 : i1
            %207 = arith.select %206, %c3_i64, %c0_i64 : i64
            %208 = arith.addi %203, %207 overflow<nsw> : i64
            %209 = arith.cmpi eq, %208, %c1_i64 : i64
            %210 = arith.select %209, %c262144_i64, %c0_i64 : i64
            %211 = arith.cmpi eq, %208, %c2_i64 : i64
            %212 = arith.select %211, %c524288_i64, %210 : i64
            %213 = arith.addi %208, %c8_i64 overflow<nsw> : i64
            ascvec.get_buf mte2, %213, 0
            ascvec.get_buf s, %c1_i64, 0
            %214 = arith.muli %0, %c16_i64 : i64
            %215 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xui64, 1>
            %216 = arith.index_cast %214 : i64 to index
            %217 = arith.addi %216, %c30544 : index
            %218 = memref.load %215[%217] : memref<?xui64, 1>
            %219 = cannir.scalar_storage_cast %218 : ui64 -> i64
            %220 = arith.addi %216, %c30864 : index
            %221 = memref.load %215[%220] : memref<?xui64, 1>
            %222 = cannir.scalar_storage_cast %221 : ui64 -> i64
            %223 = arith.cmpi ne, %219, %c32768_i64 : i64
            scf.if %223 {
              %359 = arith.divui %212, %c8_i64 : i64
              %360 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%360, %alloc_0, %c1_i32, %c0_i32, %359, %219, %c2_i64, %c1024_i64, %222, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %224 = arith.addi %214, %c1_i64 : i64
            %225 = arith.index_cast %224 : i64 to index
            %226 = arith.addi %225, %c30544 : index
            %227 = memref.load %215[%226] : memref<?xui64, 1>
            %228 = cannir.scalar_storage_cast %227 : ui64 -> i64
            %229 = arith.addi %225, %c30864 : index
            %230 = memref.load %215[%229] : memref<?xui64, 1>
            %231 = cannir.scalar_storage_cast %230 : ui64 -> i64
            %232 = arith.cmpi ne, %228, %c32768_i64 : i64
            scf.if %232 {
              %359 = arith.divui %212, %c8_i64 : i64
              %360 = arith.addi %359, %c2048_i64 overflow<nsw> : i64
              %361 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%361, %alloc_0, %c1_i32, %c0_i32, %360, %228, %c2_i64, %c1024_i64, %231, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %233 = arith.addi %214, %c2_i64 : i64
            %234 = arith.index_cast %233 : i64 to index
            %235 = arith.addi %234, %c30544 : index
            %236 = memref.load %215[%235] : memref<?xui64, 1>
            %237 = cannir.scalar_storage_cast %236 : ui64 -> i64
            %238 = arith.addi %234, %c30864 : index
            %239 = memref.load %215[%238] : memref<?xui64, 1>
            %240 = cannir.scalar_storage_cast %239 : ui64 -> i64
            %241 = arith.cmpi ne, %237, %c32768_i64 : i64
            scf.if %241 {
              %359 = arith.divui %212, %c8_i64 : i64
              %360 = arith.addi %359, %c4096_i64 overflow<nsw> : i64
              %361 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%361, %alloc_0, %c1_i32, %c0_i32, %360, %237, %c2_i64, %c1024_i64, %240, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %242 = arith.addi %214, %c3_i64 : i64
            %243 = arith.index_cast %242 : i64 to index
            %244 = arith.addi %243, %c30544 : index
            %245 = memref.load %215[%244] : memref<?xui64, 1>
            %246 = cannir.scalar_storage_cast %245 : ui64 -> i64
            %247 = arith.addi %243, %c30864 : index
            %248 = memref.load %215[%247] : memref<?xui64, 1>
            %249 = cannir.scalar_storage_cast %248 : ui64 -> i64
            %250 = arith.cmpi ne, %246, %c32768_i64 : i64
            scf.if %250 {
              %359 = arith.divui %212, %c8_i64 : i64
              %360 = arith.addi %359, %c6144_i64 overflow<nsw> : i64
              %361 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%361, %alloc_0, %c1_i32, %c0_i32, %360, %246, %c2_i64, %c1024_i64, %249, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %251 = arith.addi %214, %c4_i64 : i64
            %252 = arith.index_cast %251 : i64 to index
            %253 = arith.addi %252, %c30544 : index
            %254 = memref.load %215[%253] : memref<?xui64, 1>
            %255 = cannir.scalar_storage_cast %254 : ui64 -> i64
            %256 = arith.addi %252, %c30864 : index
            %257 = memref.load %215[%256] : memref<?xui64, 1>
            %258 = cannir.scalar_storage_cast %257 : ui64 -> i64
            %259 = arith.cmpi ne, %255, %c32768_i64 : i64
            scf.if %259 {
              %359 = arith.divui %212, %c8_i64 : i64
              %360 = arith.addi %359, %c8192_i64 overflow<nsw> : i64
              %361 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%361, %alloc_0, %c1_i32, %c0_i32, %360, %255, %c2_i64, %c1024_i64, %258, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %260 = arith.addi %214, %c5_i64 : i64
            %261 = arith.index_cast %260 : i64 to index
            %262 = arith.addi %261, %c30544 : index
            %263 = memref.load %215[%262] : memref<?xui64, 1>
            %264 = cannir.scalar_storage_cast %263 : ui64 -> i64
            %265 = arith.addi %261, %c30864 : index
            %266 = memref.load %215[%265] : memref<?xui64, 1>
            %267 = cannir.scalar_storage_cast %266 : ui64 -> i64
            %268 = arith.cmpi ne, %264, %c32768_i64 : i64
            scf.if %268 {
              %359 = arith.divui %212, %c8_i64 : i64
              %360 = arith.addi %359, %c10240_i64 overflow<nsw> : i64
              %361 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%361, %alloc_0, %c1_i32, %c0_i32, %360, %264, %c2_i64, %c1024_i64, %267, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %269 = arith.addi %214, %c6_i64 : i64
            %270 = arith.index_cast %269 : i64 to index
            %271 = arith.addi %270, %c30544 : index
            %272 = memref.load %215[%271] : memref<?xui64, 1>
            %273 = cannir.scalar_storage_cast %272 : ui64 -> i64
            %274 = arith.addi %270, %c30864 : index
            %275 = memref.load %215[%274] : memref<?xui64, 1>
            %276 = cannir.scalar_storage_cast %275 : ui64 -> i64
            %277 = arith.cmpi ne, %273, %c32768_i64 : i64
            scf.if %277 {
              %359 = arith.divui %212, %c8_i64 : i64
              %360 = arith.addi %359, %c12288_i64 overflow<nsw> : i64
              %361 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%361, %alloc_0, %c1_i32, %c0_i32, %360, %273, %c2_i64, %c1024_i64, %276, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %278 = arith.addi %214, %c7_i64 : i64
            %279 = arith.index_cast %278 : i64 to index
            %280 = arith.addi %279, %c30544 : index
            %281 = memref.load %215[%280] : memref<?xui64, 1>
            %282 = cannir.scalar_storage_cast %281 : ui64 -> i64
            %283 = arith.addi %279, %c30864 : index
            %284 = memref.load %215[%283] : memref<?xui64, 1>
            %285 = cannir.scalar_storage_cast %284 : ui64 -> i64
            %286 = arith.cmpi ne, %282, %c32768_i64 : i64
            scf.if %286 {
              %359 = arith.divui %212, %c8_i64 : i64
              %360 = arith.addi %359, %c14336_i64 overflow<nsw> : i64
              %361 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%361, %alloc_0, %c1_i32, %c0_i32, %360, %282, %c2_i64, %c1024_i64, %285, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %287 = arith.addi %214, %c8_i64 : i64
            %288 = arith.index_cast %287 : i64 to index
            %289 = arith.addi %288, %c30544 : index
            %290 = memref.load %215[%289] : memref<?xui64, 1>
            %291 = cannir.scalar_storage_cast %290 : ui64 -> i64
            %292 = arith.addi %288, %c30864 : index
            %293 = memref.load %215[%292] : memref<?xui64, 1>
            %294 = cannir.scalar_storage_cast %293 : ui64 -> i64
            %295 = arith.cmpi ne, %291, %c32768_i64 : i64
            scf.if %295 {
              %359 = arith.divui %212, %c8_i64 : i64
              %360 = arith.addi %359, %c16384_i64 overflow<nsw> : i64
              %361 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%361, %alloc_0, %c1_i32, %c0_i32, %360, %291, %c2_i64, %c1024_i64, %294, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %296 = arith.addi %214, %c9_i64 : i64
            %297 = arith.index_cast %296 : i64 to index
            %298 = arith.addi %297, %c30544 : index
            %299 = memref.load %215[%298] : memref<?xui64, 1>
            %300 = cannir.scalar_storage_cast %299 : ui64 -> i64
            %301 = arith.addi %297, %c30864 : index
            %302 = memref.load %215[%301] : memref<?xui64, 1>
            %303 = cannir.scalar_storage_cast %302 : ui64 -> i64
            %304 = arith.cmpi ne, %300, %c32768_i64 : i64
            scf.if %304 {
              %359 = arith.divui %212, %c8_i64 : i64
              %360 = arith.addi %359, %c18432_i64 overflow<nsw> : i64
              %361 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%361, %alloc_0, %c1_i32, %c0_i32, %360, %300, %c2_i64, %c1024_i64, %303, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %305 = arith.addi %214, %c10_i64 : i64
            %306 = arith.index_cast %305 : i64 to index
            %307 = arith.addi %306, %c30544 : index
            %308 = memref.load %215[%307] : memref<?xui64, 1>
            %309 = cannir.scalar_storage_cast %308 : ui64 -> i64
            %310 = arith.addi %306, %c30864 : index
            %311 = memref.load %215[%310] : memref<?xui64, 1>
            %312 = cannir.scalar_storage_cast %311 : ui64 -> i64
            %313 = arith.cmpi ne, %309, %c32768_i64 : i64
            scf.if %313 {
              %359 = arith.divui %212, %c8_i64 : i64
              %360 = arith.addi %359, %c20480_i64 overflow<nsw> : i64
              %361 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%361, %alloc_0, %c1_i32, %c0_i32, %360, %309, %c2_i64, %c1024_i64, %312, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %314 = arith.addi %214, %c11_i64 : i64
            %315 = arith.index_cast %314 : i64 to index
            %316 = arith.addi %315, %c30544 : index
            %317 = memref.load %215[%316] : memref<?xui64, 1>
            %318 = cannir.scalar_storage_cast %317 : ui64 -> i64
            %319 = arith.addi %315, %c30864 : index
            %320 = memref.load %215[%319] : memref<?xui64, 1>
            %321 = cannir.scalar_storage_cast %320 : ui64 -> i64
            %322 = arith.cmpi ne, %318, %c32768_i64 : i64
            scf.if %322 {
              %359 = arith.divui %212, %c8_i64 : i64
              %360 = arith.addi %359, %c22528_i64 overflow<nsw> : i64
              %361 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%361, %alloc_0, %c1_i32, %c0_i32, %360, %318, %c2_i64, %c1024_i64, %321, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %323 = arith.addi %214, %c12_i64 : i64
            %324 = arith.index_cast %323 : i64 to index
            %325 = arith.addi %324, %c30544 : index
            %326 = memref.load %215[%325] : memref<?xui64, 1>
            %327 = cannir.scalar_storage_cast %326 : ui64 -> i64
            %328 = arith.addi %324, %c30864 : index
            %329 = memref.load %215[%328] : memref<?xui64, 1>
            %330 = cannir.scalar_storage_cast %329 : ui64 -> i64
            %331 = arith.cmpi ne, %327, %c32768_i64 : i64
            scf.if %331 {
              %359 = arith.divui %212, %c8_i64 : i64
              %360 = arith.addi %359, %c24576_i64 overflow<nsw> : i64
              %361 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%361, %alloc_0, %c1_i32, %c0_i32, %360, %327, %c2_i64, %c1024_i64, %330, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %332 = arith.addi %214, %c13_i64 : i64
            %333 = arith.index_cast %332 : i64 to index
            %334 = arith.addi %333, %c30544 : index
            %335 = memref.load %215[%334] : memref<?xui64, 1>
            %336 = cannir.scalar_storage_cast %335 : ui64 -> i64
            %337 = arith.addi %333, %c30864 : index
            %338 = memref.load %215[%337] : memref<?xui64, 1>
            %339 = cannir.scalar_storage_cast %338 : ui64 -> i64
            %340 = arith.cmpi ne, %336, %c32768_i64 : i64
            scf.if %340 {
              %359 = arith.divui %212, %c8_i64 : i64
              %360 = arith.addi %359, %c26624_i64 overflow<nsw> : i64
              %361 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%361, %alloc_0, %c1_i32, %c0_i32, %360, %336, %c2_i64, %c1024_i64, %339, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %341 = arith.addi %214, %c14_i64 : i64
            %342 = arith.index_cast %341 : i64 to index
            %343 = arith.addi %342, %c30544 : index
            %344 = memref.load %215[%343] : memref<?xui64, 1>
            %345 = cannir.scalar_storage_cast %344 : ui64 -> i64
            %346 = arith.addi %342, %c30864 : index
            %347 = memref.load %215[%346] : memref<?xui64, 1>
            %348 = cannir.scalar_storage_cast %347 : ui64 -> i64
            %349 = arith.cmpi ne, %345, %c32768_i64 : i64
            scf.if %349 {
              %359 = arith.divui %212, %c8_i64 : i64
              %360 = arith.addi %359, %c28672_i64 overflow<nsw> : i64
              %361 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%361, %alloc_0, %c1_i32, %c0_i32, %360, %345, %c2_i64, %c1024_i64, %348, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %350 = arith.addi %214, %c15_i64 : i64
            %351 = arith.index_cast %350 : i64 to index
            %352 = arith.addi %351, %c30544 : index
            %353 = memref.load %215[%352] : memref<?xui64, 1>
            %354 = cannir.scalar_storage_cast %353 : ui64 -> i64
            %355 = arith.addi %351, %c30864 : index
            %356 = memref.load %215[%355] : memref<?xui64, 1>
            %357 = cannir.scalar_storage_cast %356 : ui64 -> i64
            %358 = arith.cmpi ne, %354, %c32768_i64 : i64
            scf.if %358 {
              %359 = arith.divui %212, %c8_i64 : i64
              %360 = arith.addi %359, %c30720_i64 overflow<nsw> : i64
              %361 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%361, %alloc_0, %c1_i32, %c0_i32, %360, %354, %c2_i64, %c1024_i64, %357, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            ascvec.release_buf s, %c1_i64, 0
            ascvec.release_buf mte2, %213, 0
          } else {
            %203 = arith.remsi %arg20, %c3_i64 : i64
            %204 = arith.cmpi ne, %203, %c0_i64 : i64
            %205 = arith.cmpi slt, %203, %c0_i64 : i64
            %206 = arith.andi %204, %205 : i1
            %207 = arith.select %206, %c3_i64, %c0_i64 : i64
            %208 = arith.addi %203, %207 overflow<nsw> : i64
            %209 = arith.cmpi eq, %208, %c1_i64 : i64
            %210 = arith.select %209, %c262144_i64, %c0_i64 : i64
            %211 = arith.cmpi eq, %208, %c2_i64 : i64
            %212 = arith.select %211, %c524288_i64, %210 : i64
            %213 = arith.addi %208, %c8_i64 overflow<nsw> : i64
            ascvec.get_buf mte2, %213, 0
            ascvec.get_buf s, %c1_i64, 0
            %214 = arith.muli %arg19, %c32_i64 : i64
            %215 = arith.muli %0, %c16_i64 : i64
            %216 = arith.addi %214, %215 : i64
            %217 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xui64, 1>
            %218 = arith.index_cast %216 : i64 to index
            %219 = arith.addi %218, %c30544 : index
            %220 = memref.load %217[%219] : memref<?xui64, 1>
            %221 = cannir.scalar_storage_cast %220 : ui64 -> i64
            %222 = arith.addi %218, %c30864 : index
            %223 = memref.load %217[%222] : memref<?xui64, 1>
            %224 = cannir.scalar_storage_cast %223 : ui64 -> i64
            %225 = arith.cmpi ne, %221, %c32768_i64 : i64
            scf.if %225 {
              %361 = arith.divui %212, %c8_i64 : i64
              %362 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%362, %alloc_0, %c1_i32, %c0_i32, %361, %221, %c2_i64, %c1024_i64, %224, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %226 = arith.addi %216, %c1_i64 : i64
            %227 = arith.index_cast %226 : i64 to index
            %228 = arith.addi %227, %c30544 : index
            %229 = memref.load %217[%228] : memref<?xui64, 1>
            %230 = cannir.scalar_storage_cast %229 : ui64 -> i64
            %231 = arith.addi %227, %c30864 : index
            %232 = memref.load %217[%231] : memref<?xui64, 1>
            %233 = cannir.scalar_storage_cast %232 : ui64 -> i64
            %234 = arith.cmpi ne, %230, %c32768_i64 : i64
            scf.if %234 {
              %361 = arith.divui %212, %c8_i64 : i64
              %362 = arith.addi %361, %c2048_i64 overflow<nsw> : i64
              %363 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%363, %alloc_0, %c1_i32, %c0_i32, %362, %230, %c2_i64, %c1024_i64, %233, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %235 = arith.addi %216, %c2_i64 : i64
            %236 = arith.index_cast %235 : i64 to index
            %237 = arith.addi %236, %c30544 : index
            %238 = memref.load %217[%237] : memref<?xui64, 1>
            %239 = cannir.scalar_storage_cast %238 : ui64 -> i64
            %240 = arith.addi %236, %c30864 : index
            %241 = memref.load %217[%240] : memref<?xui64, 1>
            %242 = cannir.scalar_storage_cast %241 : ui64 -> i64
            %243 = arith.cmpi ne, %239, %c32768_i64 : i64
            scf.if %243 {
              %361 = arith.divui %212, %c8_i64 : i64
              %362 = arith.addi %361, %c4096_i64 overflow<nsw> : i64
              %363 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%363, %alloc_0, %c1_i32, %c0_i32, %362, %239, %c2_i64, %c1024_i64, %242, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %244 = arith.addi %216, %c3_i64 : i64
            %245 = arith.index_cast %244 : i64 to index
            %246 = arith.addi %245, %c30544 : index
            %247 = memref.load %217[%246] : memref<?xui64, 1>
            %248 = cannir.scalar_storage_cast %247 : ui64 -> i64
            %249 = arith.addi %245, %c30864 : index
            %250 = memref.load %217[%249] : memref<?xui64, 1>
            %251 = cannir.scalar_storage_cast %250 : ui64 -> i64
            %252 = arith.cmpi ne, %248, %c32768_i64 : i64
            scf.if %252 {
              %361 = arith.divui %212, %c8_i64 : i64
              %362 = arith.addi %361, %c6144_i64 overflow<nsw> : i64
              %363 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%363, %alloc_0, %c1_i32, %c0_i32, %362, %248, %c2_i64, %c1024_i64, %251, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %253 = arith.addi %216, %c4_i64 : i64
            %254 = arith.index_cast %253 : i64 to index
            %255 = arith.addi %254, %c30544 : index
            %256 = memref.load %217[%255] : memref<?xui64, 1>
            %257 = cannir.scalar_storage_cast %256 : ui64 -> i64
            %258 = arith.addi %254, %c30864 : index
            %259 = memref.load %217[%258] : memref<?xui64, 1>
            %260 = cannir.scalar_storage_cast %259 : ui64 -> i64
            %261 = arith.cmpi ne, %257, %c32768_i64 : i64
            scf.if %261 {
              %361 = arith.divui %212, %c8_i64 : i64
              %362 = arith.addi %361, %c8192_i64 overflow<nsw> : i64
              %363 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%363, %alloc_0, %c1_i32, %c0_i32, %362, %257, %c2_i64, %c1024_i64, %260, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %262 = arith.addi %216, %c5_i64 : i64
            %263 = arith.index_cast %262 : i64 to index
            %264 = arith.addi %263, %c30544 : index
            %265 = memref.load %217[%264] : memref<?xui64, 1>
            %266 = cannir.scalar_storage_cast %265 : ui64 -> i64
            %267 = arith.addi %263, %c30864 : index
            %268 = memref.load %217[%267] : memref<?xui64, 1>
            %269 = cannir.scalar_storage_cast %268 : ui64 -> i64
            %270 = arith.cmpi ne, %266, %c32768_i64 : i64
            scf.if %270 {
              %361 = arith.divui %212, %c8_i64 : i64
              %362 = arith.addi %361, %c10240_i64 overflow<nsw> : i64
              %363 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%363, %alloc_0, %c1_i32, %c0_i32, %362, %266, %c2_i64, %c1024_i64, %269, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %271 = arith.addi %216, %c6_i64 : i64
            %272 = arith.index_cast %271 : i64 to index
            %273 = arith.addi %272, %c30544 : index
            %274 = memref.load %217[%273] : memref<?xui64, 1>
            %275 = cannir.scalar_storage_cast %274 : ui64 -> i64
            %276 = arith.addi %272, %c30864 : index
            %277 = memref.load %217[%276] : memref<?xui64, 1>
            %278 = cannir.scalar_storage_cast %277 : ui64 -> i64
            %279 = arith.cmpi ne, %275, %c32768_i64 : i64
            scf.if %279 {
              %361 = arith.divui %212, %c8_i64 : i64
              %362 = arith.addi %361, %c12288_i64 overflow<nsw> : i64
              %363 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%363, %alloc_0, %c1_i32, %c0_i32, %362, %275, %c2_i64, %c1024_i64, %278, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %280 = arith.addi %216, %c7_i64 : i64
            %281 = arith.index_cast %280 : i64 to index
            %282 = arith.addi %281, %c30544 : index
            %283 = memref.load %217[%282] : memref<?xui64, 1>
            %284 = cannir.scalar_storage_cast %283 : ui64 -> i64
            %285 = arith.addi %281, %c30864 : index
            %286 = memref.load %217[%285] : memref<?xui64, 1>
            %287 = cannir.scalar_storage_cast %286 : ui64 -> i64
            %288 = arith.cmpi ne, %284, %c32768_i64 : i64
            scf.if %288 {
              %361 = arith.divui %212, %c8_i64 : i64
              %362 = arith.addi %361, %c14336_i64 overflow<nsw> : i64
              %363 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%363, %alloc_0, %c1_i32, %c0_i32, %362, %284, %c2_i64, %c1024_i64, %287, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %289 = arith.addi %216, %c8_i64 : i64
            %290 = arith.index_cast %289 : i64 to index
            %291 = arith.addi %290, %c30544 : index
            %292 = memref.load %217[%291] : memref<?xui64, 1>
            %293 = cannir.scalar_storage_cast %292 : ui64 -> i64
            %294 = arith.addi %290, %c30864 : index
            %295 = memref.load %217[%294] : memref<?xui64, 1>
            %296 = cannir.scalar_storage_cast %295 : ui64 -> i64
            %297 = arith.cmpi ne, %293, %c32768_i64 : i64
            scf.if %297 {
              %361 = arith.divui %212, %c8_i64 : i64
              %362 = arith.addi %361, %c16384_i64 overflow<nsw> : i64
              %363 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%363, %alloc_0, %c1_i32, %c0_i32, %362, %293, %c2_i64, %c1024_i64, %296, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %298 = arith.addi %216, %c9_i64 : i64
            %299 = arith.index_cast %298 : i64 to index
            %300 = arith.addi %299, %c30544 : index
            %301 = memref.load %217[%300] : memref<?xui64, 1>
            %302 = cannir.scalar_storage_cast %301 : ui64 -> i64
            %303 = arith.addi %299, %c30864 : index
            %304 = memref.load %217[%303] : memref<?xui64, 1>
            %305 = cannir.scalar_storage_cast %304 : ui64 -> i64
            %306 = arith.cmpi ne, %302, %c32768_i64 : i64
            scf.if %306 {
              %361 = arith.divui %212, %c8_i64 : i64
              %362 = arith.addi %361, %c18432_i64 overflow<nsw> : i64
              %363 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%363, %alloc_0, %c1_i32, %c0_i32, %362, %302, %c2_i64, %c1024_i64, %305, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %307 = arith.addi %216, %c10_i64 : i64
            %308 = arith.index_cast %307 : i64 to index
            %309 = arith.addi %308, %c30544 : index
            %310 = memref.load %217[%309] : memref<?xui64, 1>
            %311 = cannir.scalar_storage_cast %310 : ui64 -> i64
            %312 = arith.addi %308, %c30864 : index
            %313 = memref.load %217[%312] : memref<?xui64, 1>
            %314 = cannir.scalar_storage_cast %313 : ui64 -> i64
            %315 = arith.cmpi ne, %311, %c32768_i64 : i64
            scf.if %315 {
              %361 = arith.divui %212, %c8_i64 : i64
              %362 = arith.addi %361, %c20480_i64 overflow<nsw> : i64
              %363 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%363, %alloc_0, %c1_i32, %c0_i32, %362, %311, %c2_i64, %c1024_i64, %314, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %316 = arith.addi %216, %c11_i64 : i64
            %317 = arith.index_cast %316 : i64 to index
            %318 = arith.addi %317, %c30544 : index
            %319 = memref.load %217[%318] : memref<?xui64, 1>
            %320 = cannir.scalar_storage_cast %319 : ui64 -> i64
            %321 = arith.addi %317, %c30864 : index
            %322 = memref.load %217[%321] : memref<?xui64, 1>
            %323 = cannir.scalar_storage_cast %322 : ui64 -> i64
            %324 = arith.cmpi ne, %320, %c32768_i64 : i64
            scf.if %324 {
              %361 = arith.divui %212, %c8_i64 : i64
              %362 = arith.addi %361, %c22528_i64 overflow<nsw> : i64
              %363 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%363, %alloc_0, %c1_i32, %c0_i32, %362, %320, %c2_i64, %c1024_i64, %323, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %325 = arith.addi %216, %c12_i64 : i64
            %326 = arith.index_cast %325 : i64 to index
            %327 = arith.addi %326, %c30544 : index
            %328 = memref.load %217[%327] : memref<?xui64, 1>
            %329 = cannir.scalar_storage_cast %328 : ui64 -> i64
            %330 = arith.addi %326, %c30864 : index
            %331 = memref.load %217[%330] : memref<?xui64, 1>
            %332 = cannir.scalar_storage_cast %331 : ui64 -> i64
            %333 = arith.cmpi ne, %329, %c32768_i64 : i64
            scf.if %333 {
              %361 = arith.divui %212, %c8_i64 : i64
              %362 = arith.addi %361, %c24576_i64 overflow<nsw> : i64
              %363 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%363, %alloc_0, %c1_i32, %c0_i32, %362, %329, %c2_i64, %c1024_i64, %332, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %334 = arith.addi %216, %c13_i64 : i64
            %335 = arith.index_cast %334 : i64 to index
            %336 = arith.addi %335, %c30544 : index
            %337 = memref.load %217[%336] : memref<?xui64, 1>
            %338 = cannir.scalar_storage_cast %337 : ui64 -> i64
            %339 = arith.addi %335, %c30864 : index
            %340 = memref.load %217[%339] : memref<?xui64, 1>
            %341 = cannir.scalar_storage_cast %340 : ui64 -> i64
            %342 = arith.cmpi ne, %338, %c32768_i64 : i64
            scf.if %342 {
              %361 = arith.divui %212, %c8_i64 : i64
              %362 = arith.addi %361, %c26624_i64 overflow<nsw> : i64
              %363 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%363, %alloc_0, %c1_i32, %c0_i32, %362, %338, %c2_i64, %c1024_i64, %341, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %343 = arith.addi %216, %c14_i64 : i64
            %344 = arith.index_cast %343 : i64 to index
            %345 = arith.addi %344, %c30544 : index
            %346 = memref.load %217[%345] : memref<?xui64, 1>
            %347 = cannir.scalar_storage_cast %346 : ui64 -> i64
            %348 = arith.addi %344, %c30864 : index
            %349 = memref.load %217[%348] : memref<?xui64, 1>
            %350 = cannir.scalar_storage_cast %349 : ui64 -> i64
            %351 = arith.cmpi ne, %347, %c32768_i64 : i64
            scf.if %351 {
              %361 = arith.divui %212, %c8_i64 : i64
              %362 = arith.addi %361, %c28672_i64 overflow<nsw> : i64
              %363 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%363, %alloc_0, %c1_i32, %c0_i32, %362, %347, %c2_i64, %c1024_i64, %350, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            %352 = arith.addi %216, %c15_i64 : i64
            %353 = arith.index_cast %352 : i64 to index
            %354 = arith.addi %353, %c30544 : index
            %355 = memref.load %217[%354] : memref<?xui64, 1>
            %356 = cannir.scalar_storage_cast %355 : ui64 -> i64
            %357 = arith.addi %353, %c30864 : index
            %358 = memref.load %217[%357] : memref<?xui64, 1>
            %359 = cannir.scalar_storage_cast %358 : ui64 -> i64
            %360 = arith.cmpi ne, %356, %c32768_i64 : i64
            scf.if %360 {
              %361 = arith.divui %212, %c8_i64 : i64
              %362 = arith.addi %361, %c30720_i64 overflow<nsw> : i64
              %363 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 1>
              "ascvec.copy_gm2ub_plan"(%363, %alloc_0, %c1_i32, %c0_i32, %362, %356, %c2_i64, %c1024_i64, %359, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 32768 : i64, pad_storage_bits = false, storage_dtype = ui8, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<16384xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            }
            ascvec.release_buf s, %c1_i64, 0
            ascvec.release_buf mte2, %213, 0
            scf.if %178 {
              %361 = arith.cmpi ne, %175, %c0_i64 : i64
              scf.if %361 {
                ascvec.sync_wait {dst_pipe = 1 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
                %362 = arith.remsi %171, %c2_i64 : i64
                %363 = arith.cmpi ne, %362, %c0_i64 : i64
                %364 = arith.cmpi slt, %362, %c0_i64 : i64
                %365 = arith.andi %363, %364 : i1
                %366 = arith.select %365, %c2_i64, %c0_i64 : i64
                %367 = arith.addi %362, %366 overflow<nsw> : i64
                %368 = arith.muli %367, %c2_i64 overflow<nsw> : i64
                %369 = arith.remsi %175, %c2_i64 : i64
                %370 = arith.cmpi ne, %369, %c0_i64 : i64
                %371 = arith.cmpi slt, %369, %c0_i64 : i64
                %372 = arith.andi %370, %371 : i1
                %373 = arith.select %372, %c2_i64, %c0_i64 : i64
                %374 = arith.addi %369, %373 overflow<nsw> : i64
                %375 = arith.addi %368, %374 overflow<nsw> : i64
                %376 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xf32, 1>
                %377 = arith.index_cast %375 : i64 to index
                %378 = arith.addi %377, %c63008 : index
                %379 = memref.load %376[%378] : memref<?xf32, 1>
                %380 = arith.cmpf ogt, %379, %cst : f32
                %381 = arith.extui %380 : i1 to i32
                %382 = arith.remsi %171, %c8_i64 : i64
                %383 = arith.cmpi ne, %382, %c0_i64 : i64
                %384 = arith.cmpi slt, %382, %c0_i64 : i64
                %385 = arith.andi %383, %384 : i1
                %386 = arith.select %385, %c8_i64, %c0_i64 : i64
                %387 = arith.addi %382, %386 overflow<nsw> : i64
                %388 = arith.remsi %175, %c8_i64 : i64
                %389 = arith.cmpi ne, %388, %c0_i64 : i64
                %390 = arith.cmpi slt, %388, %c0_i64 : i64
                %391 = arith.andi %389, %390 : i1
                %392 = arith.select %391, %c8_i64, %c0_i64 : i64
                %393 = arith.addi %388, %392 overflow<nsw> : i64
                %394 = cannir.scalar_storage_cast %381 : i32 -> ui32
                %395 = ascscalar.bitcast_memref %5 byte_offset = %c0_i64 : memref<3072xui8, 9> -> memref<?xui32, 9>
                %396 = arith.index_cast %387 : i64 to index
                %397 = arith.index_cast %393 : i64 to index
                %398 = arith.index_cast %0 : i64 to index
                %399 = arith.muli %396, %c16 : index
                %400 = arith.muli %397, %c2 : index
                %401 = arith.addi %399, %400 : index
                %402 = arith.addi %401, %398 : index
                %c0_10 = arith.constant 0 : index
                %c1_11 = arith.constant 1 : index
                scf.for %arg21 = %c0_10 to %c1_11 step %c1_11 {
                  ascvec.ssbuf_store %394, %395[%402] : ui32, memref<?xui32, 9>
                }
                ascvec.sync_intra_arrive s, %c2_i64
              }
            }
          }
        } else {
          scf.if %178 {
            %202 = arith.cmpi ne, %175, %c0_i64 : i64
            scf.if %202 {
              ascvec.sync_wait {dst_pipe = 1 : i32, event_id = 0 : i32, src_pipe = 2 : i32}
              %203 = arith.remsi %171, %c2_i64 : i64
              %204 = arith.cmpi ne, %203, %c0_i64 : i64
              %205 = arith.cmpi slt, %203, %c0_i64 : i64
              %206 = arith.andi %204, %205 : i1
              %207 = arith.select %206, %c2_i64, %c0_i64 : i64
              %208 = arith.addi %203, %207 overflow<nsw> : i64
              %209 = arith.muli %208, %c2_i64 overflow<nsw> : i64
              %210 = arith.remsi %175, %c2_i64 : i64
              %211 = arith.cmpi ne, %210, %c0_i64 : i64
              %212 = arith.cmpi slt, %210, %c0_i64 : i64
              %213 = arith.andi %211, %212 : i1
              %214 = arith.select %213, %c2_i64, %c0_i64 : i64
              %215 = arith.addi %210, %214 overflow<nsw> : i64
              %216 = arith.addi %209, %215 overflow<nsw> : i64
              %217 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xf32, 1>
              %218 = arith.index_cast %216 : i64 to index
              %219 = arith.addi %218, %c63008 : index
              %220 = memref.load %217[%219] : memref<?xf32, 1>
              %221 = arith.cmpf ogt, %220, %cst : f32
              %222 = arith.extui %221 : i1 to i32
              %223 = arith.remsi %171, %c8_i64 : i64
              %224 = arith.cmpi ne, %223, %c0_i64 : i64
              %225 = arith.cmpi slt, %223, %c0_i64 : i64
              %226 = arith.andi %224, %225 : i1
              %227 = arith.select %226, %c8_i64, %c0_i64 : i64
              %228 = arith.addi %223, %227 overflow<nsw> : i64
              %229 = arith.remsi %175, %c8_i64 : i64
              %230 = arith.cmpi ne, %229, %c0_i64 : i64
              %231 = arith.cmpi slt, %229, %c0_i64 : i64
              %232 = arith.andi %230, %231 : i1
              %233 = arith.select %232, %c8_i64, %c0_i64 : i64
              %234 = arith.addi %229, %233 overflow<nsw> : i64
              %235 = cannir.scalar_storage_cast %222 : i32 -> ui32
              %236 = ascscalar.bitcast_memref %5 byte_offset = %c0_i64 : memref<3072xui8, 9> -> memref<?xui32, 9>
              %237 = arith.index_cast %228 : i64 to index
              %238 = arith.index_cast %234 : i64 to index
              %239 = arith.index_cast %0 : i64 to index
              %240 = arith.muli %237, %c16 : index
              %241 = arith.muli %238, %c2 : index
              %242 = arith.addi %240, %241 : index
              %243 = arith.addi %242, %239 : index
              %c0_10 = arith.constant 0 : index
              %c1_11 = arith.constant 1 : index
              scf.for %arg21 = %c0_10 to %c1_11 step %c1_11 {
                ascvec.ssbuf_store %235, %236[%243] : ui32, memref<?xui32, 9>
              }
              ascvec.sync_intra_arrive s, %c2_i64
            }
          }
        }
        scf.if %160 {
          %202 = arith.addi %157, %c1_i64 : i64
          %203 = arith.remsi %202, %c3_i64 : i64
          %204 = arith.cmpi ne, %203, %c0_i64 : i64
          %205 = arith.cmpi slt, %203, %c0_i64 : i64
          %206 = arith.andi %204, %205 : i1
          %207 = arith.select %206, %c3_i64, %c0_i64 : i64
          %208 = arith.addi %203, %207 overflow<nsw> : i64
          %209 = arith.cmpi eq, %208, %c1_i64 : i64
          %210 = arith.select %209, %c262144_i64, %c0_i64 : i64
          %211 = arith.cmpi eq, %208, %c2_i64 : i64
          %212 = arith.select %211, %c524288_i64, %210 : i64
          %213 = arith.divui %212, %c8_i64 : i64
          %214 = ascscalar.bind_local_ptr %213 : memref<1xbf16, 1>
          %215 = arith.addi %208, %c8_i64 overflow<nsw> : i64
          ascvec.get_buf v, %215, 0
          %216 = arith.remsi %151, %c2_i64 : i64
          %217 = arith.cmpi ne, %216, %c0_i64 : i64
          %218 = arith.cmpi slt, %216, %c0_i64 : i64
          %219 = arith.andi %217, %218 : i1
          %220 = arith.select %219, %c2_i64, %c0_i64 : i64
          %221 = arith.addi %216, %220 overflow<nsw> : i64
          %222 = arith.addi %221, %c2_i64 overflow<nsw> : i64
          ascvec.get_buf v, %222, 0
          %223 = arith.muli %221, %c64_i64 overflow<nsw> : i64
          %224 = arith.index_cast %223 : i64 to index
          %225 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xf32, 1>
          ascvec.vf(%214, %221, %225, %224, %arg12, %arg11 : memref<1xbf16, 1>, i64, memref<?xf32, 1>, index, f32, f32) {
          ^bb0(%arg21: memref<1xbf16, 1>, %arg22: i64, %arg23: memref<?xf32, 1>, %arg24: index, %arg25: f32, %arg26: f32):
            %cst_10 = arith.constant 1.000000e+00 : f32
            %cst_11 = arith.constant 0.001953125 : f32
            %c8_i32_12 = arith.constant 8 : i32
            %c3_i64_13 = arith.constant 3 : i64
            %c48_i32 = arith.constant 48 : i32
            %c2_i64_14 = arith.constant 2 : i64
            %c1_i64_15 = arith.constant 1 : i64
            %c16_i32 = arith.constant 16 : i32
            %c60064_16 = arith.constant 60064 : index
            %c4_i32_17 = arith.constant 4 : i32
            %c64_i64_18 = arith.constant 64 : i64
            %c1_i32_19 = arith.constant 1 : i32
            %c32_i32_20 = arith.constant 32 : i32
            %c64_i32 = arith.constant 64 : i32
            %c512_i32 = arith.constant 512 : i32
            %cst_21 = arith.constant 0.000000e+00 : f32
            %c15_i32 = arith.constant 15 : i32
            %c0_i32_22 = arith.constant 0 : i32
            %226 = ascvec.create_mask : !ascvec.mask
            %227 = ascvec.create_mask pat_mode "PAT_VL16" : !ascvec.mask
            %228 = ascvec.arange %c0_i32_22 : i32 -> <ui32>
            %229 = ascvec.duplicate_scalar_nomask %c15_i32 : i32 -> <ui32>
            %230 = ascvec.and %228, %229, %226 : <ui32>, <ui32> -> <ui32>
            %231 = ascvec.create_mask pat_mode "PAT_VL32" : !ascvec.mask
            scf.for %arg27 = %c0_i32_22 to %c8_i32_12 step %c1_i32_19  : i32 {
              %233 = ascvec.duplicate_scalar_nomask %cst_21 : f32 -> <f32>
              %234 = scf.for %arg28 = %c0_i32_22 to %c32_i32_20 step %c1_i32_19 iter_args(%arg29 = %233) -> (!ascvec.vreg<f32>)  : i32 {
                %262 = arith.muli %arg28, %c512_i32 overflow<nsw> : i32
                %263 = arith.muli %arg27, %c64_i32 overflow<nsw> : i32
                %264 = arith.addi %262, %263 overflow<nsw> : i32
                %265 = arith.index_cast %264 : i32 to index
                %266 = ascvec.loadalign_unpack %arg21[%265] {mode = "b16_to_b32"} : memref<1xbf16, 1>, index -> <bf16>
                %267 = ascvec.cast %266, %226 {rounding = 6 : i32} : <bf16> -> <f32>
                %268 = ascvec.mul %267, %267, %226 : <f32>
                %269 = ascvec.add %arg29, %268, %226 : <f32>
                scf.yield %269 : !ascvec.vreg<f32>
              }
              %235 = ascvec.add_scalar %230, %c0_i32_22, %226 : <ui32>, i32
              %236 = ascvec.gather_reg %234, %235 : <f32>, <ui32> -> <f32>
              %237 = ascvec.reduce_sum %236, %227 : <f32>
              %238 = arith.muli %arg22, %c64_i64_18 : i64
              %239 = arith.muli %arg27, %c4_i32_17 overflow<nsw> : i32
              %240 = arith.extsi %239 : i32 to i64
              %241 = arith.addi %238, %240 : i64
              %242 = arith.index_cast %241 : i64 to index
              %243 = arith.addi %242, %c60064_16 : index
              ascvec.storealign_1st %arg23[%243], %237 : memref<?xf32, 1>, index, <f32>
              %244 = ascvec.add_scalar %230, %c16_i32, %226 : <ui32>, i32
              %245 = ascvec.gather_reg %234, %244 : <f32>, <ui32> -> <f32>
              %246 = ascvec.reduce_sum %245, %227 : <f32>
              %247 = arith.addi %241, %c1_i64_15 : i64
              %248 = arith.index_cast %247 : i64 to index
              %249 = arith.addi %248, %c60064_16 : index
              ascvec.storealign_1st %arg23[%249], %246 : memref<?xf32, 1>, index, <f32>
              %250 = ascvec.add_scalar %230, %c32_i32_20, %226 : <ui32>, i32
              %251 = ascvec.gather_reg %234, %250 : <f32>, <ui32> -> <f32>
              %252 = ascvec.reduce_sum %251, %227 : <f32>
              %253 = arith.addi %241, %c2_i64_14 : i64
              %254 = arith.index_cast %253 : i64 to index
              %255 = arith.addi %254, %c60064_16 : index
              ascvec.storealign_1st %arg23[%255], %252 : memref<?xf32, 1>, index, <f32>
              %256 = ascvec.add_scalar %230, %c48_i32, %226 : <ui32>, i32
              %257 = ascvec.gather_reg %234, %256 : <f32>, <ui32> -> <f32>
              %258 = ascvec.reduce_sum %257, %227 : <f32>
              %259 = arith.addi %241, %c3_i64_13 : i64
              %260 = arith.index_cast %259 : i64 to index
              %261 = arith.addi %260, %c60064_16 : index
              ascvec.storealign_1st %arg23[%261], %258 : memref<?xf32, 1>, index, <f32>
            }
            ascvec.mem_bar vst_vld
            %232 = arith.addi %arg24, %c60064_16 : index
            %c0_23 = arith.constant 0 : index
            %c1_24 = arith.constant 1 : index
            scf.for %arg27 = %c0_23 to %c1_24 step %c1_24 {
              %233 = ascvec.loadalign %arg23[%232] : memref<?xf32, 1>, index -> <f32>
              %234 = ascvec.mul_scalar %233, %cst_11, %226 : <f32>, f32
              %235 = ascvec.duplicate_scalar_nomask %cst_10 : f32 -> <f32>
              %236 = ascvec.add_scalar %234, %arg25, %226 : <f32>, f32
              %237 = ascvec.sqrt %236, %226 : <f32>
              %238 = ascvec.div %235, %237, %226 : <f32>
              %239 = ascvec.mul_scalar %238, %arg26, %226 : <f32>, f32
              ascvec.storealign %arg23[%232], %239, %231 : memref<?xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %222, 0
          ascvec.release_buf v, %215, 0
        }
        %197 = cannir.datastruct_get %8["head"] : <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %198 = arith.addi %197, %c1_i64 : i64
        %199 = arith.cmpi eq, %198, %c6_i64 : i64
        %200 = arith.select %199, %c0_i64, %198 : i64
        cannir.datastruct_set %200, %8["head"] : i64, <"SparsePrefillBoundedInt64Metadata", ["head", "active", "job", "query", "tile", "blocks", "length", "global_block"], [i64, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>, !cannir.array<6xi64>], [0, 8, 56, 104, 152, 200, 248, 296], 344, 8>
        %201 = arith.addi %arg20, %111 : i64
        scf.yield %201 : i64
      }
      %113:5 = scf.if %108 -> (i64, i64, i64, i64, i64) {
        scf.yield %c2_i64, %arg15, %arg16, %arg17, %arg18 : i64, i64, i64, i64, i64
      } else {
        %114 = arith.addi %arg15, %c4_i64 : i64
        %115 = arith.addi %arg16, %c1_i64 : i64
        %116 = arith.cmpi slt, %114, %c4_i64 : i64
        %117 = arith.select %116, %arg14, %c1_i64 : i64
        %118:2 = scf.if %116 -> (i64, i64) {
          %119 = arith.index_cast %114 : i64 to index
          %120 = memref.load %alloc_2[%119] : memref<4xi32>
          %121 = arith.extsi %120 : i32 to i64
          %122 = arith.addi %121, %c63_i64 : i64
          %123 = arith.remsi %122, %c64_i64 : i64
          %124 = arith.cmpi ne, %123, %c0_i64 : i64
          %125 = arith.cmpi slt, %123, %c0_i64 : i64
          %126 = arith.andi %124, %125 : i1
          %127 = arith.divsi %122, %c64_i64 : i64
          %128 = arith.extui %126 : i1 to i64
          %129 = arith.subi %127, %128 : i64
          %130 = arith.maxsi %129, %c1_i64 : i64
          scf.yield %121, %130 : i64, i64
        } else {
          scf.yield %arg17, %arg18 : i64, i64
        }
        scf.yield %117, %114, %115, %118#0, %118#1 : i64, i64, i64, i64, i64
      }
      scf.yield %112, %113#0, %113#1, %113#2, %113#3, %113#4 : i64, i64, i64, i64, i64, i64
    }
    return
  }
  func.func @kernel_fused_norm_rope_attn_rope_cast_fwd_kernel_0_aic(%arg0: !cannir.tensor<<gm, bf16>, <<(4, 32768)>, <(32768, 1)>>, <<(4, 32768)>, <(32768, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(32, 512)>, <(512, 1)>>, <<(32, 512)>, <(512, 1)>>, identity>, %arg2: !cannir.tensor<<gm, i32>, <<(4, 32)>, <(32, 1)>>, <<(4, 32)>, <(32, 1)>>, identity>, %arg3: !cannir.tensor<<gm, i32>, <<(4)>, <(1)>>, <<(4)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, i32>, <<(4)>, <(1)>>, <<(4)>, <(1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(37, 64)>, <(64, 1)>>, <<(37, 64)>, <(64, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(64)>, <(1)>>, <<(64)>, <(1)>>, identity>, %arg7: !cannir.tensor<<gm, f8E4M3FN>, <<(32, 4096)>, <(4096, 1)>>, <<(32, 4096)>, <(4096, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(1024)>, <(1)>>, <<(1024)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, f32>, <<(256)>, <(1)>>, <<(256)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, f32>, <<(256)>, <(1)>>, <<(256)>, <(1)>>, identity>, %arg11: f32, %arg12: f32) attributes {cannir.core = "aic", cannir.cube_state_mode = 1 : i32, cannir.cube_state_planned, cannir.device, cannir.kernel_group = "kernel_fused_norm_rope_attn_rope_cast_fwd_kernel_0"} {
    %c28672 = arith.constant 28672 : index
    %c20480 = arith.constant 20480 : index
    %c53504 = arith.constant 53504 : index
    %c49408 = arith.constant 49408 : index
    %c160000 = arith.constant 160000 : index
    %c25344 = arith.constant 25344 : index
    %c159744 = arith.constant 159744 : index
    %c25088 = arith.constant 25088 : index
    %c45312 = arith.constant 45312 : index
    %c24832 = arith.constant 24832 : index
    %c45056 = arith.constant 45056 : index
    %c24576 = arith.constant 24576 : index
    %c241920 = arith.constant 241920 : index
    %c241664 = arith.constant 241664 : index
    %c155904 = arith.constant 155904 : index
    %c17152 = arith.constant 17152 : index
    %c155648 = arith.constant 155648 : index
    %c16896 = arith.constant 16896 : index
    %c41216 = arith.constant 41216 : index
    %c16640 = arith.constant 16640 : index
    %c40960 = arith.constant 40960 : index
    %c237824 = arith.constant 237824 : index
    %c237568 = arith.constant 237568 : index
    %c151808 = arith.constant 151808 : index
    %c8960 = arith.constant 8960 : index
    %c151552 = arith.constant 151552 : index
    %c8704 = arith.constant 8704 : index
    %c37120 = arith.constant 37120 : index
    %c36864 = arith.constant 36864 : index
    %c233728 = arith.constant 233728 : index
    %c233472 = arith.constant 233472 : index
    %c147712 = arith.constant 147712 : index
    %c147456 = arith.constant 147456 : index
    %c33024 = arith.constant 33024 : index
    %c229632 = arith.constant 229632 : index
    %c229376 = arith.constant 229376 : index
    %c32768 = arith.constant 32768 : index
    %c16384 = arith.constant 16384 : index
    %c49152_i64 = arith.constant 49152 : i64
    %c0_i64 = arith.constant 0 : i64
    %c230400_i64 = arith.constant 230400 : i64
    %c0_i32 = arith.constant 0 : i32
    %c128_i32 = arith.constant 128 : i32
    %c4_i64 = arith.constant 4 : i64
    %c10_i64 = arith.constant 10 : i64
    %c26_i64 = arith.constant 26 : i64
    %c5_i64 = arith.constant 5 : i64
    %c11_i64 = arith.constant 11 : i64
    %c27_i64 = arith.constant 27 : i64
    %c1_i64 = arith.constant 1 : i64
    %c17_i64 = arith.constant 17 : i64
    %c6_i64 = arith.constant 6 : i64
    %c22_i64 = arith.constant 22 : i64
    %c7_i64 = arith.constant 7 : i64
    %c23_i64 = arith.constant 23 : i64
    %c8_i64 = arith.constant 8 : i64
    %c24_i64 = arith.constant 24 : i64
    %c9_i64 = arith.constant 9 : i64
    %c25_i64 = arith.constant 25 : i64
    %c63_i64 = arith.constant 63 : i64
    %c64_i64 = arith.constant 64 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c28_i64 = arith.constant 28 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c458752_i64 = arith.constant 458752 : i64
    %c4_i32 = arith.constant 4 : i32
    %c1_i32 = arith.constant 1 : i32
    %c131072_i64 = arith.constant 131072 : i64
    %c16_i64 = arith.constant 16 : i64
    %c2_i32 = arith.constant 2 : i32
    %c262144_i64 = arith.constant 262144 : i64
    %c3_i32 = arith.constant 3 : i32
    %c393216_i64 = arith.constant 393216 : i64
    %c2_i64 = arith.constant 2 : i64
    %c3932160_i64 = arith.constant 3932160 : i64
    %c3670016_i64 = arith.constant 3670016 : i64
    %c786432_i64 = arith.constant 786432 : i64
    %c2621440_i64 = arith.constant 2621440 : i64
    %c524288_i64 = arith.constant 524288 : i64
    %c2359296_i64 = arith.constant 2359296 : i64
    %c1048576_i64 = arith.constant 1048576 : i64
    %c2883584_i64 = arith.constant 2883584 : i64
    %c3_i64 = arith.constant 3 : i64
    %c1310720_i64 = arith.constant 1310720 : i64
    %c3145728_i64 = arith.constant 3145728 : i64
    %c18_i64 = arith.constant 18 : i64
    %c20_i64 = arith.constant 20 : i64
    %c21_i64 = arith.constant 21 : i64
    %true = arith.constant true
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c98304_i64 = arith.constant 98304 : i64
    %c8_i32 = arith.constant 8 : i32
    %c64_i32 = arith.constant 64 : i32
    %c512_i32 = arith.constant 512 : i32
    %c16_i32 = arith.constant 16 : i32
    %c24_i32 = arith.constant 24 : i32
    %c256 = arith.constant 256 : index
    %c32768_i64 = arith.constant 32768 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index
    %c768 = arith.constant 768 : index
    %c4096 = arith.constant 4096 : index
    %c4352 = arith.constant 4352 : index
    %c8192 = arith.constant 8192 : index
    %c8448 = arith.constant 8448 : index
    %c12288 = arith.constant 12288 : index
    %c12544 = arith.constant 12544 : index
    %c129024 = arith.constant 129024 : index
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<131072xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<4xi32>
    %0 = ascscalar.bind_local_ptr %c0_i64 : memref<253952xui8, 1>
    %1 = ascscalar.bind_local_ptr %c230400_i64 : memref<2048xf32, 1>
    %2 = ascscalar.bind_local_ptr %c0_i64 : memref<524288xui8, 2>
    %3 = ascscalar.bind_local_ptr %c0_i64 : memref<65536xui8, 4>
    %4 = ascscalar.bind_local_ptr %c0_i64 : memref<262144xui8, 5>
    %5 = cannir.datastruct_alloc : <"SparsePrefillCubeCursor", ["slot"], [ui32], [0], 4, 4>
    %6 = cannir.scalar_storage_cast %c0_i32 : i32 -> ui32
    cannir.datastruct_set %6, %5["slot"] : ui32, <"SparsePrefillCubeCursor", ["slot"], [ui32], [0], 4, 4>
    %7 = ascscalar.bind_local_ptr %c0_i64 : memref<3072xui8, 9>
    asccube.sync_notify {dst_pipe = 5 : i32, event_id = 0 : i32, src_pipe = 4 : i32}
    asccube.sync_intra_arrive mte1, %c10_i64
    asccube.sync_intra_arrive mte1, %c26_i64
    asccube.sync_notify {dst_pipe = 5 : i32, event_id = 1 : i32, src_pipe = 4 : i32}
    asccube.sync_intra_arrive mte1, %c11_i64
    asccube.sync_intra_arrive mte1, %c27_i64
    asccube.sync_intra_arrive mte1, %c1_i64
    asccube.sync_intra_arrive mte1, %c17_i64
    asccube.sync_intra_arrive mte1, %c6_i64
    asccube.sync_intra_arrive mte1, %c22_i64
    asccube.sync_intra_arrive mte1, %c7_i64
    asccube.sync_intra_arrive mte1, %c23_i64
    asccube.sync_intra_arrive mte1, %c8_i64
    asccube.sync_intra_arrive mte1, %c24_i64
    asccube.sync_intra_arrive mte1, %c9_i64
    asccube.sync_intra_arrive mte1, %c25_i64
    %8 = ascscalar.get_block_idx
    %9 = arith.index_cast %8 : i64 to index
    %10 = memref.load %alloc_0[%9] : memref<4xi32>
    %11 = arith.extsi %10 : i32 to i64
    %12 = arith.addi %11, %c63_i64 : i64
    %13 = arith.remsi %12, %c64_i64 : i64
    %14 = arith.cmpi ne, %13, %c0_i64 : i64
    %15 = arith.cmpi slt, %13, %c0_i64 : i64
    %16 = arith.andi %14, %15 : i1
    %17 = arith.divsi %12, %c64_i64 : i64
    %18 = arith.extui %16 : i1 to i64
    %19 = arith.subi %17, %18 : i64
    %20 = arith.maxsi %19, %c1_i64 : i64
    asccube.sync_wait {dst_pipe = 5 : i32, event_id = 0 : i32, src_pipe = 4 : i32}
    %21 = arith.muli %8, %c65536_i64 : i64
    "asccube.copy_gm2l1_plan"(%2, %alloc, %c0_i32, %c0_i32, %c0_i64, %21, %c28_i64, %c1024_i64, %c2048_i64, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 262144 : i64, l1_base = 0 : i64, l1_capacity = 524288 : i64, pad_storage_bits = false, storage_dtype = bf16}> : (memref<524288xui8, 2>, memref<131072xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
    %22 = arith.addi %21, %c1024_i64 : i64
    "asccube.copy_gm2l1_plan"(%2, %alloc, %c0_i32, %c0_i32, %c458752_i64, %22, %c28_i64, %c1024_i64, %c2048_i64, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 262144 : i64, l1_base = 0 : i64, l1_capacity = 524288 : i64, pad_storage_bits = false, storage_dtype = bf16}> : (memref<524288xui8, 2>, memref<131072xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
    asccube.sync_notify {dst_pipe = 4 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
    asccube.sync_intra_wait mte1, %c10_i64
    asccube.sync_intra_wait mte1, %c26_i64
    asccube.set_mmad_direction m
    %23 = cannir.datastruct_get %5["slot"] : <"SparsePrefillCubeCursor", ["slot"], [ui32], [0], 4, 4>
    %24 = cannir.scalar_storage_cast %23 : ui32 -> i32
    %25 = arith.remui %24, %c4_i32 : i32
    %26 = arith.cmpi eq, %25, %c1_i32 : i32
    %27 = arith.select %26, %c131072_i64, %c0_i64 : i64
    %28 = arith.cmpi eq, %25, %c2_i32 : i32
    %29 = arith.select %28, %c262144_i64, %27 : i64
    %30 = arith.cmpi eq, %25, %c3_i32 : i32
    %31 = arith.select %30, %c393216_i64, %29 : i64
    asccube.sync_wait {dst_pipe = 4 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
    %32 = arith.extui %25 : i32 to i64
    asccube.get_buf mte1, %32, 0
    %33 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 3>
    %34 = ascscalar.bitcast_memref %2 byte_offset = %c0_i64 : memref<524288xui8, 2> -> memref<?xbf16, 2>
    %35 = arith.index_cast %31 : i64 to index
    %36 = arith.divui %35, %c16 : index
    asccube.copy_l12l0a %33[%36], %34[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
    %37 = arith.addi %36, %c256 : index
    asccube.copy_l12l0a %33[%37], %34[%c256], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
    %38 = arith.addi %36, %c512 : index
    asccube.copy_l12l0a %33[%38], %34[%c229376], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
    %39 = arith.addi %36, %c768 : index
    asccube.copy_l12l0a %33[%39], %34[%c229632], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
    asccube.release_buf mte1, %32, 0
    asccube.sync_intra_wait mte1, %c6_i64
    asccube.sync_intra_wait mte1, %c22_i64
    asccube.get_buf mte1, %c4_i64, 0
    %40 = ascscalar.bitcast_memref %3 byte_offset = %c0_i64 : memref<65536xui8, 4> -> memref<?xbf16, 4>
    asccube.copy_l12l0b %40[%c0], %34[%c32768], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
    asccube.copy_l12l0b %40[%c256], %34[%c33024], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
    asccube.copy_l12l0b %40[%c512], %34[%c147456], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
    asccube.copy_l12l0b %40[%c768], %34[%c147712], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
    asccube.release_buf mte1, %c4_i64, 0
    asccube.get_buf m, %32, 0
    asccube.get_buf m, %c4_i64, 0
    %41 = ascscalar.bind_local_ptr %c131072_i64 : memref<1xf32, 5>
    %42 = arith.divui %31, %c8_i64 : i64
    %43 = ascscalar.bind_local_ptr %42 : memref<1xbf16, 3>
    asccube.mmad %41[%c0], %43, %40, %c64_i32, %c128_i32, %c64_i32, %c2_i32 {c_matrix_init_val = true, mmad_direction = 0 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<?xbf16, 4>
    asccube.release_buf m, %32, 0
    asccube.release_buf m, %c4_i64, 0
    %44 = arith.addi %24, %c1_i32 : i32
    %45 = arith.remui %44, %c4_i32 : i32
    %46 = arith.cmpi eq, %45, %c1_i32 : i32
    %47 = arith.select %46, %c131072_i64, %c0_i64 : i64
    %48 = arith.cmpi eq, %45, %c2_i32 : i32
    %49 = arith.select %48, %c262144_i64, %47 : i64
    %50 = arith.cmpi eq, %45, %c3_i32 : i32
    %51 = arith.select %50, %c393216_i64, %49 : i64
    %52 = arith.extui %45 : i32 to i64
    asccube.get_buf mte1, %52, 0
    %53 = arith.index_cast %51 : i64 to index
    %54 = arith.divui %53, %c16 : index
    asccube.copy_l12l0a %33[%54], %34[%c4096], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
    %55 = arith.addi %54, %c256 : index
    asccube.copy_l12l0a %33[%55], %34[%c4352], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
    %56 = arith.addi %54, %c512 : index
    asccube.copy_l12l0a %33[%56], %34[%c233472], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
    %57 = arith.addi %54, %c768 : index
    asccube.copy_l12l0a %33[%57], %34[%c233728], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
    asccube.release_buf mte1, %52, 0
    asccube.get_buf mte1, %c5_i64, 0
    asccube.copy_l12l0b %40[%c8192], %34[%c36864], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
    asccube.copy_l12l0b %40[%c8448], %34[%c37120], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
    asccube.copy_l12l0b %40[%c8704], %34[%c151552], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
    asccube.copy_l12l0b %40[%c8960], %34[%c151808], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
    asccube.release_buf mte1, %c5_i64, 0
    asccube.get_buf m, %52, 0
    asccube.get_buf m, %c5_i64, 0
    %58 = arith.divui %51, %c8_i64 : i64
    %59 = ascscalar.bind_local_ptr %58 : memref<1xbf16, 3>
    %60 = ascscalar.bind_local_ptr %c16384_i64 : memref<1xbf16, 4>
    asccube.mmad %41[%c0], %59, %60, %c64_i32, %c128_i32, %c64_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
    asccube.release_buf m, %52, 0
    asccube.release_buf m, %c5_i64, 0
    %61 = arith.addi %24, %c2_i32 : i32
    %62 = arith.remui %61, %c4_i32 : i32
    %63 = arith.cmpi eq, %62, %c1_i32 : i32
    %64 = arith.select %63, %c131072_i64, %c0_i64 : i64
    %65 = arith.cmpi eq, %62, %c2_i32 : i32
    %66 = arith.select %65, %c262144_i64, %64 : i64
    %67 = arith.cmpi eq, %62, %c3_i32 : i32
    %68 = arith.select %67, %c393216_i64, %66 : i64
    %69 = arith.extui %62 : i32 to i64
    asccube.get_buf mte1, %69, 0
    %70 = arith.index_cast %68 : i64 to index
    %71 = arith.divui %70, %c16 : index
    asccube.copy_l12l0a %33[%71], %34[%c8192], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
    %72 = arith.addi %71, %c256 : index
    asccube.copy_l12l0a %33[%72], %34[%c8448], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
    %73 = arith.addi %71, %c512 : index
    asccube.copy_l12l0a %33[%73], %34[%c237568], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
    %74 = arith.addi %71, %c768 : index
    asccube.copy_l12l0a %33[%74], %34[%c237824], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
    asccube.release_buf mte1, %69, 0
    asccube.get_buf mte1, %c6_i64, 0
    asccube.copy_l12l0b %40[%c16384], %34[%c40960], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
    asccube.copy_l12l0b %40[%c16640], %34[%c41216], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
    asccube.copy_l12l0b %40[%c16896], %34[%c155648], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
    asccube.copy_l12l0b %40[%c17152], %34[%c155904], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
    asccube.release_buf mte1, %c6_i64, 0
    asccube.get_buf m, %69, 0
    asccube.get_buf m, %c6_i64, 0
    %75 = arith.divui %68, %c8_i64 : i64
    %76 = ascscalar.bind_local_ptr %75 : memref<1xbf16, 3>
    %77 = ascscalar.bind_local_ptr %c32768_i64 : memref<1xbf16, 4>
    asccube.mmad %41[%c0], %76, %77, %c64_i32, %c128_i32, %c64_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
    asccube.release_buf m, %69, 0
    asccube.release_buf m, %c6_i64, 0
    %78 = arith.addi %24, %c3_i32 : i32
    %79 = arith.remui %78, %c4_i32 : i32
    %80 = arith.cmpi eq, %79, %c1_i32 : i32
    %81 = arith.select %80, %c131072_i64, %c0_i64 : i64
    %82 = arith.cmpi eq, %79, %c2_i32 : i32
    %83 = arith.select %82, %c262144_i64, %81 : i64
    %84 = arith.cmpi eq, %79, %c3_i32 : i32
    %85 = arith.select %84, %c393216_i64, %83 : i64
    %86 = arith.extui %79 : i32 to i64
    asccube.get_buf mte1, %86, 0
    %87 = arith.index_cast %85 : i64 to index
    %88 = arith.divui %87, %c16 : index
    asccube.copy_l12l0a %33[%88], %34[%c12288], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
    %89 = arith.addi %88, %c256 : index
    asccube.copy_l12l0a %33[%89], %34[%c12544], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
    %90 = arith.addi %88, %c512 : index
    asccube.copy_l12l0a %33[%90], %34[%c241664], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
    %91 = arith.addi %88, %c768 : index
    asccube.copy_l12l0a %33[%91], %34[%c241920], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
    asccube.release_buf mte1, %86, 0
    %92 = arith.cmpi eq, %20, %c1_i64 : i64
    scf.if %92 {
      asccube.sync_notify {dst_pipe = 5 : i32, event_id = 0 : i32, src_pipe = 4 : i32}
      asccube.sync_intra_arrive mte1, %c10_i64
      asccube.sync_intra_arrive mte1, %c26_i64
    }
    asccube.get_buf mte1, %c7_i64, 0
    asccube.copy_l12l0b %40[%c24576], %34[%c45056], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
    asccube.copy_l12l0b %40[%c24832], %34[%c45312], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
    asccube.copy_l12l0b %40[%c25088], %34[%c159744], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
    asccube.copy_l12l0b %40[%c25344], %34[%c160000], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<?xbf16, 2>
    asccube.release_buf mte1, %c7_i64, 0
    asccube.get_buf m, %86, 0
    asccube.get_buf m, %c7_i64, 0
    %93 = arith.divui %85, %c8_i64 : i64
    %94 = ascscalar.bind_local_ptr %93 : memref<1xbf16, 3>
    %95 = ascscalar.bind_local_ptr %c49152_i64 : memref<1xbf16, 4>
    asccube.mmad %41[%c0], %94, %95, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
    asccube.release_buf m, %86, 0
    asccube.release_buf m, %c7_i64, 0
    asccube.sync_intra_wait fixpipe, %c0_i64
    asccube.sync_intra_wait fixpipe, %c16_i64
    %96 = ascscalar.bitcast_memref %1 byte_offset = %c0_i64 : memref<2048xf32, 1> -> memref<?xf32, 1>
    %97 = ascscalar.bitcast_memref %4 byte_offset = %c0_i64 : memref<262144xui8, 5> -> memref<?xf32, 5>
    asccube.copy_l0c2ub %96[%c0], %97[%c32768], %c64_i32, %c64_i32, %c512_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = false, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
    asccube.sync_intra_arrive fixpipe, %c0_i64
    asccube.sync_intra_arrive fixpipe, %c16_i64
    %98:2 = scf.for %arg13 = %8 to %c4_i64 step %c4_i64 iter_args(%arg14 = %c0_i64, %arg15 = %c0_i64) -> (i64, i64)  : i64 {
      %99 = arith.index_cast %arg13 : i64 to index
      %100 = memref.load %alloc_0[%99] : memref<4xi32>
      %101 = arith.extsi %100 : i32 to i64
      %102 = arith.addi %101, %c63_i64 : i64
      %103 = arith.remsi %102, %c64_i64 : i64
      %104 = arith.cmpi ne, %103, %c0_i64 : i64
      %105 = arith.cmpi slt, %103, %c0_i64 : i64
      %106 = arith.andi %104, %105 : i1
      %107 = arith.divsi %102, %c64_i64 : i64
      %108 = arith.extui %106 : i1 to i64
      %109 = arith.subi %107, %108 : i64
      %110 = arith.maxsi %109, %c1_i64 : i64
      %111 = arith.subi %110, %c1_i64 : i64
      scf.for %arg16 = %c0_i64 to %111 step %c1_i64  : i64 {
        %162 = arith.addi %arg16, %c1_i64 : i64
        %163 = arith.addi %arg14, %arg16 : i64
        %164 = arith.addi %163, %c1_i64 : i64
        %165 = arith.cmpi eq, %162, %c0_i64 : i64
        scf.if %165 {
          %344 = arith.remsi %arg15, %c2_i64 : i64
          %345 = arith.cmpi ne, %344, %c0_i64 : i64
          %346 = arith.cmpi slt, %344, %c0_i64 : i64
          %347 = arith.andi %345, %346 : i1
          %348 = arith.select %347, %c2_i64, %c0_i64 : i64
          %349 = arith.addi %344, %348 overflow<nsw> : i64
          %350 = arith.addi %349, %c10_i64 overflow<nsw> : i64
          asccube.sync_intra_wait mte1, %350
          %351 = arith.addi %349, %c26_i64 overflow<nsw> : i64
          asccube.sync_intra_wait mte1, %351
        }
        %166 = arith.remsi %arg15, %c2_i64 : i64
        %167 = arith.cmpi ne, %166, %c0_i64 : i64
        %168 = arith.cmpi slt, %166, %c0_i64 : i64
        %169 = arith.andi %167, %168 : i1
        %170 = arith.select %169, %c2_i64, %c0_i64 : i64
        %171 = arith.addi %166, %170 overflow<nsw> : i64
        %172 = arith.cmpi ne, %171, %c0_i64 : i64
        %173 = arith.select %172, %c262144_i64, %c0_i64 : i64
        %174 = arith.select %172, %c3932160_i64, %c3670016_i64 : i64
        %175 = arith.remsi %164, %c4_i64 : i64
        %176 = arith.cmpi ne, %175, %c0_i64 : i64
        %177 = arith.cmpi slt, %175, %c0_i64 : i64
        %178 = arith.andi %176, %177 : i1
        %179 = arith.select %178, %c4_i64, %c0_i64 : i64
        %180 = arith.addi %175, %179 overflow<nsw> : i64
        %181 = arith.cmpi eq, %180, %c1_i64 : i64
        %182 = arith.select %181, %c786432_i64, %c524288_i64 : i64
        %183 = arith.select %181, %c2621440_i64, %c2359296_i64 : i64
        %184 = arith.cmpi eq, %180, %c2_i64 : i64
        %185 = arith.select %184, %c1048576_i64, %182 : i64
        %186 = arith.select %184, %c2883584_i64, %183 : i64
        %187 = arith.cmpi eq, %180, %c3_i64 : i64
        %188 = arith.select %187, %c1310720_i64, %185 : i64
        %189 = arith.select %187, %c3145728_i64, %186 : i64
        asccube.set_mmad_direction m
        %190 = cannir.datastruct_get %5["slot"] : <"SparsePrefillCubeCursor", ["slot"], [ui32], [0], 4, 4>
        %191 = cannir.scalar_storage_cast %190 : ui32 -> i32
        %192 = arith.remui %191, %c4_i32 : i32
        %193 = arith.cmpi eq, %192, %c1_i32 : i32
        %194 = arith.select %193, %c131072_i64, %c0_i64 : i64
        %195 = arith.cmpi eq, %192, %c2_i32 : i32
        %196 = arith.select %195, %c262144_i64, %194 : i64
        %197 = arith.cmpi eq, %192, %c3_i32 : i32
        %198 = arith.select %197, %c393216_i64, %196 : i64
        scf.if %165 {
          %344 = arith.cmpi eq, %171, %c0_i64 : i64
          scf.if %344 {
            asccube.sync_wait {dst_pipe = 4 : i32, event_id = 0 : i32, src_pipe = 5 : i32}
          } else {
            asccube.sync_wait {dst_pipe = 4 : i32, event_id = 1 : i32, src_pipe = 5 : i32}
          }
        }
        %199 = arith.extui %192 : i32 to i64
        asccube.get_buf mte1, %199, 0
        %200 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 2>
        %201 = arith.index_cast %173 : i64 to index
        %202 = arith.divui %201, %c16 : index
        %203 = arith.index_cast %198 : i64 to index
        %204 = arith.divui %203, %c16 : index
        asccube.copy_l12l0a %33[%204], %200[%202], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        %205 = arith.addi %202, %c256 : index
        %206 = arith.addi %204, %c256 : index
        asccube.copy_l12l0a %33[%206], %200[%205], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        %207 = arith.index_cast %174 : i64 to index
        %208 = arith.divui %207, %c16 : index
        %209 = arith.addi %204, %c512 : index
        asccube.copy_l12l0a %33[%209], %200[%208], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        %210 = arith.addi %208, %c256 : index
        %211 = arith.addi %204, %c768 : index
        asccube.copy_l12l0a %33[%211], %200[%210], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %199, 0
        %212 = arith.addi %180, %c6_i64 overflow<nsw> : i64
        asccube.sync_intra_wait mte1, %212
        %213 = arith.addi %180, %c22_i64 overflow<nsw> : i64
        asccube.sync_intra_wait mte1, %213
        asccube.get_buf mte1, %c4_i64, 0
        %214 = arith.index_cast %188 : i64 to index
        %215 = arith.divui %214, %c16 : index
        asccube.copy_l12l0b %40[%c0], %200[%215], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        %216 = arith.addi %215, %c256 : index
        asccube.copy_l12l0b %40[%c256], %200[%216], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        %217 = arith.index_cast %189 : i64 to index
        %218 = arith.divui %217, %c16 : index
        asccube.copy_l12l0b %40[%c512], %200[%218], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        %219 = arith.addi %218, %c256 : index
        asccube.copy_l12l0b %40[%c768], %200[%219], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %c4_i64, 0
        asccube.get_buf m, %199, 0
        asccube.get_buf m, %c4_i64, 0
        %220 = arith.divui %198, %c8_i64 : i64
        %221 = ascscalar.bind_local_ptr %220 : memref<1xbf16, 3>
        asccube.mmad %41[%c0], %221, %40, %c64_i32, %c128_i32, %c64_i32, %c2_i32 {c_matrix_init_val = true, mmad_direction = 0 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<?xbf16, 4>
        asccube.release_buf m, %199, 0
        asccube.release_buf m, %c4_i64, 0
        %222 = arith.addi %191, %c1_i32 : i32
        %223 = arith.remui %222, %c4_i32 : i32
        %224 = arith.cmpi eq, %223, %c1_i32 : i32
        %225 = arith.select %224, %c131072_i64, %c0_i64 : i64
        %226 = arith.cmpi eq, %223, %c2_i32 : i32
        %227 = arith.select %226, %c262144_i64, %225 : i64
        %228 = arith.cmpi eq, %223, %c3_i32 : i32
        %229 = arith.select %228, %c393216_i64, %227 : i64
        %230 = arith.extui %223 : i32 to i64
        asccube.get_buf mte1, %230, 0
        %231 = arith.addi %202, %c4096 : index
        %232 = arith.index_cast %229 : i64 to index
        %233 = arith.divui %232, %c16 : index
        asccube.copy_l12l0a %33[%233], %200[%231], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        %234 = arith.addi %202, %c4352 : index
        %235 = arith.addi %233, %c256 : index
        asccube.copy_l12l0a %33[%235], %200[%234], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        %236 = arith.addi %208, %c4096 : index
        %237 = arith.addi %233, %c512 : index
        asccube.copy_l12l0a %33[%237], %200[%236], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        %238 = arith.addi %208, %c4352 : index
        %239 = arith.addi %233, %c768 : index
        asccube.copy_l12l0a %33[%239], %200[%238], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %230, 0
        asccube.get_buf mte1, %c5_i64, 0
        %240 = arith.addi %215, %c4096 : index
        asccube.copy_l12l0b %40[%c8192], %200[%240], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        %241 = arith.addi %215, %c4352 : index
        asccube.copy_l12l0b %40[%c8448], %200[%241], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        %242 = arith.addi %218, %c4096 : index
        asccube.copy_l12l0b %40[%c8704], %200[%242], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        %243 = arith.addi %218, %c4352 : index
        asccube.copy_l12l0b %40[%c8960], %200[%243], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %c5_i64, 0
        asccube.get_buf m, %230, 0
        asccube.get_buf m, %c5_i64, 0
        %244 = arith.divui %229, %c8_i64 : i64
        %245 = ascscalar.bind_local_ptr %244 : memref<1xbf16, 3>
        asccube.mmad %41[%c0], %245, %60, %c64_i32, %c128_i32, %c64_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %230, 0
        asccube.release_buf m, %c5_i64, 0
        %246 = arith.addi %191, %c2_i32 : i32
        %247 = arith.remui %246, %c4_i32 : i32
        %248 = arith.cmpi eq, %247, %c1_i32 : i32
        %249 = arith.select %248, %c131072_i64, %c0_i64 : i64
        %250 = arith.cmpi eq, %247, %c2_i32 : i32
        %251 = arith.select %250, %c262144_i64, %249 : i64
        %252 = arith.cmpi eq, %247, %c3_i32 : i32
        %253 = arith.select %252, %c393216_i64, %251 : i64
        %254 = arith.extui %247 : i32 to i64
        asccube.get_buf mte1, %254, 0
        %255 = arith.addi %202, %c8192 : index
        %256 = arith.index_cast %253 : i64 to index
        %257 = arith.divui %256, %c16 : index
        asccube.copy_l12l0a %33[%257], %200[%255], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        %258 = arith.addi %202, %c8448 : index
        %259 = arith.addi %257, %c256 : index
        asccube.copy_l12l0a %33[%259], %200[%258], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        %260 = arith.addi %208, %c8192 : index
        %261 = arith.addi %257, %c512 : index
        asccube.copy_l12l0a %33[%261], %200[%260], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        %262 = arith.addi %208, %c8448 : index
        %263 = arith.addi %257, %c768 : index
        asccube.copy_l12l0a %33[%263], %200[%262], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %254, 0
        asccube.get_buf mte1, %c6_i64, 0
        %264 = arith.addi %215, %c8192 : index
        asccube.copy_l12l0b %40[%c16384], %200[%264], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        %265 = arith.addi %215, %c8448 : index
        asccube.copy_l12l0b %40[%c16640], %200[%265], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        %266 = arith.addi %218, %c8192 : index
        asccube.copy_l12l0b %40[%c16896], %200[%266], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        %267 = arith.addi %218, %c8448 : index
        asccube.copy_l12l0b %40[%c17152], %200[%267], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %c6_i64, 0
        asccube.get_buf m, %254, 0
        asccube.get_buf m, %c6_i64, 0
        %268 = arith.divui %253, %c8_i64 : i64
        %269 = ascscalar.bind_local_ptr %268 : memref<1xbf16, 3>
        asccube.mmad %41[%c0], %269, %77, %c64_i32, %c128_i32, %c64_i32, %c2_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %254, 0
        asccube.release_buf m, %c6_i64, 0
        %270 = arith.addi %191, %c3_i32 : i32
        %271 = arith.remui %270, %c4_i32 : i32
        %272 = arith.cmpi eq, %271, %c1_i32 : i32
        %273 = arith.select %272, %c131072_i64, %c0_i64 : i64
        %274 = arith.cmpi eq, %271, %c2_i32 : i32
        %275 = arith.select %274, %c262144_i64, %273 : i64
        %276 = arith.cmpi eq, %271, %c3_i32 : i32
        %277 = arith.select %276, %c393216_i64, %275 : i64
        %278 = arith.extui %271 : i32 to i64
        asccube.get_buf mte1, %278, 0
        %279 = arith.addi %202, %c12288 : index
        %280 = arith.index_cast %277 : i64 to index
        %281 = arith.divui %280, %c16 : index
        asccube.copy_l12l0a %33[%281], %200[%279], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        %282 = arith.addi %202, %c12544 : index
        %283 = arith.addi %281, %c256 : index
        asccube.copy_l12l0a %33[%283], %200[%282], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        %284 = arith.addi %208, %c12288 : index
        %285 = arith.addi %281, %c512 : index
        asccube.copy_l12l0a %33[%285], %200[%284], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        %286 = arith.addi %208, %c12544 : index
        %287 = arith.addi %281, %c768 : index
        asccube.copy_l12l0a %33[%287], %200[%286], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<1xbf16, 3>, memref<1xbf16, 2>
        asccube.release_buf mte1, %278, 0
        %288 = arith.addi %arg16, %c2_i64 : i64
        %289 = arith.cmpi eq, %288, %110 : i64
        scf.if %289 {
          %344 = arith.cmpi eq, %171, %c0_i64 : i64
          scf.if %344 {
            asccube.sync_notify {dst_pipe = 5 : i32, event_id = 0 : i32, src_pipe = 4 : i32}
          } else {
            asccube.sync_notify {dst_pipe = 5 : i32, event_id = 1 : i32, src_pipe = 4 : i32}
          }
          %345 = arith.addi %171, %c10_i64 overflow<nsw> : i64
          asccube.sync_intra_arrive mte1, %345
          %346 = arith.addi %171, %c26_i64 overflow<nsw> : i64
          asccube.sync_intra_arrive mte1, %346
        }
        asccube.get_buf mte1, %c7_i64, 0
        %290 = arith.addi %215, %c12288 : index
        asccube.copy_l12l0b %40[%c24576], %200[%290], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        %291 = arith.addi %215, %c12544 : index
        asccube.copy_l12l0b %40[%c24832], %200[%291], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        %292 = arith.addi %218, %c12288 : index
        asccube.copy_l12l0b %40[%c25088], %200[%292], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        %293 = arith.addi %218, %c12544 : index
        asccube.copy_l12l0b %40[%c25344], %200[%293], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c2_i32, %c4_i32 : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %c7_i64, 0
        asccube.get_buf m, %278, 0
        asccube.get_buf m, %c7_i64, 0
        %294 = arith.divui %277, %c8_i64 : i64
        %295 = ascscalar.bind_local_ptr %294 : memref<1xbf16, 3>
        asccube.mmad %41[%c0], %295, %95, %c64_i32, %c128_i32, %c64_i32, %c3_i32 {c_matrix_init_val = false, mmad_direction = 0 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        asccube.release_buf m, %278, 0
        asccube.release_buf m, %c7_i64, 0
        asccube.sync_intra_wait fixpipe, %c0_i64
        asccube.sync_intra_wait fixpipe, %c16_i64
        asccube.copy_l0c2ub %96[%c0], %97[%c32768], %c64_i32, %c64_i32, %c512_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = false, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
        asccube.sync_intra_arrive fixpipe, %c0_i64
        asccube.sync_intra_arrive fixpipe, %c16_i64
        %296 = arith.remsi %163, %c4_i64 : i64
        %297 = arith.cmpi ne, %296, %c0_i64 : i64
        %298 = arith.cmpi slt, %296, %c0_i64 : i64
        %299 = arith.andi %297, %298 : i1
        %300 = arith.select %299, %c4_i64, %c0_i64 : i64
        %301 = arith.addi %296, %300 overflow<nsw> : i64
        %302 = arith.cmpi eq, %301, %c1_i64 : i64
        %303 = arith.select %302, %c786432_i64, %c524288_i64 : i64
        %304 = arith.select %302, %c2621440_i64, %c2359296_i64 : i64
        %305 = arith.cmpi eq, %301, %c2_i64 : i64
        %306 = arith.select %305, %c1048576_i64, %303 : i64
        %307 = arith.select %305, %c2883584_i64, %304 : i64
        %308 = arith.cmpi eq, %301, %c3_i64 : i64
        %309 = arith.select %308, %c1310720_i64, %306 : i64
        %310 = arith.select %308, %c3145728_i64, %307 : i64
        asccube.set_mmad_direction n
        %311 = cannir.datastruct_get %5["slot"] : <"SparsePrefillCubeCursor", ["slot"], [ui32], [0], 4, 4>
        %312 = cannir.scalar_storage_cast %311 : ui32 -> i32
        %313 = arith.cmpi eq, %312, %c1_i32 : i32
        %314 = arith.select %313, %c131072_i64, %c0_i64 : i64
        %315 = arith.cmpi eq, %312, %c2_i32 : i32
        %316 = arith.select %315, %c262144_i64, %314 : i64
        %317 = arith.cmpi eq, %312, %c3_i32 : i32
        %318 = arith.select %317, %c393216_i64, %316 : i64
        %319 = arith.addi %312, %c1_i32 : i32
        %320 = arith.remui %319, %c4_i32 : i32
        %321 = cannir.scalar_storage_cast %320 : i32 -> ui32
        cannir.datastruct_set %321, %5["slot"] : ui32, <"SparsePrefillCubeCursor", ["slot"], [ui32], [0], 4, 4>
        %322 = arith.cmpi eq, %arg16, %c0_i64 : i64
        %323 = arith.select %322, %c2_i32, %c0_i32 : i32
        %324 = arith.cmpi eq, %162, %110 : i64
        %325 = arith.select %324, %c3_i32, %323 : i32
        asccube.get_buf mte1, %c4_i64, 0
        %326 = arith.index_cast %309 : i64 to index
        %327 = arith.divui %326, %c16 : index
        asccube.copy_l12l0b %40[%c0], %200[%327], %c0_i32, %c0_i32, %c2_i32, %c8_i32, %c2_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
        %328 = arith.index_cast %310 : i64 to index
        %329 = arith.divui %328, %c16 : index
        asccube.copy_l12l0b %40[%c4096], %200[%329], %c0_i32, %c0_i32, %c2_i32, %c8_i32, %c2_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %c4_i64, 0
        asccube.sync_intra_wait mte1, %c1_i64
        asccube.sync_intra_wait mte1, %c17_i64
        %330 = arith.extui %312 : i32 to i64
        asccube.get_buf mte1, %330, 0
        %331 = arith.index_cast %318 : i64 to index
        %332 = arith.divui %331, %c16 : index
        asccube.copy_l12l0a %33[%332], %34[%c129024], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
        asccube.release_buf mte1, %330, 0
        asccube.sync_intra_arrive mte1, %c1_i64
        asccube.sync_intra_arrive mte1, %c17_i64
        asccube.get_buf m, %330, 0
        %333 = arith.cmpi ne, %arg16, %c0_i64 : i64
        %334 = scf.if %333 -> (i32) {
          asccube.sync_intra_wait s, %c2_i64
          asccube.sync_intra_wait s, %c18_i64
          %344 = arith.remsi %arg15, %c8_i64 : i64
          %345 = arith.cmpi ne, %344, %c0_i64 : i64
          %346 = arith.cmpi slt, %344, %c0_i64 : i64
          %347 = arith.andi %345, %346 : i1
          %348 = arith.select %347, %c8_i64, %c0_i64 : i64
          %349 = arith.addi %344, %348 overflow<nsw> : i64
          %350 = arith.remsi %arg16, %c8_i64 : i64
          %351 = arith.cmpi ne, %350, %c0_i64 : i64
          %352 = arith.cmpi slt, %350, %c0_i64 : i64
          %353 = arith.andi %351, %352 : i1
          %354 = arith.select %353, %c8_i64, %c0_i64 : i64
          %355 = arith.addi %350, %354 overflow<nsw> : i64
          %356 = ascscalar.bitcast_memref %7 byte_offset = %c0_i64 : memref<3072xui8, 9> -> memref<?xui32, 9>
          %357 = arith.index_cast %349 : i64 to index
          %358 = arith.index_cast %355 : i64 to index
          %359 = arith.muli %357, %c16 : index
          %360 = arith.muli %358, %c2 : index
          %361 = arith.addi %359, %360 : index
          %362 = asccube.ssbuf_load %356[%361] : memref<?xui32, 9> -> ui32
          %363 = cannir.scalar_storage_cast %362 : ui32 -> i32
          %364 = arith.addi %361, %c1 : index
          %365 = asccube.ssbuf_load %356[%364] : memref<?xui32, 9> -> ui32
          %366 = cannir.scalar_storage_cast %365 : ui32 -> i32
          %367 = arith.addi %363, %366 : i32
          asccube.sync_intra_arrive s, %c2_i64
          asccube.sync_intra_arrive s, %c18_i64
          %368 = arith.cmpi ne, %367, %c0_i32 : i32
          scf.if %368 {
            asccube.sync_notify {dst_pipe = 11 : i32, event_id = 0 : i32, src_pipe = 3 : i32}
            asccube.sync_wait {dst_pipe = 11 : i32, event_id = 0 : i32, src_pipe = 3 : i32}
            asccube.sync_intra_wait fixpipe, %c4_i64
            asccube.sync_intra_wait fixpipe, %c20_i64
            %369 = ascscalar.bitcast_memref %0 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xf32, 1>
            asccube.copy_l0c2ub %369[%c49408], %97[%c0], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
            asccube.sync_intra_arrive fixpipe, %c4_i64
            asccube.sync_intra_arrive fixpipe, %c20_i64
            asccube.sync_intra_wait fixpipe, %c5_i64
            asccube.sync_intra_wait fixpipe, %c21_i64
            asccube.copy_l0c2ub %369[%c53504], %97[%c8192], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
            asccube.sync_intra_arrive fixpipe, %c5_i64
            asccube.sync_intra_arrive fixpipe, %c21_i64
            asccube.sync_intra_wait fixpipe, %c4_i64
            asccube.sync_intra_wait fixpipe, %c20_i64
            asccube.copy_l0c2ub %369[%c49408], %97[%c16384], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
            asccube.sync_intra_arrive fixpipe, %c4_i64
            asccube.sync_intra_arrive fixpipe, %c20_i64
            asccube.sync_intra_wait fixpipe, %c5_i64
            asccube.sync_intra_wait fixpipe, %c21_i64
            asccube.copy_l0c2ub %369[%c53504], %97[%c24576], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
            asccube.sync_intra_arrive fixpipe, %c5_i64
            asccube.sync_intra_arrive fixpipe, %c21_i64
            asccube.sync_notify {dst_pipe = 3 : i32, event_id = 0 : i32, src_pipe = 11 : i32}
            asccube.sync_wait {dst_pipe = 3 : i32, event_id = 0 : i32, src_pipe = 11 : i32}
          }
          scf.yield %367 : i32
        } else {
          scf.yield %c0_i32 : i32
        }
        asccube.get_buf m, %c4_i64, 0
        %335 = arith.cmpi ne, %334, %c0_i32 : i32
        %336 = arith.select %322, %true, %335 : i1
        %337 = arith.divui %318, %c8_i64 : i64
        %338 = ascscalar.bind_local_ptr %337 : memref<1xbf16, 3>
        scf.if %336 {
          asccube.mmad %97[%c0], %338, %40, %c64_i32, %c64_i32, %c128_i32, %325 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<?xbf16, 4>
        } else {
          asccube.mmad %97[%c0], %338, %40, %c64_i32, %c64_i32, %c128_i32, %325 {c_matrix_init_val = false, mmad_direction = 1 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<?xbf16, 4>
        }
        asccube.release_buf m, %c4_i64, 0
        asccube.get_buf mte1, %c5_i64, 0
        asccube.copy_l12l0b %40[%c8192], %200[%327], %c0_i32, %c8_i32, %c2_i32, %c8_i32, %c2_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.copy_l12l0b %40[%c12288], %200[%329], %c0_i32, %c8_i32, %c2_i32, %c8_i32, %c2_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %c5_i64, 0
        asccube.get_buf m, %c5_i64, 0
        %339 = ascscalar.bind_local_ptr %c32768_i64 : memref<1xf32, 5>
        scf.if %336 {
          asccube.mmad %339[%c0], %338, %60, %c64_i32, %c64_i32, %c128_i32, %325 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        } else {
          asccube.mmad %339[%c0], %338, %60, %c64_i32, %c64_i32, %c128_i32, %325 {c_matrix_init_val = false, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        }
        asccube.release_buf m, %c5_i64, 0
        asccube.get_buf mte1, %c6_i64, 0
        asccube.copy_l12l0b %40[%c16384], %200[%327], %c0_i32, %c16_i32, %c2_i32, %c8_i32, %c2_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.copy_l12l0b %40[%c20480], %200[%329], %c0_i32, %c16_i32, %c2_i32, %c8_i32, %c2_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %c6_i64, 0
        asccube.get_buf m, %c6_i64, 0
        %340 = ascscalar.bind_local_ptr %c65536_i64 : memref<1xf32, 5>
        scf.if %336 {
          asccube.mmad %340[%c0], %338, %77, %c64_i32, %c64_i32, %c128_i32, %325 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        } else {
          asccube.mmad %340[%c0], %338, %77, %c64_i32, %c64_i32, %c128_i32, %325 {c_matrix_init_val = false, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        }
        asccube.release_buf m, %c6_i64, 0
        asccube.get_buf mte1, %c7_i64, 0
        asccube.copy_l12l0b %40[%c24576], %200[%327], %c0_i32, %c24_i32, %c2_i32, %c8_i32, %c2_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.copy_l12l0b %40[%c28672], %200[%329], %c0_i32, %c24_i32, %c2_i32, %c8_i32, %c2_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
        asccube.release_buf mte1, %c7_i64, 0
        %341 = arith.addi %301, %c6_i64 overflow<nsw> : i64
        asccube.sync_intra_arrive mte1, %341
        %342 = arith.addi %301, %c22_i64 overflow<nsw> : i64
        asccube.sync_intra_arrive mte1, %342
        asccube.get_buf m, %c7_i64, 0
        %343 = ascscalar.bind_local_ptr %c98304_i64 : memref<1xf32, 5>
        scf.if %336 {
          asccube.mmad %343[%c0], %338, %95, %c64_i32, %c64_i32, %c128_i32, %325 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        } else {
          asccube.mmad %343[%c0], %338, %95, %c64_i32, %c64_i32, %c128_i32, %325 {c_matrix_init_val = false, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
        }
        asccube.release_buf m, %c7_i64, 0
        asccube.release_buf m, %330, 0
        scf.if %324 {
          asccube.sync_intra_wait fixpipe, %c4_i64
          asccube.sync_intra_wait fixpipe, %c20_i64
          %344 = ascscalar.bitcast_memref %0 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xf32, 1>
          asccube.copy_l0c2ub %344[%c49408], %97[%c0], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
          asccube.sync_intra_arrive fixpipe, %c4_i64
          asccube.sync_intra_arrive fixpipe, %c20_i64
          asccube.sync_intra_wait fixpipe, %c5_i64
          asccube.sync_intra_wait fixpipe, %c21_i64
          asccube.copy_l0c2ub %344[%c53504], %97[%c8192], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
          asccube.sync_intra_arrive fixpipe, %c5_i64
          asccube.sync_intra_arrive fixpipe, %c21_i64
          asccube.sync_intra_wait fixpipe, %c4_i64
          asccube.sync_intra_wait fixpipe, %c20_i64
          asccube.copy_l0c2ub %344[%c49408], %97[%c16384], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
          asccube.sync_intra_arrive fixpipe, %c4_i64
          asccube.sync_intra_arrive fixpipe, %c20_i64
          asccube.sync_intra_wait fixpipe, %c5_i64
          asccube.sync_intra_wait fixpipe, %c21_i64
          asccube.copy_l0c2ub %344[%c53504], %97[%c24576], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
          asccube.sync_intra_arrive fixpipe, %c5_i64
          asccube.sync_intra_arrive fixpipe, %c21_i64
        }
      }
      %112 = arith.addi %arg14, %110 : i64
      %113 = arith.subi %112, %c1_i64 : i64
      %114 = arith.remsi %113, %c4_i64 : i64
      %115 = arith.cmpi ne, %114, %c0_i64 : i64
      %116 = arith.cmpi slt, %114, %c0_i64 : i64
      %117 = arith.andi %115, %116 : i1
      %118 = arith.select %117, %c4_i64, %c0_i64 : i64
      %119 = arith.addi %114, %118 overflow<nsw> : i64
      %120 = arith.cmpi eq, %119, %c1_i64 : i64
      %121 = arith.select %120, %c786432_i64, %c524288_i64 : i64
      %122 = arith.select %120, %c2621440_i64, %c2359296_i64 : i64
      %123 = arith.cmpi eq, %119, %c2_i64 : i64
      %124 = arith.select %123, %c1048576_i64, %121 : i64
      %125 = arith.select %123, %c2883584_i64, %122 : i64
      %126 = arith.cmpi eq, %119, %c3_i64 : i64
      %127 = arith.select %126, %c1310720_i64, %124 : i64
      %128 = arith.select %126, %c3145728_i64, %125 : i64
      asccube.set_mmad_direction n
      %129 = cannir.datastruct_get %5["slot"] : <"SparsePrefillCubeCursor", ["slot"], [ui32], [0], 4, 4>
      %130 = cannir.scalar_storage_cast %129 : ui32 -> i32
      %131 = arith.cmpi eq, %130, %c1_i32 : i32
      %132 = arith.select %131, %c131072_i64, %c0_i64 : i64
      %133 = arith.cmpi eq, %130, %c2_i32 : i32
      %134 = arith.select %133, %c262144_i64, %132 : i64
      %135 = arith.cmpi eq, %130, %c3_i32 : i32
      %136 = arith.select %135, %c393216_i64, %134 : i64
      %137 = arith.addi %130, %c1_i32 : i32
      %138 = arith.remui %137, %c4_i32 : i32
      %139 = cannir.scalar_storage_cast %138 : i32 -> ui32
      cannir.datastruct_set %139, %5["slot"] : ui32, <"SparsePrefillCubeCursor", ["slot"], [ui32], [0], 4, 4>
      %140 = arith.cmpi eq, %111, %c0_i64 : i64
      asccube.get_buf mte1, %c4_i64, 0
      %141 = ascscalar.bind_local_ptr %c0_i64 : memref<1xbf16, 2>
      %142 = arith.index_cast %127 : i64 to index
      %143 = arith.divui %142, %c16 : index
      asccube.copy_l12l0b %40[%c0], %141[%143], %c0_i32, %c0_i32, %c2_i32, %c8_i32, %c2_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
      %144 = arith.index_cast %128 : i64 to index
      %145 = arith.divui %144, %c16 : index
      asccube.copy_l12l0b %40[%c4096], %141[%145], %c0_i32, %c0_i32, %c2_i32, %c8_i32, %c2_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
      asccube.release_buf mte1, %c4_i64, 0
      asccube.sync_intra_wait mte1, %c1_i64
      asccube.sync_intra_wait mte1, %c17_i64
      %146 = arith.extui %130 : i32 to i64
      asccube.get_buf mte1, %146, 0
      %147 = arith.index_cast %136 : i64 to index
      %148 = arith.divui %147, %c16 : index
      asccube.copy_l12l0a %33[%148], %34[%c129024], %c0_i32, %c0_i32, %c4_i32, %c4_i32, %c4_i32, %c4_i32 : memref<1xbf16, 3>, memref<?xbf16, 2>
      asccube.release_buf mte1, %146, 0
      asccube.sync_intra_arrive mte1, %c1_i64
      asccube.sync_intra_arrive mte1, %c17_i64
      asccube.get_buf m, %146, 0
      %149 = arith.cmpi ne, %111, %c0_i64 : i64
      %150 = scf.if %149 -> (i32) {
        asccube.sync_intra_wait s, %c2_i64
        asccube.sync_intra_wait s, %c18_i64
        %162 = arith.remsi %arg15, %c8_i64 : i64
        %163 = arith.cmpi ne, %162, %c0_i64 : i64
        %164 = arith.cmpi slt, %162, %c0_i64 : i64
        %165 = arith.andi %163, %164 : i1
        %166 = arith.select %165, %c8_i64, %c0_i64 : i64
        %167 = arith.addi %162, %166 overflow<nsw> : i64
        %168 = arith.remsi %111, %c8_i64 : i64
        %169 = arith.cmpi ne, %168, %c0_i64 : i64
        %170 = arith.cmpi slt, %168, %c0_i64 : i64
        %171 = arith.andi %169, %170 : i1
        %172 = arith.select %171, %c8_i64, %c0_i64 : i64
        %173 = arith.addi %168, %172 overflow<nsw> : i64
        %174 = ascscalar.bitcast_memref %7 byte_offset = %c0_i64 : memref<3072xui8, 9> -> memref<?xui32, 9>
        %175 = arith.index_cast %167 : i64 to index
        %176 = arith.index_cast %173 : i64 to index
        %177 = arith.muli %175, %c16 : index
        %178 = arith.muli %176, %c2 : index
        %179 = arith.addi %177, %178 : index
        %180 = asccube.ssbuf_load %174[%179] : memref<?xui32, 9> -> ui32
        %181 = cannir.scalar_storage_cast %180 : ui32 -> i32
        %182 = arith.addi %179, %c1 : index
        %183 = asccube.ssbuf_load %174[%182] : memref<?xui32, 9> -> ui32
        %184 = cannir.scalar_storage_cast %183 : ui32 -> i32
        %185 = arith.addi %181, %184 : i32
        asccube.sync_intra_arrive s, %c2_i64
        asccube.sync_intra_arrive s, %c18_i64
        %186 = arith.cmpi ne, %185, %c0_i32 : i32
        scf.if %186 {
          asccube.sync_notify {dst_pipe = 11 : i32, event_id = 0 : i32, src_pipe = 3 : i32}
          asccube.sync_wait {dst_pipe = 11 : i32, event_id = 0 : i32, src_pipe = 3 : i32}
          asccube.sync_intra_wait fixpipe, %c4_i64
          asccube.sync_intra_wait fixpipe, %c20_i64
          %187 = ascscalar.bitcast_memref %0 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xf32, 1>
          asccube.copy_l0c2ub %187[%c49408], %97[%c0], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
          asccube.sync_intra_arrive fixpipe, %c4_i64
          asccube.sync_intra_arrive fixpipe, %c20_i64
          asccube.sync_intra_wait fixpipe, %c5_i64
          asccube.sync_intra_wait fixpipe, %c21_i64
          asccube.copy_l0c2ub %187[%c53504], %97[%c8192], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
          asccube.sync_intra_arrive fixpipe, %c5_i64
          asccube.sync_intra_arrive fixpipe, %c21_i64
          asccube.sync_intra_wait fixpipe, %c4_i64
          asccube.sync_intra_wait fixpipe, %c20_i64
          asccube.copy_l0c2ub %187[%c49408], %97[%c16384], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
          asccube.sync_intra_arrive fixpipe, %c4_i64
          asccube.sync_intra_arrive fixpipe, %c20_i64
          asccube.sync_intra_wait fixpipe, %c5_i64
          asccube.sync_intra_wait fixpipe, %c21_i64
          asccube.copy_l0c2ub %187[%c53504], %97[%c24576], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c0_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
          asccube.sync_intra_arrive fixpipe, %c5_i64
          asccube.sync_intra_arrive fixpipe, %c21_i64
          asccube.sync_notify {dst_pipe = 3 : i32, event_id = 0 : i32, src_pipe = 11 : i32}
          asccube.sync_wait {dst_pipe = 3 : i32, event_id = 0 : i32, src_pipe = 11 : i32}
        }
        scf.yield %185 : i32
      } else {
        scf.yield %c0_i32 : i32
      }
      asccube.get_buf m, %c4_i64, 0
      %151 = arith.cmpi ne, %150, %c0_i32 : i32
      %152 = arith.select %140, %true, %151 : i1
      %153 = arith.divui %136, %c8_i64 : i64
      %154 = ascscalar.bind_local_ptr %153 : memref<1xbf16, 3>
      scf.if %152 {
        asccube.mmad %97[%c0], %154, %40, %c64_i32, %c64_i32, %c128_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<?xbf16, 4>
      } else {
        asccube.mmad %97[%c0], %154, %40, %c64_i32, %c64_i32, %c128_i32, %c3_i32 {c_matrix_init_val = false, mmad_direction = 1 : i32} : memref<?xf32, 5>, memref<1xbf16, 3>, memref<?xbf16, 4>
      }
      asccube.release_buf m, %c4_i64, 0
      asccube.get_buf mte1, %c5_i64, 0
      asccube.copy_l12l0b %40[%c8192], %141[%143], %c0_i32, %c8_i32, %c2_i32, %c8_i32, %c2_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
      asccube.copy_l12l0b %40[%c12288], %141[%145], %c0_i32, %c8_i32, %c2_i32, %c8_i32, %c2_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
      asccube.release_buf mte1, %c5_i64, 0
      asccube.get_buf m, %c5_i64, 0
      %155 = ascscalar.bind_local_ptr %c32768_i64 : memref<1xf32, 5>
      scf.if %152 {
        asccube.mmad %155[%c0], %154, %60, %c64_i32, %c64_i32, %c128_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
      } else {
        asccube.mmad %155[%c0], %154, %60, %c64_i32, %c64_i32, %c128_i32, %c3_i32 {c_matrix_init_val = false, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
      }
      asccube.release_buf m, %c5_i64, 0
      asccube.get_buf mte1, %c6_i64, 0
      asccube.copy_l12l0b %40[%c16384], %141[%143], %c0_i32, %c16_i32, %c2_i32, %c8_i32, %c2_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
      asccube.copy_l12l0b %40[%c20480], %141[%145], %c0_i32, %c16_i32, %c2_i32, %c8_i32, %c2_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
      asccube.release_buf mte1, %c6_i64, 0
      asccube.get_buf m, %c6_i64, 0
      %156 = ascscalar.bind_local_ptr %c65536_i64 : memref<1xf32, 5>
      scf.if %152 {
        asccube.mmad %156[%c0], %154, %77, %c64_i32, %c64_i32, %c128_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
      } else {
        asccube.mmad %156[%c0], %154, %77, %c64_i32, %c64_i32, %c128_i32, %c3_i32 {c_matrix_init_val = false, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
      }
      asccube.release_buf m, %c6_i64, 0
      asccube.get_buf mte1, %c7_i64, 0
      asccube.copy_l12l0b %40[%c24576], %141[%143], %c0_i32, %c24_i32, %c2_i32, %c8_i32, %c2_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
      asccube.copy_l12l0b %40[%c28672], %141[%145], %c0_i32, %c24_i32, %c2_i32, %c8_i32, %c2_i32, %c8_i32 {transpose = true} : memref<?xbf16, 4>, memref<1xbf16, 2>
      asccube.release_buf mte1, %c7_i64, 0
      %157 = arith.addi %119, %c6_i64 overflow<nsw> : i64
      asccube.sync_intra_arrive mte1, %157
      %158 = arith.addi %119, %c22_i64 overflow<nsw> : i64
      asccube.sync_intra_arrive mte1, %158
      asccube.get_buf m, %c7_i64, 0
      %159 = ascscalar.bind_local_ptr %c98304_i64 : memref<1xf32, 5>
      scf.if %152 {
        asccube.mmad %159[%c0], %154, %95, %c64_i32, %c64_i32, %c128_i32, %c3_i32 {c_matrix_init_val = true, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
      } else {
        asccube.mmad %159[%c0], %154, %95, %c64_i32, %c64_i32, %c128_i32, %c3_i32 {c_matrix_init_val = false, mmad_direction = 1 : i32} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
      }
      asccube.release_buf m, %c7_i64, 0
      asccube.release_buf m, %146, 0
      asccube.sync_intra_wait fixpipe, %c4_i64
      asccube.sync_intra_wait fixpipe, %c20_i64
      %160 = ascscalar.bitcast_memref %0 byte_offset = %c0_i64 : memref<253952xui8, 1> -> memref<?xf32, 1>
      asccube.copy_l0c2ub %160[%c49408], %97[%c0], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
      asccube.sync_intra_arrive fixpipe, %c4_i64
      asccube.sync_intra_arrive fixpipe, %c20_i64
      asccube.sync_intra_wait fixpipe, %c5_i64
      asccube.sync_intra_wait fixpipe, %c21_i64
      asccube.copy_l0c2ub %160[%c53504], %97[%c8192], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
      asccube.sync_intra_arrive fixpipe, %c5_i64
      asccube.sync_intra_arrive fixpipe, %c21_i64
      asccube.sync_intra_wait fixpipe, %c4_i64
      asccube.sync_intra_wait fixpipe, %c20_i64
      asccube.copy_l0c2ub %160[%c49408], %97[%c16384], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
      asccube.sync_intra_arrive fixpipe, %c4_i64
      asccube.sync_intra_arrive fixpipe, %c20_i64
      asccube.sync_intra_wait fixpipe, %c5_i64
      asccube.sync_intra_wait fixpipe, %c21_i64
      asccube.copy_l0c2ub %160[%c53504], %97[%c24576], %c128_i32, %c64_i32, %c128_i32, %c64_i32, %c3_i32 sub_block_id %c0_i32 {dual_dst_ctl = 1 : i32, nz2nd_en = true, relu_en = false} : memref<?xf32, 1>, memref<?xf32, 5>
      asccube.sync_intra_arrive fixpipe, %c5_i64
      asccube.sync_intra_arrive fixpipe, %c21_i64
      %161 = arith.addi %arg15, %c1_i64 : i64
      scf.yield %112, %161 : i64, i64
    }
    return
  }
}
