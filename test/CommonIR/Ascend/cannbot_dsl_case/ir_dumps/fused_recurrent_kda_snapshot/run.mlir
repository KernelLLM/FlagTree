module {
  func.func @run_89ea84c84357f39ee8fbbf856e0ddf32(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, f32>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, %arg5: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg7: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg11: f32, %arg12: f32, %arg13: i64, %arg14: i64, %arg15: i64, %arg16: i64) attributes {cannbotdsl.contract_digest = "89ea84c84357f39ee8fbbf856e0ddf328a58e4ebed221d6685a1b8942056a125", cannbotdsl.public_slot_count = 17 : i64, cannbotdsl.slot_dtype_bits = [16, 16, 16, 32, 32, 16, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64], cannbotdsl.slot_dtype_codes = [4, 4, 4, 2, 2, 4, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], cannbotdsl.slot_type_digests = ["8f15e1d567ed5c51f96e023195656b2b2efaa6dc837c7b31aafa739c26040a67", "1b63a5052d0139af8aaebc450bce89e97e677ae902fe094484174bfeb7cb8b1f", "b0813ae0c4fca5a3063c191e6b9dc3a5c63a74f31b92c9022de54d37fdbbe9d5", "f8b9e085a50e5514e32c422465bdebb2011afe7ef528f91d95ebf5ecb138ea6d", "7d5428a71908f6f567672e4c454612435054bf47a53e369ef079412388196734", "ae0e71603da7c81970e4128eaa2d19f1c750fd7167719cfecd0a9a253d6bcd7b", "fac953dfd7e5eeaac4c540a7ed359523af9e18dc925508cd339e595e80752447", "fac953dfd7e5eeaac4c540a7ed359523af9e18dc925508cd339e595e80752447", "a56bf1e2b6709e682004e78b91a48fe3cbb92b71f0052de51777ac63943e1da3", "e030c76f69ec395f0341ef8f9ab18ddc826e6df4047f270d22f84aa569a4ad93", "1c252c434943385e4309874205bbc2abd771b258c6eb6934db974ec70004a608", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel_FusedRecurrentKDASnapshotKernel___call___0(%arg16) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15] {cannbotdsl.profile = {indices = array<i64: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10>, name = "FusedRecurrentKDASnapshotKernel___call__", op_type = "FusedRecurrentKDASnapshotKernel___call__", roles = array<i64: 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0>}} : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, f32, f32, i64, i64, i64
    return
  }
  func.func @kernel_FusedRecurrentKDASnapshotKernel___call___0(%arg0: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg1: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg2: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>, %arg3: !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg4: !cannir.tensor<<gm, f32>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, %arg5: !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg7: !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>, %arg8: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg9: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg10: !cannir.tensor<<gm, i32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg11: f32, %arg12: f32, %arg13: i64, %arg14: i64, %arg15: i64) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel_FusedRecurrentKDASnapshotKernel___call___0", cannir.resource.ids_prepared} {
    %c16384_i32 = arith.constant 16384 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c65536_i32 = arith.constant 65536 : i32
    %c256_i32 = arith.constant 256 : i32
    %c0_i32 = arith.constant 0 : i32
    %c512_i64 = arith.constant 512 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c256_i64 = arith.constant 256 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i32 = arith.constant 64 : i32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c128_i32 = arith.constant 128 : i32
    %false = arith.constant false
    %c32768_i64 = arith.constant 32768 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c64_i64 = arith.constant 64 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c32_i64 = arith.constant 32 : i64
    %c16_i64 = arith.constant 16 : i64
    %c15_i64 = arith.constant 15 : i64
    %c14_i64 = arith.constant 14 : i64
    %c13_i64 = arith.constant 13 : i64
    %c12_i64 = arith.constant 12 : i64
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
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xbf16>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1xbf16>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<1xbf16>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<1xf32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<1xf32>
    %alloc_4 = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xbf16>
    %alloc_5 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1xf32>
    %alloc_6 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<1xf32>
    %alloc_7 = memref.alloc() {cannir.arg_index = 8 : i64} : memref<1xi32>
    %alloc_8 = memref.alloc() {cannir.arg_index = 9 : i64} : memref<1xi32>
    %alloc_9 = memref.alloc() {cannir.arg_index = 10 : i64} : memref<1xi32>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xf32, 1>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<128xf32, 1>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 66048 : i64} : memref<128xf32, 1>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 66560 : i64} : memref<128xf32, 1>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 67072 : i64} : memref<128xf32, 1>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 67584 : i64} : memref<64xf32, 1>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 67840 : i64} : memref<128xf32, 1>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 68352 : i64} : memref<16384xf32, 1>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 133888 : i64} : memref<16384xf32, 1>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 199424 : i64} : memref<128xbf16, 1>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 199680 : i64} : memref<128xbf16, 1>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 199936 : i64} : memref<128xbf16, 1>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 200192 : i64} : memref<128xbf16, 1>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 200448 : i64} : memref<4xf32, 1>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 200480 : i64} : memref<128xf32, 1>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 200992 : i64} : memref<128xf32, 1>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 201504 : i64} : memref<128xbf16, 1>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 201760 : i64} : memref<128xbf16, 1>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 202016 : i64} : memref<128xbf16, 1>
    %alloc_29 = memref.alloc() {cannir.byte_offset = 202272 : i64} : memref<128xbf16, 1>
    %alloc_30 = memref.alloc() {cannir.byte_offset = 202528 : i64} : memref<128xbf16, 1>
    %alloc_31 = memref.alloc() {cannir.byte_offset = 202784 : i64} : memref<128xbf16, 1>
    %alloc_32 = memref.alloc() {cannir.byte_offset = 203040 : i64} : memref<64xbf16, 1>
    %alloc_33 = memref.alloc() {cannir.byte_offset = 203168 : i64} : memref<64xbf16, 1>
    %alloc_34 = memref.alloc() {cannir.byte_offset = 203296 : i64} : memref<12288xf32, 1>
    %alloc_35 = memref.alloc() {cannir.byte_offset = 252448 : i64} : memref<96xf32, 1>
    %alloc_36 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xf32, 1>
    %alloc_37 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<4096xf32, 1>
    %alloc_38 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<128xf32, 1>
    %alloc_39 = memref.alloc() {cannir.byte_offset = 33280 : i64} : memref<128xf32, 1>
    %alloc_40 = memref.alloc() {cannir.byte_offset = 33792 : i64} : memref<128xf32, 1>
    %alloc_41 = memref.alloc() {cannir.byte_offset = 34304 : i64} : memref<64xf32, 1>
    %alloc_42 = memref.alloc() {cannir.byte_offset = 34560 : i64} : memref<64xf32, 1>
    %alloc_43 = memref.alloc() {cannir.byte_offset = 34816 : i64} : memref<32xf32, 1>
    %alloc_44 = memref.alloc() {cannir.byte_offset = 34944 : i64} : memref<4096xf32, 1>
    %alloc_45 = memref.alloc() {cannir.byte_offset = 51328 : i64} : memref<4096xf32, 1>
    %alloc_46 = memref.alloc() {cannir.byte_offset = 67712 : i64} : memref<64xbf16, 1>
    %alloc_47 = memref.alloc() {cannir.byte_offset = 67840 : i64} : memref<64xbf16, 1>
    %alloc_48 = memref.alloc() {cannir.byte_offset = 67968 : i64} : memref<32xbf16, 1>
    %alloc_49 = memref.alloc() {cannir.byte_offset = 68032 : i64} : memref<32xbf16, 1>
    %alloc_50 = memref.alloc() {cannir.byte_offset = 68096 : i64} : memref<4xf32, 1>
    %alloc_51 = memref.alloc() {cannir.byte_offset = 68128 : i64} : memref<64xf32, 1>
    %alloc_52 = memref.alloc() {cannir.byte_offset = 68384 : i64} : memref<64xf32, 1>
    %alloc_53 = memref.alloc() {cannir.byte_offset = 68640 : i64} : memref<128xbf16, 1>
    %alloc_54 = memref.alloc() {cannir.byte_offset = 68896 : i64} : memref<128xbf16, 1>
    %alloc_55 = memref.alloc() {cannir.byte_offset = 69152 : i64} : memref<128xbf16, 1>
    %alloc_56 = memref.alloc() {cannir.byte_offset = 69408 : i64} : memref<128xbf16, 1>
    %alloc_57 = memref.alloc() {cannir.byte_offset = 69664 : i64} : memref<128xbf16, 1>
    %alloc_58 = memref.alloc() {cannir.byte_offset = 69920 : i64} : memref<128xbf16, 1>
    %alloc_59 = memref.alloc() {cannir.byte_offset = 70176 : i64} : memref<64xbf16, 1>
    %alloc_60 = memref.alloc() {cannir.byte_offset = 70304 : i64} : memref<64xbf16, 1>
    %alloc_61 = memref.alloc() {cannir.byte_offset = 70432 : i64} : memref<12288xf32, 1>
    %alloc_62 = memref.alloc() {cannir.byte_offset = 119584 : i64} : memref<96xf32, 1>
    %alloc_63 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xf32, 1>
    %alloc_64 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<8192xf32, 1>
    %alloc_65 = memref.alloc() {cannir.byte_offset = 65536 : i64} : memref<128xf32, 1>
    %alloc_66 = memref.alloc() {cannir.byte_offset = 66048 : i64} : memref<128xf32, 1>
    %alloc_67 = memref.alloc() {cannir.byte_offset = 66560 : i64} : memref<128xf32, 1>
    %alloc_68 = memref.alloc() {cannir.byte_offset = 67072 : i64} : memref<64xf32, 1>
    %alloc_69 = memref.alloc() {cannir.byte_offset = 67328 : i64} : memref<64xf32, 1>
    %alloc_70 = memref.alloc() {cannir.byte_offset = 67584 : i64} : memref<64xf32, 1>
    %alloc_71 = memref.alloc() {cannir.byte_offset = 67840 : i64} : memref<8192xf32, 1>
    %alloc_72 = memref.alloc() {cannir.byte_offset = 100608 : i64} : memref<8192xf32, 1>
    %alloc_73 = memref.alloc() {cannir.byte_offset = 133376 : i64} : memref<64xbf16, 1>
    %alloc_74 = memref.alloc() {cannir.byte_offset = 133504 : i64} : memref<64xbf16, 1>
    %alloc_75 = memref.alloc() {cannir.byte_offset = 133632 : i64} : memref<64xbf16, 1>
    %alloc_76 = memref.alloc() {cannir.byte_offset = 133760 : i64} : memref<64xbf16, 1>
    %alloc_77 = memref.alloc() {cannir.byte_offset = 133888 : i64} : memref<4xf32, 1>
    %alloc_78 = memref.alloc() {cannir.byte_offset = 133920 : i64} : memref<64xf32, 1>
    %alloc_79 = memref.alloc() {cannir.byte_offset = 134176 : i64} : memref<64xf32, 1>
    %alloc_80 = memref.alloc() {cannir.byte_offset = 134432 : i64} : memref<128xbf16, 1>
    %alloc_81 = memref.alloc() {cannir.byte_offset = 134688 : i64} : memref<128xbf16, 1>
    %alloc_82 = memref.alloc() {cannir.byte_offset = 134944 : i64} : memref<128xbf16, 1>
    %alloc_83 = memref.alloc() {cannir.byte_offset = 135200 : i64} : memref<128xbf16, 1>
    %alloc_84 = memref.alloc() {cannir.byte_offset = 135456 : i64} : memref<128xbf16, 1>
    %alloc_85 = memref.alloc() {cannir.byte_offset = 135712 : i64} : memref<128xbf16, 1>
    %alloc_86 = memref.alloc() {cannir.byte_offset = 135968 : i64} : memref<64xbf16, 1>
    %alloc_87 = memref.alloc() {cannir.byte_offset = 136096 : i64} : memref<64xbf16, 1>
    %alloc_88 = memref.alloc() {cannir.byte_offset = 136224 : i64} : memref<12288xf32, 1>
    %alloc_89 = memref.alloc() {cannir.byte_offset = 185376 : i64} : memref<96xf32, 1>
    %0 = cannir.extract_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
    %1 = cannir.extract_shape %0 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 128)>
    %2 = cannir.extract_dim %1[1] : !cannir.shape<(-1, -1, -1, 128)>
    %3 = arith.muli %arg14, %2 : i64
    %4 = arith.cmpi sgt, %3, %c32_i64 : i64
    scf.if %4 {
      %5 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>
      %6 = cannir.extract_shape %5 : <<(-1)>, <(1)>> -> <(-1)>
      %7 = cannir.extract_dim %6[0] : !cannir.shape<(-1)>
      %8 = arith.muli %7, %c4_i64 : i64
      %9 = arith.cmpi eq, %8, %c0_i64 : i64
      %10 = arith.cmpi ne, %8, %c0_i64 : i64
      %11 = arith.extui %10 : i1 to i64
      %12 = arith.select %9, %c4_i64, %8 : i64
      "ascvec.copy_gm2ub_plan"(%alloc_35, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %11, %12, %c0_i64, %12, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 252448 : i64, ub_capacity = 384 : i64}> : (memref<96xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      %13 = cannir.extract_layout %arg4 : !cannir.tensor<<gm, f32>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>
      %14 = cannir.extract_shape %13 : <<(-1, 128)>, <(128, 1)>> -> <(-1, 128)>
      %15 = cannir.extract_dim %14[0] : !cannir.shape<(-1, 128)>
      %16 = arith.muli %15, %c512_i64 : i64
      %17 = arith.cmpi eq, %16, %c0_i64 : i64
      %18 = arith.cmpi ne, %16, %c0_i64 : i64
      %19 = arith.extui %18 : i1 to i64
      %20 = arith.select %17, %c4_i64, %16 : i64
      ascvec.get_buf mte2, %c15_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%alloc_34, %alloc_3, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %19, %20, %c0_i64, %20, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 203296 : i64, ub_capacity = 49152 : i64}> : (memref<12288xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c15_i64, 0 {cannir.buffer.generated_sync}
      %21 = ascscalar.get_block_idx
      %22 = ascscalar.get_block_num
      %23 = arith.muli %3, %c2_i64 : i64
      %24 = arith.remsi %23, %22 : i64
      %25 = arith.cmpi ne, %24, %c0_i64 : i64
      %26 = arith.cmpi slt, %24, %c0_i64 : i64
      %27 = arith.cmpi slt, %22, %c0_i64 : i64
      %28 = arith.xori %26, %27 : i1
      %29 = arith.andi %25, %28 : i1
      %30 = arith.divsi %23, %22 : i64
      %31 = arith.extui %29 : i1 to i64
      %32 = arith.subi %30, %31 : i64
      %33 = arith.muli %32, %22 : i64
      %34 = arith.subi %23, %33 : i64
      %35 = arith.remsi %32, %c2_i64 : i64
      %36 = arith.cmpi ne, %35, %c0_i64 : i64
      %37 = arith.cmpi slt, %35, %c0_i64 : i64
      %38 = arith.andi %36, %37 : i1
      %39 = arith.divsi %32, %c2_i64 : i64
      %40 = arith.extui %38 : i1 to i64
      %41 = arith.subi %39, %40 : i64
      %42 = arith.addi %32, %c1_i64 : i64
      %43 = arith.remsi %42, %c2_i64 : i64
      %44 = arith.cmpi ne, %43, %c0_i64 : i64
      %45 = arith.cmpi slt, %43, %c0_i64 : i64
      %46 = arith.andi %44, %45 : i1
      %47 = arith.divsi %42, %c2_i64 : i64
      %48 = arith.extui %46 : i1 to i64
      %49 = arith.subi %47, %48 : i64
      %50 = arith.subi %49, %41 : i64
      %51 = arith.cmpi slt, %21, %34 : i64
      %52 = arith.select %51, %49, %41 : i64
      %53 = arith.select %51, %21, %34 : i64
      %54 = arith.muli %21, %41 : i64
      %55 = arith.muli %53, %50 : i64
      %56 = arith.addi %54, %55 : i64
      %57:2 = scf.for %arg16 = %c0_i64 to %52 step %c1_i64 iter_args(%arg17 = %c0_i64, %arg18 = %c0_i64) -> (i64, i64)  : i64 {
        %69 = arith.addi %56, %arg16 : i64
        %70 = arith.maxsi %2, %c1_i64 : i64
        %71 = arith.remsi %69, %70 : i64
        %72 = arith.cmpi slt, %71, %c0_i64 : i64
        %73 = arith.divsi %69, %70 : i64
        %74 = arith.extui %72 : i1 to i64
        %75 = arith.subi %73, %74 : i64
        %76 = arith.maxsi %arg14, %c1_i64 : i64
        %77 = arith.remsi %75, %76 : i64
        %78 = arith.cmpi slt, %77, %c0_i64 : i64
        %79 = arith.select %78, %76, %c0_i64 : i64
        %80 = arith.addi %77, %79 : i64
        %81 = arith.select %72, %70, %c0_i64 : i64
        %82 = arith.addi %71, %81 : i64
        %83 = arith.muli %80, %arg15 : i64
        %84 = arith.addi %83, %arg15 : i64
        %85 = arith.cmpi eq, %arg13, %c1_i64 : i64
        %86 = arith.select %85, %c0_i64, %80 : i64
        %87:2 = scf.if %85 -> (i64, i64) {
          %211 = arith.index_cast %80 : i64 to index
          %212 = memref.load %alloc_9[%211] : memref<1xi32>
          %213 = arith.addi %80, %c1_i64 : i64
          %214 = arith.index_cast %213 : i64 to index
          %215 = memref.load %alloc_9[%214] : memref<1xi32>
          %216 = arith.extsi %212 : i32 to i64
          %217 = arith.extsi %215 : i32 to i64
          scf.yield %216, %217 : i64, i64
        } else {
          scf.yield %83, %84 : i64, i64
        }
        %88 = arith.index_cast %80 : i64 to index
        %89 = memref.load %alloc_8[%88] : memref<1xi32>
        %90 = arith.cmpi sgt, %89, %c0_i32 : i32
        %91 = arith.extsi %89 : i32 to i64
        %92 = arith.addi %87#0, %91 : i64
        %93 = arith.subi %92, %c1_i64 : i64
        %94 = arith.select %90, %93, %87#0 : i64
        %95 = arith.index_cast %94 : i64 to index
        %96 = memref.load %alloc_7[%95] : memref<1xi32>
        %97 = arith.extsi %96 : i32 to i64
        %98 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
        %99 = cannir.extract_physical_layout %arg6 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
        %100 = cannir.extract_stride %99 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
        %101 = cannir.extract_dim %100[0] : !cannir.stride<(-1, 16384, 128, 1)>
        %102 = arith.muli %97, %101 : i64
        %103 = arith.muli %102, %c4_i64 : i64
        %104 = arith.muli %82, %c65536_i64 : i64
        %105 = arith.addi %103, %104 : i64
        ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%98, %alloc_5, %c0_i32, %c0_i32, %c0_i64, %105, %c1_i64, %c65536_i64, %c0_i64, %c65536_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 65536 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
        %106 = arith.select %85, %87#0, %c0_i64 : i64
        %107 = arith.addi %2, %82 : i64
        %108 = arith.muli %2, %c2_i64 : i64
        %109 = arith.addi %108, %82 : i64
        %110 = arith.cmpi eq, %arg18, %c0_i64 : i64
        %111 = ascscalar.memref_address %alloc_26 : memref<128xbf16, 1>
        %112 = ascscalar.memref_address %alloc_27 : memref<128xbf16, 1>
        %113 = arith.select %110, %111, %112 : i64
        %114 = arith.select %110, %c7_i64, %c8_i64 : i64
        %115 = arith.addi %arg18, %c1_i64 : i64
        %116 = arith.remui %115, %c2_i64 : i64
        %117 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
        %118 = cannir.extract_shape %117 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 128)>
        %119 = cannir.extract_dim %118[2] : !cannir.shape<(-1, -1, -1, 128)>
        %120 = arith.minsi %106, %119 : i64
        %121 = arith.subi %119, %120 : i64
        %122 = arith.minsi %121, %c1_i64 : i64
        %123 = arith.cmpi sle, %106, %119 : i64
        %124 = arith.select %123, %122, %c0_i64 : i64
        %125 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
        %126 = cannir.extract_stride %125 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
        %127 = cannir.extract_dim %126[2] : !cannir.stride<(-1, -1, -1, 1)>
        %128 = arith.muli %127, %c2_i64 : i64
        %129 = cannir.extract_dim %126[0] : !cannir.stride<(-1, -1, -1, 1)>
        %130 = arith.muli %86, %129 : i64
        %131 = arith.muli %130, %c2_i64 : i64
        %132 = cannir.extract_dim %126[1] : !cannir.stride<(-1, -1, -1, 1)>
        %133 = arith.muli %82, %132 : i64
        %134 = arith.muli %133, %c2_i64 : i64
        %135 = arith.addi %131, %134 : i64
        %136 = arith.muli %106, %127 : i64
        %137 = arith.muli %136, %c2_i64 : i64
        %138 = arith.addi %135, %137 : i64
        %139 = ascscalar.bind_local_ptr %113 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %114, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%139, %alloc, %c0_i32, %c0_i32, %c0_i64, %138, %124, %c256_i64, %128, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %114, 0 {cannir.buffer.generated_sync}
        %140 = ascscalar.memref_address %alloc_28 : memref<128xbf16, 1>
        %141 = ascscalar.memref_address %alloc_29 : memref<128xbf16, 1>
        %142 = arith.select %110, %140, %141 : i64
        %143 = arith.select %110, %c9_i64, %c10_i64 : i64
        %144 = arith.muli %107, %132 : i64
        %145 = arith.muli %144, %c2_i64 : i64
        %146 = arith.addi %131, %145 : i64
        %147 = arith.addi %146, %137 : i64
        %148 = ascscalar.bind_local_ptr %142 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %143, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%148, %alloc, %c0_i32, %c0_i32, %c0_i64, %147, %124, %c256_i64, %128, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %143, 0 {cannir.buffer.generated_sync}
        %149 = ascscalar.memref_address %alloc_30 : memref<128xbf16, 1>
        %150 = ascscalar.memref_address %alloc_31 : memref<128xbf16, 1>
        %151 = arith.select %110, %149, %150 : i64
        %152 = arith.select %110, %c11_i64, %c12_i64 : i64
        %153 = cannir.extract_dim %1[2] : !cannir.shape<(-1, -1, -1, 128)>
        %154 = arith.minsi %106, %153 : i64
        %155 = arith.subi %153, %154 : i64
        %156 = arith.minsi %155, %c1_i64 : i64
        %157 = arith.cmpi sle, %106, %153 : i64
        %158 = arith.select %157, %156, %c0_i64 : i64
        %159 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
        %160 = cannir.extract_stride %159 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
        %161 = cannir.extract_dim %160[2] : !cannir.stride<(-1, -1, -1, 1)>
        %162 = arith.muli %161, %c2_i64 : i64
        %163 = cannir.extract_dim %160[0] : !cannir.stride<(-1, -1, -1, 1)>
        %164 = arith.muli %86, %163 : i64
        %165 = arith.muli %164, %c2_i64 : i64
        %166 = cannir.extract_dim %160[1] : !cannir.stride<(-1, -1, -1, 1)>
        %167 = arith.muli %82, %166 : i64
        %168 = arith.muli %167, %c2_i64 : i64
        %169 = arith.addi %165, %168 : i64
        %170 = arith.muli %106, %161 : i64
        %171 = arith.muli %170, %c2_i64 : i64
        %172 = arith.addi %169, %171 : i64
        %173 = ascscalar.bind_local_ptr %151 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %152, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%173, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %172, %158, %c256_i64, %162, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %152, 0 {cannir.buffer.generated_sync}
        %174 = ascscalar.memref_address %alloc_32 : memref<64xbf16, 1>
        %175 = ascscalar.memref_address %alloc_33 : memref<64xbf16, 1>
        %176 = arith.select %110, %174, %175 : i64
        %177 = arith.select %110, %c13_i64, %c14_i64 : i64
        %178 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>
        %179 = cannir.extract_shape %178 : <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
        %180 = cannir.extract_dim %179[2] : !cannir.shape<(-1, -1, -1, 1)>
        %181 = arith.minsi %106, %180 : i64
        %182 = arith.subi %180, %181 : i64
        %183 = arith.minsi %182, %c1_i64 : i64
        %184 = arith.cmpi sle, %106, %180 : i64
        %185 = arith.select %184, %183, %c0_i64 : i64
        %186 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>
        %187 = cannir.extract_stride %186 : <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
        %188 = cannir.extract_dim %187[2] : !cannir.stride<(-1, -1, -1, 1)>
        %189 = arith.muli %188, %c2_i64 : i64
        %190 = cannir.extract_dim %187[0] : !cannir.stride<(-1, -1, -1, 1)>
        %191 = arith.muli %86, %190 : i64
        %192 = arith.muli %191, %c2_i64 : i64
        %193 = cannir.extract_dim %187[1] : !cannir.stride<(-1, -1, -1, 1)>
        %194 = arith.muli %82, %193 : i64
        %195 = arith.muli %194, %c2_i64 : i64
        %196 = arith.addi %192, %195 : i64
        %197 = arith.muli %106, %188 : i64
        %198 = arith.muli %197, %c2_i64 : i64
        %199 = arith.addi %196, %198 : i64
        %200 = ascscalar.bind_local_ptr %176 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %177, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%200, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %199, %185, %c2_i64, %189, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %177, 0 {cannir.buffer.generated_sync}
        %201 = ascscalar.memref_address %alloc_19 : memref<128xbf16, 1>
        %202 = ascscalar.memref_address %alloc_20 : memref<128xbf16, 1>
        %203 = arith.select %110, %201, %202 : i64
        %204 = arith.select %110, %c3_i64, %c4_i64 : i64
        %205 = arith.muli %109, %132 : i64
        %206 = arith.muli %205, %c2_i64 : i64
        %207 = arith.addi %131, %206 : i64
        %208 = arith.addi %207, %137 : i64
        %209 = ascscalar.bind_local_ptr %203 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %204, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%209, %alloc, %c0_i32, %c0_i32, %c0_i64, %208, %124, %c256_i64, %128, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %204, 0 {cannir.buffer.generated_sync}
        %210:2 = scf.for %arg19 = %87#0 to %87#1 step %c1_i64 iter_args(%arg20 = %arg17, %arg21 = %116) -> (i64, i64)  : i64 {
          %211 = arith.subi %arg19, %87#0 : i64
          %212 = arith.select %85, %arg19, %211 : i64
          %213 = arith.cmpi eq, %arg20, %c0_i64 : i64
          %214 = arith.select %213, %111, %112 : i64
          %215 = ascscalar.bind_local_ptr %214 : memref<128xbf16, 1>
          %216 = arith.select %213, %c7_i64, %c8_i64 : i64
          %217 = arith.addi %arg20, %c1_i64 overflow<nsw> : i64
          %218 = arith.remui %217, %c2_i64 : i64
          %219 = arith.select %213, %140, %141 : i64
          %220 = ascscalar.bind_local_ptr %219 : memref<128xbf16, 1>
          %221 = arith.select %213, %c9_i64, %c10_i64 : i64
          ascvec.get_buf v, %216, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %221, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%215, %220, %alloc_12, %alloc_11, %alloc_23 : memref<128xbf16, 1>, memref<128xbf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<4xf32, 1>) {
          ^bb0(%arg22: memref<128xbf16, 1>, %arg23: memref<128xbf16, 1>, %arg24: memref<128xf32, 1>, %arg25: memref<128xf32, 1>, %arg26: memref<4xf32, 1>):
            %c3 = arith.constant 3 : index
            %c2_90 = arith.constant 2 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c0_91 = arith.constant 0 : index
            %c64_i32_92 = arith.constant 64 : i32
            %c0_93 = arith.constant 0 : index
            %c1_94 = arith.constant 1 : index
            scf.for %arg27 = %c0_93 to %c1_94 step %c1_94 {
              %mask, %remaining_out = ascvec.update_mask %c64_i32_92 : !ascvec.mask
              %287 = ascvec.loadalign_unpack %arg22[%c0_91] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %288 = ascvec.loadalign_unpack %arg22[%c64] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %289 = ascvec.loadalign_unpack %arg23[%c0_91] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %290 = ascvec.loadalign_unpack %arg23[%c64] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %291 = ascvec.cast %287, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %292 = ascvec.cast %288, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %293 = ascvec.cast %289, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %294 = ascvec.cast %290, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %295 = ascvec.mul %291, %291, %mask : <f32>
              %296 = ascvec.mul %292, %292, %mask : <f32>
              %297 = ascvec.mul %293, %293, %mask : <f32>
              %298 = ascvec.mul %294, %294, %mask : <f32>
              %299 = ascvec.reduce_sum %295, %mask : <f32>
              %300 = ascvec.reduce_sum %296, %mask : <f32>
              %301 = ascvec.reduce_sum %297, %mask : <f32>
              %302 = ascvec.reduce_sum %298, %mask : <f32>
              ascvec.storealign %arg24[%c0_91], %291, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg24[%c64], %292, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg25[%c0_91], %293, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg25[%c64], %294, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign_1st %arg26[%c0_91], %299 : memref<4xf32, 1>, index, <f32>
              ascvec.storealign_1st %arg26[%c1], %300 : memref<4xf32, 1>, index, <f32>
              ascvec.storealign_1st %arg26[%c2_90], %301 : memref<4xf32, 1>, index, <f32>
              ascvec.storealign_1st %arg26[%c3], %302 : memref<4xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %221, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %216, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_23 : memref<4xf32, 1>) {
          ^bb0(%arg22: memref<4xf32, 1>):
            %c3 = arith.constant 3 : index
            %c2_90 = arith.constant 2 : index
            %c1 = arith.constant 1 : index
            %cst = arith.constant 9.99999997E-7 : f32
            %c0_91 = arith.constant 0 : index
            %c1_i32_92 = arith.constant 1 : i32
            %c0_93 = arith.constant 0 : index
            %c1_94 = arith.constant 1 : index
            scf.for %arg23 = %c0_93 to %c1_94 step %c1_94 {
              %mask, %remaining_out = ascvec.update_mask %c1_i32_92 : !ascvec.mask
              %287 = ascvec.loadalign_brc %arg22[%c0_91] : memref<4xf32, 1>, index -> <f32>
              %288 = ascvec.loadalign_brc %arg22[%c1] : memref<4xf32, 1>, index -> <f32>
              %289 = ascvec.loadalign_brc %arg22[%c2_90] : memref<4xf32, 1>, index -> <f32>
              %290 = ascvec.loadalign_brc %arg22[%c3] : memref<4xf32, 1>, index -> <f32>
              %291 = ascvec.add %287, %288, %mask : <f32>
              %292 = ascvec.add %289, %290, %mask : <f32>
              %293 = ascvec.add_scalar %291, %cst, %mask : <f32>, f32
              %294 = ascvec.add_scalar %292, %cst, %mask : <f32>, f32
              %295 = ascvec.sqrt %293, %mask : <f32>
              %296 = ascvec.sqrt %294, %mask : <f32>
              ascvec.storealign_1st %arg22[%c0_91], %295 : memref<4xf32, 1>, index, <f32>
              ascvec.storealign_1st %arg22[%c1], %296 : memref<4xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.vf(%alloc_23, %alloc_12, %alloc_11, %arg11 : memref<4xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, f32) {
          ^bb0(%arg22: memref<4xf32, 1>, %arg23: memref<128xf32, 1>, %arg24: memref<128xf32, 1>, %arg25: f32):
            %c64 = arith.constant 64 : index
            %c1 = arith.constant 1 : index
            %c0_90 = arith.constant 0 : index
            %c64_i32_91 = arith.constant 64 : i32
            %c0_92 = arith.constant 0 : index
            %c1_93 = arith.constant 1 : index
            scf.for %arg26 = %c0_92 to %c1_93 step %c1_93 {
              %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
              %287 = ascvec.loadalign_brc %arg22[%c0_90] : memref<4xf32, 1>, index -> <f32>
              %288 = ascvec.loadalign_brc %arg22[%c1] : memref<4xf32, 1>, index -> <f32>
              %289 = ascvec.duplicate %287, %mask : <f32> -> <f32>
              %290 = ascvec.duplicate %288, %mask : <f32> -> <f32>
              %291 = ascvec.loadalign %arg23[%c0_90] : memref<128xf32, 1>, index -> <f32>
              %292 = ascvec.loadalign %arg23[%c64] : memref<128xf32, 1>, index -> <f32>
              %293 = ascvec.loadalign %arg24[%c0_90] : memref<128xf32, 1>, index -> <f32>
              %294 = ascvec.loadalign %arg24[%c64] : memref<128xf32, 1>, index -> <f32>
              %295 = ascvec.div %291, %289, %mask : <f32>
              %296 = ascvec.div %292, %289, %mask : <f32>
              %297 = ascvec.div %293, %290, %mask : <f32>
              %298 = ascvec.div %294, %290, %mask : <f32>
              %299 = ascvec.mul_scalar %295, %arg25, %mask : <f32>, f32
              %300 = ascvec.mul_scalar %296, %arg25, %mask : <f32>, f32
              ascvec.storealign %arg23[%c0_90], %299, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg23[%c64], %300, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg24[%c0_90], %297, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg24[%c64], %298, %mask : memref<128xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          %222 = arith.select %213, %149, %150 : i64
          %223 = ascscalar.bind_local_ptr %222 : memref<128xbf16, 1>
          %224 = arith.select %213, %c11_i64, %c12_i64 : i64
          %225 = arith.select %213, %174, %175 : i64
          %226 = ascscalar.bind_local_ptr %225 : memref<64xbf16, 1>
          %227 = arith.select %213, %c13_i64, %c14_i64 : i64
          %228 = arith.select %213, %201, %202 : i64
          %229 = ascscalar.bind_local_ptr %228 : memref<128xbf16, 1>
          %230 = arith.select %213, %c3_i64, %c4_i64 : i64
          %231 = arith.trunci %82 : i64 to i32
          %232 = arith.index_cast %231 : i32 to index
          %233 = arith.muli %231, %c128_i32 : i32
          %234 = arith.index_cast %233 : i32 to index
          %235 = arith.addi %233, %c64_i32 : i32
          %236 = arith.index_cast %235 : i32 to index
          ascvec.get_buf v, %224, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %227, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %230, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_35, %232, %223, %alloc_34, %234, %236, %arg12, %alloc_13, %226, %alloc_15, %229, %alloc_14 : memref<96xf32, 1>, index, memref<128xbf16, 1>, memref<12288xf32, 1>, index, index, f32, memref<128xf32, 1>, memref<64xbf16, 1>, memref<64xf32, 1>, memref<128xbf16, 1>, memref<128xf32, 1>) {
          ^bb0(%arg22: memref<96xf32, 1>, %arg23: index, %arg24: memref<128xbf16, 1>, %arg25: memref<12288xf32, 1>, %arg26: index, %arg27: index, %arg28: f32, %arg29: memref<128xf32, 1>, %arg30: memref<64xbf16, 1>, %arg31: memref<64xf32, 1>, %arg32: memref<128xbf16, 1>, %arg33: memref<128xf32, 1>):
            %c64 = arith.constant 64 : index
            %c0_90 = arith.constant 0 : index
            %cst = arith.constant 1.000000e+00 : f32
            %c64_i32_91 = arith.constant 64 : i32
            %c0_92 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg34 = %c0_92 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
              %287 = ascvec.duplicate_scalar_nomask %cst : f32 -> <f32>
              %288 = ascvec.loadalign_brc %arg22[%arg23] : memref<96xf32, 1>, index -> <f32>
              %289 = ascvec.exp %288, %mask : <f32>
              %290 = ascvec.neg %289, %mask : <f32>
              %291 = ascvec.loadalign_unpack %arg24[%c0_90] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %292 = ascvec.loadalign_unpack %arg24[%c64] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %293 = ascvec.cast %291, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %294 = ascvec.cast %292, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %295 = ascvec.loadalign %arg25[%arg26] : memref<12288xf32, 1>, index -> <f32>
              %296 = ascvec.loadalign %arg25[%arg27] : memref<12288xf32, 1>, index -> <f32>
              %297 = ascvec.add %293, %295, %mask : <f32>
              %298 = ascvec.add %294, %296, %mask : <f32>
              %299 = ascvec.mul %290, %297, %mask : <f32>
              %300 = ascvec.mul %290, %298, %mask : <f32>
              %301 = ascvec.exp %299, %mask : <f32>
              %302 = ascvec.exp %300, %mask : <f32>
              %303 = ascvec.add_scalar %301, %cst, %mask : <f32>, f32
              %304 = ascvec.add_scalar %302, %cst, %mask : <f32>, f32
              %305 = ascvec.div %287, %303, %mask : <f32>
              %306 = ascvec.div %287, %304, %mask : <f32>
              %307 = ascvec.mul_scalar %305, %arg28, %mask : <f32>, f32
              %308 = ascvec.mul_scalar %306, %arg28, %mask : <f32>, f32
              %309 = ascvec.exp %307, %mask : <f32>
              %310 = ascvec.exp %308, %mask : <f32>
              ascvec.storealign %arg29[%c0_90], %309, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg29[%c64], %310, %mask : memref<128xf32, 1>, index, <f32>
              %311 = ascvec.loadalign_unpack %arg30[%c0_90] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
              %312 = ascvec.cast %311, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %313 = ascvec.neg %312, %mask : <f32>
              %314 = ascvec.exp %313, %mask : <f32>
              %315 = ascvec.add_scalar %314, %cst, %mask : <f32>, f32
              %316 = ascvec.div %287, %315, %mask : <f32>
              ascvec.storealign %arg31[%c0_90], %316, %mask : memref<64xf32, 1>, index, <f32>
              %317 = ascvec.loadalign_unpack %arg32[%c0_90] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %318 = ascvec.cast %317, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              ascvec.storealign %arg33[%c0_90], %318, %mask : memref<128xf32, 1>, index, <f32>
              %319 = ascvec.loadalign_unpack %arg32[%c64] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %320 = ascvec.cast %319, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              ascvec.storealign %arg33[%c64], %320, %mask : memref<128xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %230, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %227, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %224, 0 {cannir.buffer.generated_sync}
          %237 = arith.addi %arg19, %c1_i64 : i64
          %238 = arith.cmpi slt, %237, %87#1 : i64
          %239 = scf.if %238 -> (i64) {
            %287 = arith.subi %237, %87#0 : i64
            %288 = arith.select %85, %237, %287 : i64
            %289 = arith.cmpi eq, %arg21, %c0_i64 : i64
            %290 = arith.select %289, %111, %112 : i64
            %291 = arith.select %289, %c7_i64, %c8_i64 : i64
            %292 = arith.addi %arg21, %c1_i64 overflow<nsw> : i64
            %293 = arith.remui %292, %c2_i64 : i64
            %294 = arith.minsi %288, %119 : i64
            %295 = arith.subi %119, %294 : i64
            %296 = arith.minsi %295, %c1_i64 : i64
            %297 = arith.cmpi sle, %288, %119 : i64
            %298 = arith.select %297, %296, %c0_i64 : i64
            %299 = arith.muli %288, %127 : i64
            %300 = arith.muli %299, %c2_i64 : i64
            %301 = arith.addi %135, %300 : i64
            %302 = ascscalar.bind_local_ptr %290 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %291, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%302, %alloc, %c0_i32, %c0_i32, %c0_i64, %301, %298, %c256_i64, %128, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %291, 0 {cannir.buffer.generated_sync}
            %303 = arith.select %289, %140, %141 : i64
            %304 = arith.select %289, %c9_i64, %c10_i64 : i64
            %305 = arith.addi %146, %300 : i64
            %306 = ascscalar.bind_local_ptr %303 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %304, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%306, %alloc, %c0_i32, %c0_i32, %c0_i64, %305, %298, %c256_i64, %128, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %304, 0 {cannir.buffer.generated_sync}
            %307 = arith.select %289, %149, %150 : i64
            %308 = arith.select %289, %c11_i64, %c12_i64 : i64
            %309 = arith.minsi %288, %153 : i64
            %310 = arith.subi %153, %309 : i64
            %311 = arith.minsi %310, %c1_i64 : i64
            %312 = arith.cmpi sle, %288, %153 : i64
            %313 = arith.select %312, %311, %c0_i64 : i64
            %314 = arith.muli %288, %161 : i64
            %315 = arith.muli %314, %c2_i64 : i64
            %316 = arith.addi %169, %315 : i64
            %317 = ascscalar.bind_local_ptr %307 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %308, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%317, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %316, %313, %c256_i64, %162, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %308, 0 {cannir.buffer.generated_sync}
            %318 = arith.select %289, %174, %175 : i64
            %319 = arith.select %289, %c13_i64, %c14_i64 : i64
            %320 = arith.minsi %288, %180 : i64
            %321 = arith.subi %180, %320 : i64
            %322 = arith.minsi %321, %c1_i64 : i64
            %323 = arith.cmpi sle, %288, %180 : i64
            %324 = arith.select %323, %322, %c0_i64 : i64
            %325 = arith.muli %288, %188 : i64
            %326 = arith.muli %325, %c2_i64 : i64
            %327 = arith.addi %196, %326 : i64
            %328 = ascscalar.bind_local_ptr %318 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %319, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%328, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %327, %324, %c2_i64, %189, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %319, 0 {cannir.buffer.generated_sync}
            %329 = arith.select %289, %201, %202 : i64
            %330 = arith.select %289, %c3_i64, %c4_i64 : i64
            %331 = arith.addi %207, %300 : i64
            %332 = ascscalar.bind_local_ptr %329 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %330, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%332, %alloc, %c0_i32, %c0_i32, %c0_i64, %331, %298, %c256_i64, %128, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %330, 0 {cannir.buffer.generated_sync}
            scf.yield %293 : i64
          } else {
            scf.yield %arg21 : i64
          }
          %240 = ascscalar.memref_address %alloc_17 : memref<16384xf32, 1>
          %241 = ascscalar.memref_address %alloc_18 : memref<16384xf32, 1>
          %242 = arith.select %213, %240, %241 : i64
          %243 = ascscalar.bind_local_ptr %242 : memref<16384xf32, 1>
          %244 = arith.select %213, %c1_i64, %c2_i64 : i64
          ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_13, %alloc_11, %alloc_10, %alloc_24 : memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>, memref<128xf32, 1>) {
          ^bb0(%arg22: memref<128xf32, 1>, %arg23: memref<128xf32, 1>, %arg24: memref<16384xf32, 1>, %arg25: memref<128xf32, 1>):
            %c64 = arith.constant 64 : index
            %c1_i32_90 = arith.constant 1 : i32
            %c0_i32_91 = arith.constant 0 : i32
            %c128_i32_92 = arith.constant 128 : i32
            %c0_93 = arith.constant 0 : index
            %c64_i32_94 = arith.constant 64 : i32
            %mask, %remaining_out = ascvec.update_mask %c64_i32_94 : !ascvec.mask
            %287 = ascvec.loadalign %arg22[%c0_93] : memref<128xf32, 1>, index -> <f32>
            %288 = ascvec.loadalign %arg22[%c64] : memref<128xf32, 1>, index -> <f32>
            %289 = ascvec.loadalign %arg23[%c0_93] : memref<128xf32, 1>, index -> <f32>
            %290 = ascvec.loadalign %arg23[%c64] : memref<128xf32, 1>, index -> <f32>
            %291 = ascvec.mul %287, %289, %mask : <f32>
            %292 = ascvec.mul %288, %290, %mask : <f32>
            scf.for %arg26 = %c0_i32_91 to %c128_i32_92 step %c1_i32_90  : i32 {
              %293 = arith.muli %arg26, %c128_i32_92 overflow<nsw> : i32
              %294 = arith.addi %293, %c64_i32_94 overflow<nsw> : i32
              %295 = arith.index_cast %293 : i32 to index
              %296 = ascvec.loadalign %arg24[%295] : memref<16384xf32, 1>, index -> <f32>
              %297 = arith.index_cast %294 : i32 to index
              %298 = ascvec.loadalign %arg24[%297] : memref<16384xf32, 1>, index -> <f32>
              %299 = ascvec.mul %296, %291, %mask : <f32>
              %300 = ascvec.madd_two_address %298, %292, %299, %mask : <f32>
              %301 = ascvec.reduce_sum %300, %mask : <f32>
              %302 = arith.index_cast %arg26 : i32 to index
              ascvec.storealign_1st %arg25[%302], %301 : memref<128xf32, 1>, index, <f32>
            } {cannir.loop_unroll = 4 : i64}
          } {cannir.vf_mode = "simd"}
          ascvec.vf(%alloc_15, %alloc_24, %alloc_14, %alloc_25 : memref<64xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>) {
          ^bb0(%arg22: memref<64xf32, 1>, %arg23: memref<128xf32, 1>, %arg24: memref<128xf32, 1>, %arg25: memref<128xf32, 1>):
            %c64 = arith.constant 64 : index
            %c0_90 = arith.constant 0 : index
            %c64_i32_91 = arith.constant 64 : i32
            %c0_92 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg26 = %c0_92 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
              %287 = ascvec.loadalign_brc %arg22[%c0_90] : memref<64xf32, 1>, index -> <f32>
              %288 = ascvec.loadalign %arg23[%c0_90] : memref<128xf32, 1>, index -> <f32>
              %289 = ascvec.loadalign %arg24[%c0_90] : memref<128xf32, 1>, index -> <f32>
              %290 = ascvec.sub %289, %288, %mask : <f32>
              %291 = ascvec.mul %290, %287, %mask : <f32>
              ascvec.storealign %arg25[%c0_90], %291, %mask : memref<128xf32, 1>, index, <f32>
              %292 = ascvec.loadalign %arg23[%c64] : memref<128xf32, 1>, index -> <f32>
              %293 = ascvec.loadalign %arg24[%c64] : memref<128xf32, 1>, index -> <f32>
              %294 = ascvec.sub %293, %292, %mask : <f32>
              %295 = ascvec.mul %294, %287, %mask : <f32>
              ascvec.storealign %arg25[%c64], %295, %mask : memref<128xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.get_buf v, %244, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_13, %alloc_11, %alloc_12, %alloc_10, %alloc_25, %243, %alloc_16 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>, memref<128xf32, 1>) {
          ^bb0(%arg22: memref<128xf32, 1>, %arg23: memref<128xf32, 1>, %arg24: memref<128xf32, 1>, %arg25: memref<16384xf32, 1>, %arg26: memref<128xf32, 1>, %arg27: memref<16384xf32, 1>, %arg28: memref<128xf32, 1>):
            %c64 = arith.constant 64 : index
            %c1_i32_90 = arith.constant 1 : i32
            %c0_i32_91 = arith.constant 0 : i32
            %c128_i32_92 = arith.constant 128 : i32
            %c0_93 = arith.constant 0 : index
            %c64_i32_94 = arith.constant 64 : i32
            %mask, %remaining_out = ascvec.update_mask %c64_i32_94 : !ascvec.mask
            %287 = ascvec.loadalign %arg22[%c0_93] : memref<128xf32, 1>, index -> <f32>
            %288 = ascvec.loadalign %arg22[%c64] : memref<128xf32, 1>, index -> <f32>
            %289 = ascvec.loadalign %arg23[%c0_93] : memref<128xf32, 1>, index -> <f32>
            %290 = ascvec.loadalign %arg23[%c64] : memref<128xf32, 1>, index -> <f32>
            %291 = ascvec.loadalign %arg24[%c0_93] : memref<128xf32, 1>, index -> <f32>
            %292 = ascvec.loadalign %arg24[%c64] : memref<128xf32, 1>, index -> <f32>
            scf.for %arg29 = %c0_i32_91 to %c128_i32_92 step %c1_i32_90  : i32 {
              %293 = arith.muli %arg29, %c128_i32_92 overflow<nsw> : i32
              %294 = arith.addi %293, %c64_i32_94 overflow<nsw> : i32
              %295 = arith.index_cast %293 : i32 to index
              %296 = ascvec.loadalign %arg25[%295] : memref<16384xf32, 1>, index -> <f32>
              %297 = arith.index_cast %294 : i32 to index
              %298 = ascvec.loadalign %arg25[%297] : memref<16384xf32, 1>, index -> <f32>
              %299 = arith.index_cast %arg29 : i32 to index
              %300 = ascvec.loadalign_brc %arg26[%299] : memref<128xf32, 1>, index -> <f32>
              %301 = ascvec.mul %300, %289, %mask : <f32>
              %302 = ascvec.mul %300, %290, %mask : <f32>
              %303 = ascvec.madd_two_address %296, %287, %301, %mask : <f32>
              %304 = ascvec.madd_two_address %298, %288, %302, %mask : <f32>
              ascvec.storealign %arg25[%295], %303, %mask : memref<16384xf32, 1>, index, <f32>
              ascvec.storealign %arg25[%297], %304, %mask : memref<16384xf32, 1>, index, <f32>
              ascvec.storealign %arg27[%295], %303, %mask : memref<16384xf32, 1>, index, <f32>
              ascvec.storealign %arg27[%297], %304, %mask : memref<16384xf32, 1>, index, <f32>
              %305 = ascvec.mul %303, %291, %mask : <f32>
              %306 = ascvec.madd_two_address %304, %292, %305, %mask : <f32>
              %307 = ascvec.reduce_sum %306, %mask : <f32>
              ascvec.storealign_1st %arg28[%299], %307 : memref<128xf32, 1>, index, <f32>
            } {cannir.loop_unroll = 1 : i64}
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %244, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          %245 = ascscalar.memref_address %alloc_21 : memref<128xbf16, 1>
          %246 = ascscalar.memref_address %alloc_22 : memref<128xbf16, 1>
          %247 = arith.select %213, %245, %246 : i64
          %248 = ascscalar.bind_local_ptr %247 : memref<128xbf16, 1>
          %249 = arith.select %213, %c5_i64, %c6_i64 : i64
          ascvec.get_buf v, %249, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_16, %248 : memref<128xf32, 1>, memref<128xbf16, 1>) {
          ^bb0(%arg22: memref<128xf32, 1>, %arg23: memref<128xbf16, 1>):
            %c64_i32_90 = arith.constant 64 : i32
            %c0_i32_91 = arith.constant 0 : i32
            %c128_i32_92 = arith.constant 128 : i32
            scf.for %arg24 = %c0_i32_91 to %c128_i32_92 step %c64_i32_90  : i32 {
              %287 = arith.subi %c128_i32_92, %arg24 overflow<nsw> : i32
              %mask, %remaining_out = ascvec.update_mask %287 : !ascvec.mask
              %288 = arith.index_cast %arg24 : i32 to index
              %289 = ascvec.loadalign %arg22[%288] : memref<128xf32, 1>, index -> <f32>
              %290 = ascvec.cast %289, %mask : <f32> -> <bf16>
              ascvec.storealign_pack %arg23[%288], %290, %mask {mode = "b32_to_b16"} : memref<128xbf16, 1>, index, <bf16>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %249, 0 {cannir.buffer.generated_sync}
          %250 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
          %251 = cannir.extract_stride %250 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
          %252 = cannir.extract_dim %251[0] : !cannir.stride<(-1, -1, -1, 1)>
          %253 = arith.muli %86, %252 : i64
          %254 = cannir.extract_dim %251[1] : !cannir.stride<(-1, -1, -1, 1)>
          %255 = arith.muli %82, %254 : i64
          %256 = arith.addi %253, %255 : i64
          %257 = cannir.extract_dim %251[2] : !cannir.stride<(-1, -1, -1, 1)>
          %258 = arith.muli %212, %257 : i64
          %259 = arith.addi %256, %258 : i64
          %260 = arith.muli %259, %c16_i64 : i64
          %261 = ascscalar.memref_address %alloc_4 : memref<1xbf16>
          %262 = arith.divui %260, %c8_i64 : i64
          %263 = arith.addi %261, %262 : i64
          %264 = ascscalar.bind_gm_ptr %263 : memref<1xbf16>
          %265 = ascscalar.bind_local_ptr %247 : memref<1xbf16, 1>
          %266 = cannir.extract_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
          %267 = cannir.extract_stride %266 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
          %268 = cannir.extract_dim %267[2] : !cannir.stride<(-1, -1, -1, 1)>
          %269 = arith.index_cast %268 : i64 to index
          %270 = arith.muli %269, %c2 : index
          %271 = arith.index_cast %270 : index to i64
          ascvec.get_buf mte3, %249, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %264[%c0], %265[%c0], %c1_i32, %c256_i32, %271, %c256_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
          ascvec.release_buf mte3, %249, 0 {cannir.buffer.generated_sync}
          %272 = arith.index_cast %arg19 : i64 to index
          %273 = memref.load %alloc_7[%272] : memref<1xi32>
          %274 = arith.extsi %273 : i32 to i64
          %275 = cannir.extract_physical_layout %arg7 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
          %276 = cannir.extract_stride %275 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
          %277 = cannir.extract_dim %276[0] : !cannir.stride<(-1, 16384, 128, 1)>
          %278 = arith.muli %274, %277 : i64
          %279 = arith.muli %82, %c16384_i64 : i64
          %280 = arith.addi %278, %279 : i64
          %281 = arith.muli %280, %c32_i64 : i64
          %282 = ascscalar.memref_address %alloc_6 : memref<1xf32>
          %283 = arith.divui %281, %c8_i64 : i64
          %284 = arith.addi %282, %283 : i64
          %285 = ascscalar.bind_gm_ptr %284 : memref<1xf32>
          %286 = ascscalar.bind_local_ptr %242 : memref<1xf32, 1>
          ascvec.get_buf mte3, %244, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %285[%c0], %286[%c0], %c1_i32, %c65536_i32, %c0_i64, %c65536_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
          ascvec.release_buf mte3, %244, 0 {cannir.buffer.generated_sync}
          scf.yield %218, %239 : i64, i64
        }
        scf.yield %210#0, %210#1 : i64, i64
      }
      %58 = arith.muli %22, %41 : i64
      %59 = arith.muli %34, %50 : i64
      %60 = arith.addi %58, %59 : i64
      %61 = arith.select %38, %c2_i64, %c0_i64 : i64
      %62 = arith.addi %35, %61 overflow<nsw> : i64
      %63 = arith.cmpi eq, %62, %c0_i64 : i64
      %64 = arith.select %63, %51, %false : i1
      %65:6 = scf.if %64 -> (i64, i64, i64, i64, i64, i64) {
        %69 = arith.remsi %21, %c2_i64 : i64
        %70 = arith.cmpi ne, %69, %c0_i64 : i64
        %71 = arith.cmpi slt, %69, %c0_i64 : i64
        %72 = arith.andi %70, %71 : i1
        %73 = arith.divsi %21, %c2_i64 : i64
        %74 = arith.extui %72 : i1 to i64
        %75 = arith.subi %73, %74 : i64
        %76 = arith.addi %60, %75 : i64
        %77 = arith.select %72, %c2_i64, %c0_i64 : i64
        %78 = arith.addi %69, %77 overflow<nsw> : i64
        %79 = arith.maxsi %2, %c1_i64 : i64
        %80 = arith.remsi %76, %79 : i64
        %81 = arith.cmpi slt, %80, %c0_i64 : i64
        %82 = arith.divsi %76, %79 : i64
        %83 = arith.extui %81 : i1 to i64
        %84 = arith.subi %82, %83 : i64
        %85 = arith.maxsi %arg14, %c1_i64 : i64
        %86 = arith.remsi %84, %85 : i64
        %87 = arith.cmpi slt, %86, %c0_i64 : i64
        %88 = arith.select %87, %85, %c0_i64 : i64
        %89 = arith.addi %86, %88 : i64
        %90 = arith.select %81, %79, %c0_i64 : i64
        %91 = arith.addi %80, %90 : i64
        %92 = arith.muli %89, %arg15 : i64
        %93 = arith.addi %92, %arg15 : i64
        %94 = arith.cmpi eq, %arg13, %c1_i64 : i64
        %95 = arith.select %94, %c0_i64, %89 : i64
        %96:2 = scf.if %94 -> (i64, i64) {
          %224 = arith.index_cast %89 : i64 to index
          %225 = memref.load %alloc_9[%224] : memref<1xi32>
          %226 = arith.addi %89, %c1_i64 : i64
          %227 = arith.index_cast %226 : i64 to index
          %228 = memref.load %alloc_9[%227] : memref<1xi32>
          %229 = arith.extsi %225 : i32 to i64
          %230 = arith.extsi %228 : i32 to i64
          scf.yield %229, %230 : i64, i64
        } else {
          scf.yield %92, %93 : i64, i64
        }
        %97 = arith.index_cast %89 : i64 to index
        %98 = memref.load %alloc_8[%97] : memref<1xi32>
        %99 = arith.cmpi sgt, %98, %c0_i32 : i32
        %100 = arith.extsi %98 : i32 to i64
        %101 = arith.addi %96#0, %100 : i64
        %102 = arith.subi %101, %c1_i64 : i64
        %103 = arith.select %99, %102, %96#0 : i64
        %104 = arith.index_cast %103 : i64 to index
        %105 = memref.load %alloc_7[%104] : memref<1xi32>
        %106 = arith.extsi %105 : i32 to i64
        %107 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
        %108 = cannir.extract_physical_layout %arg6 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
        %109 = cannir.extract_stride %108 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
        %110 = cannir.extract_dim %109[0] : !cannir.stride<(-1, 16384, 128, 1)>
        %111 = arith.muli %106, %110 : i64
        %112 = arith.muli %111, %c4_i64 : i64
        %113 = arith.muli %91, %c65536_i64 : i64
        %114 = arith.addi %112, %113 : i64
        %115 = arith.muli %78, %c32768_i64 overflow<nsw> : i64
        %116 = arith.addi %114, %115 : i64
        ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%107, %alloc_5, %c0_i32, %c0_i32, %c0_i64, %116, %c1_i64, %c32768_i64, %c0_i64, %c32768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 65536 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
        %117 = arith.select %94, %96#0, %c0_i64 : i64
        %118 = arith.addi %2, %91 : i64
        %119 = arith.muli %2, %c2_i64 : i64
        %120 = arith.addi %119, %91 : i64
        %121 = arith.cmpi eq, %57#1, %c0_i64 : i64
        %122 = ascscalar.memref_address %alloc_26 : memref<128xbf16, 1>
        %123 = ascscalar.memref_address %alloc_27 : memref<128xbf16, 1>
        %124 = arith.select %121, %122, %123 : i64
        %125 = arith.select %121, %c7_i64, %c8_i64 : i64
        %126 = arith.addi %57#1, %c1_i64 overflow<nsw> : i64
        %127 = arith.remui %126, %c2_i64 : i64
        %128 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
        %129 = cannir.extract_shape %128 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 128)>
        %130 = cannir.extract_dim %129[2] : !cannir.shape<(-1, -1, -1, 128)>
        %131 = arith.minsi %117, %130 : i64
        %132 = arith.subi %130, %131 : i64
        %133 = arith.minsi %132, %c1_i64 : i64
        %134 = arith.cmpi sle, %117, %130 : i64
        %135 = arith.select %134, %133, %c0_i64 : i64
        %136 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
        %137 = cannir.extract_stride %136 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
        %138 = cannir.extract_dim %137[2] : !cannir.stride<(-1, -1, -1, 1)>
        %139 = arith.muli %138, %c2_i64 : i64
        %140 = cannir.extract_dim %137[0] : !cannir.stride<(-1, -1, -1, 1)>
        %141 = arith.muli %95, %140 : i64
        %142 = arith.muli %141, %c2_i64 : i64
        %143 = cannir.extract_dim %137[1] : !cannir.stride<(-1, -1, -1, 1)>
        %144 = arith.muli %91, %143 : i64
        %145 = arith.muli %144, %c2_i64 : i64
        %146 = arith.addi %142, %145 : i64
        %147 = arith.muli %117, %138 : i64
        %148 = arith.muli %147, %c2_i64 : i64
        %149 = arith.addi %146, %148 : i64
        %150 = ascscalar.bind_local_ptr %124 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %125, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%150, %alloc, %c0_i32, %c0_i32, %c0_i64, %149, %135, %c256_i64, %139, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %125, 0 {cannir.buffer.generated_sync}
        %151 = ascscalar.memref_address %alloc_28 : memref<128xbf16, 1>
        %152 = ascscalar.memref_address %alloc_29 : memref<128xbf16, 1>
        %153 = arith.select %121, %151, %152 : i64
        %154 = arith.select %121, %c9_i64, %c10_i64 : i64
        %155 = arith.muli %118, %143 : i64
        %156 = arith.muli %155, %c2_i64 : i64
        %157 = arith.addi %142, %156 : i64
        %158 = arith.addi %157, %148 : i64
        %159 = ascscalar.bind_local_ptr %153 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %154, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%159, %alloc, %c0_i32, %c0_i32, %c0_i64, %158, %135, %c256_i64, %139, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %154, 0 {cannir.buffer.generated_sync}
        %160 = ascscalar.memref_address %alloc_30 : memref<128xbf16, 1>
        %161 = ascscalar.memref_address %alloc_31 : memref<128xbf16, 1>
        %162 = arith.select %121, %160, %161 : i64
        %163 = arith.select %121, %c11_i64, %c12_i64 : i64
        %164 = cannir.extract_dim %1[2] : !cannir.shape<(-1, -1, -1, 128)>
        %165 = arith.minsi %117, %164 : i64
        %166 = arith.subi %164, %165 : i64
        %167 = arith.minsi %166, %c1_i64 : i64
        %168 = arith.cmpi sle, %117, %164 : i64
        %169 = arith.select %168, %167, %c0_i64 : i64
        %170 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
        %171 = cannir.extract_stride %170 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
        %172 = cannir.extract_dim %171[2] : !cannir.stride<(-1, -1, -1, 1)>
        %173 = arith.muli %172, %c2_i64 : i64
        %174 = cannir.extract_dim %171[0] : !cannir.stride<(-1, -1, -1, 1)>
        %175 = arith.muli %95, %174 : i64
        %176 = arith.muli %175, %c2_i64 : i64
        %177 = cannir.extract_dim %171[1] : !cannir.stride<(-1, -1, -1, 1)>
        %178 = arith.muli %91, %177 : i64
        %179 = arith.muli %178, %c2_i64 : i64
        %180 = arith.addi %176, %179 : i64
        %181 = arith.muli %117, %172 : i64
        %182 = arith.muli %181, %c2_i64 : i64
        %183 = arith.addi %180, %182 : i64
        %184 = ascscalar.bind_local_ptr %162 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %163, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%184, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %183, %169, %c256_i64, %173, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %163, 0 {cannir.buffer.generated_sync}
        %185 = ascscalar.memref_address %alloc_32 : memref<64xbf16, 1>
        %186 = ascscalar.memref_address %alloc_33 : memref<64xbf16, 1>
        %187 = arith.select %121, %185, %186 : i64
        %188 = arith.select %121, %c13_i64, %c14_i64 : i64
        %189 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>
        %190 = cannir.extract_shape %189 : <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
        %191 = cannir.extract_dim %190[2] : !cannir.shape<(-1, -1, -1, 1)>
        %192 = arith.minsi %117, %191 : i64
        %193 = arith.subi %191, %192 : i64
        %194 = arith.minsi %193, %c1_i64 : i64
        %195 = arith.cmpi sle, %117, %191 : i64
        %196 = arith.select %195, %194, %c0_i64 : i64
        %197 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>
        %198 = cannir.extract_stride %197 : <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
        %199 = cannir.extract_dim %198[2] : !cannir.stride<(-1, -1, -1, 1)>
        %200 = arith.muli %199, %c2_i64 : i64
        %201 = cannir.extract_dim %198[0] : !cannir.stride<(-1, -1, -1, 1)>
        %202 = arith.muli %95, %201 : i64
        %203 = arith.muli %202, %c2_i64 : i64
        %204 = cannir.extract_dim %198[1] : !cannir.stride<(-1, -1, -1, 1)>
        %205 = arith.muli %91, %204 : i64
        %206 = arith.muli %205, %c2_i64 : i64
        %207 = arith.addi %203, %206 : i64
        %208 = arith.muli %117, %199 : i64
        %209 = arith.muli %208, %c2_i64 : i64
        %210 = arith.addi %207, %209 : i64
        %211 = ascscalar.bind_local_ptr %187 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %188, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%211, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %210, %196, %c2_i64, %200, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %188, 0 {cannir.buffer.generated_sync}
        %212 = ascscalar.memref_address %alloc_19 : memref<128xbf16, 1>
        %213 = ascscalar.memref_address %alloc_20 : memref<128xbf16, 1>
        %214 = arith.select %121, %212, %213 : i64
        %215 = arith.select %121, %c3_i64, %c4_i64 : i64
        %216 = arith.muli %120, %143 : i64
        %217 = arith.muli %216, %c2_i64 : i64
        %218 = arith.addi %142, %217 : i64
        %219 = arith.addi %218, %148 : i64
        %220 = arith.muli %78, %c128_i64 overflow<nsw> : i64
        %221 = arith.addi %219, %220 : i64
        %222 = ascscalar.bind_local_ptr %214 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %215, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%222, %alloc, %c0_i32, %c0_i32, %c0_i64, %221, %135, %c128_i64, %139, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %215, 0 {cannir.buffer.generated_sync}
        %223:2 = scf.for %arg16 = %96#0 to %96#1 step %c1_i64 iter_args(%arg17 = %57#0, %arg18 = %127) -> (i64, i64)  : i64 {
          %224 = arith.subi %arg16, %96#0 : i64
          %225 = arith.select %94, %arg16, %224 : i64
          %226 = arith.cmpi eq, %arg17, %c0_i64 : i64
          %227 = arith.select %226, %122, %123 : i64
          %228 = ascscalar.bind_local_ptr %227 : memref<128xbf16, 1>
          %229 = arith.select %226, %c7_i64, %c8_i64 : i64
          %230 = arith.addi %arg17, %c1_i64 overflow<nsw> : i64
          %231 = arith.remui %230, %c2_i64 : i64
          %232 = arith.select %226, %151, %152 : i64
          %233 = ascscalar.bind_local_ptr %232 : memref<128xbf16, 1>
          %234 = arith.select %226, %c9_i64, %c10_i64 : i64
          ascvec.get_buf v, %229, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %234, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%228, %233, %alloc_12, %alloc_11, %alloc_23 : memref<128xbf16, 1>, memref<128xbf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<4xf32, 1>) {
          ^bb0(%arg19: memref<128xbf16, 1>, %arg20: memref<128xbf16, 1>, %arg21: memref<128xf32, 1>, %arg22: memref<128xf32, 1>, %arg23: memref<4xf32, 1>):
            %c3 = arith.constant 3 : index
            %c2_90 = arith.constant 2 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c0_91 = arith.constant 0 : index
            %c64_i32_92 = arith.constant 64 : i32
            %c0_93 = arith.constant 0 : index
            %c1_94 = arith.constant 1 : index
            scf.for %arg24 = %c0_93 to %c1_94 step %c1_94 {
              %mask, %remaining_out = ascvec.update_mask %c64_i32_92 : !ascvec.mask
              %304 = ascvec.loadalign_unpack %arg19[%c0_91] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %305 = ascvec.loadalign_unpack %arg19[%c64] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %306 = ascvec.loadalign_unpack %arg20[%c0_91] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %307 = ascvec.loadalign_unpack %arg20[%c64] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %308 = ascvec.cast %304, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %309 = ascvec.cast %305, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %310 = ascvec.cast %306, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %311 = ascvec.cast %307, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %312 = ascvec.mul %308, %308, %mask : <f32>
              %313 = ascvec.mul %309, %309, %mask : <f32>
              %314 = ascvec.mul %310, %310, %mask : <f32>
              %315 = ascvec.mul %311, %311, %mask : <f32>
              %316 = ascvec.reduce_sum %312, %mask : <f32>
              %317 = ascvec.reduce_sum %313, %mask : <f32>
              %318 = ascvec.reduce_sum %314, %mask : <f32>
              %319 = ascvec.reduce_sum %315, %mask : <f32>
              ascvec.storealign %arg21[%c0_91], %308, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg21[%c64], %309, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg22[%c0_91], %310, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg22[%c64], %311, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign_1st %arg23[%c0_91], %316 : memref<4xf32, 1>, index, <f32>
              ascvec.storealign_1st %arg23[%c1], %317 : memref<4xf32, 1>, index, <f32>
              ascvec.storealign_1st %arg23[%c2_90], %318 : memref<4xf32, 1>, index, <f32>
              ascvec.storealign_1st %arg23[%c3], %319 : memref<4xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %234, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %229, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_23 : memref<4xf32, 1>) {
          ^bb0(%arg19: memref<4xf32, 1>):
            %c3 = arith.constant 3 : index
            %c2_90 = arith.constant 2 : index
            %c1 = arith.constant 1 : index
            %cst = arith.constant 9.99999997E-7 : f32
            %c0_91 = arith.constant 0 : index
            %c1_i32_92 = arith.constant 1 : i32
            %c0_93 = arith.constant 0 : index
            %c1_94 = arith.constant 1 : index
            scf.for %arg20 = %c0_93 to %c1_94 step %c1_94 {
              %mask, %remaining_out = ascvec.update_mask %c1_i32_92 : !ascvec.mask
              %304 = ascvec.loadalign_brc %arg19[%c0_91] : memref<4xf32, 1>, index -> <f32>
              %305 = ascvec.loadalign_brc %arg19[%c1] : memref<4xf32, 1>, index -> <f32>
              %306 = ascvec.loadalign_brc %arg19[%c2_90] : memref<4xf32, 1>, index -> <f32>
              %307 = ascvec.loadalign_brc %arg19[%c3] : memref<4xf32, 1>, index -> <f32>
              %308 = ascvec.add %304, %305, %mask : <f32>
              %309 = ascvec.add %306, %307, %mask : <f32>
              %310 = ascvec.add_scalar %308, %cst, %mask : <f32>, f32
              %311 = ascvec.add_scalar %309, %cst, %mask : <f32>, f32
              %312 = ascvec.sqrt %310, %mask : <f32>
              %313 = ascvec.sqrt %311, %mask : <f32>
              ascvec.storealign_1st %arg19[%c0_91], %312 : memref<4xf32, 1>, index, <f32>
              ascvec.storealign_1st %arg19[%c1], %313 : memref<4xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.vf(%alloc_23, %alloc_12, %alloc_11, %arg11 : memref<4xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, f32) {
          ^bb0(%arg19: memref<4xf32, 1>, %arg20: memref<128xf32, 1>, %arg21: memref<128xf32, 1>, %arg22: f32):
            %c64 = arith.constant 64 : index
            %c1 = arith.constant 1 : index
            %c0_90 = arith.constant 0 : index
            %c64_i32_91 = arith.constant 64 : i32
            %c0_92 = arith.constant 0 : index
            %c1_93 = arith.constant 1 : index
            scf.for %arg23 = %c0_92 to %c1_93 step %c1_93 {
              %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
              %304 = ascvec.loadalign_brc %arg19[%c0_90] : memref<4xf32, 1>, index -> <f32>
              %305 = ascvec.loadalign_brc %arg19[%c1] : memref<4xf32, 1>, index -> <f32>
              %306 = ascvec.duplicate %304, %mask : <f32> -> <f32>
              %307 = ascvec.duplicate %305, %mask : <f32> -> <f32>
              %308 = ascvec.loadalign %arg20[%c0_90] : memref<128xf32, 1>, index -> <f32>
              %309 = ascvec.loadalign %arg20[%c64] : memref<128xf32, 1>, index -> <f32>
              %310 = ascvec.loadalign %arg21[%c0_90] : memref<128xf32, 1>, index -> <f32>
              %311 = ascvec.loadalign %arg21[%c64] : memref<128xf32, 1>, index -> <f32>
              %312 = ascvec.div %308, %306, %mask : <f32>
              %313 = ascvec.div %309, %306, %mask : <f32>
              %314 = ascvec.div %310, %307, %mask : <f32>
              %315 = ascvec.div %311, %307, %mask : <f32>
              %316 = ascvec.mul_scalar %312, %arg22, %mask : <f32>, f32
              %317 = ascvec.mul_scalar %313, %arg22, %mask : <f32>, f32
              ascvec.storealign %arg20[%c0_90], %316, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg20[%c64], %317, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg21[%c0_90], %314, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg21[%c64], %315, %mask : memref<128xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          %235 = arith.select %226, %160, %161 : i64
          %236 = ascscalar.bind_local_ptr %235 : memref<128xbf16, 1>
          %237 = arith.select %226, %c11_i64, %c12_i64 : i64
          %238 = arith.select %226, %185, %186 : i64
          %239 = ascscalar.bind_local_ptr %238 : memref<64xbf16, 1>
          %240 = arith.select %226, %c13_i64, %c14_i64 : i64
          %241 = arith.select %226, %212, %213 : i64
          %242 = ascscalar.bind_local_ptr %241 : memref<128xbf16, 1>
          %243 = arith.select %226, %c3_i64, %c4_i64 : i64
          %244 = arith.trunci %91 : i64 to i32
          %245 = arith.index_cast %244 : i32 to index
          %246 = arith.muli %244, %c128_i32 : i32
          %247 = arith.index_cast %246 : i32 to index
          %248 = arith.addi %246, %c64_i32 : i32
          %249 = arith.index_cast %248 : i32 to index
          ascvec.get_buf v, %237, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %240, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %243, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_35, %245, %236, %alloc_34, %247, %249, %arg12, %alloc_13, %239, %alloc_15, %242, %alloc_14 : memref<96xf32, 1>, index, memref<128xbf16, 1>, memref<12288xf32, 1>, index, index, f32, memref<128xf32, 1>, memref<64xbf16, 1>, memref<64xf32, 1>, memref<128xbf16, 1>, memref<128xf32, 1>) {
          ^bb0(%arg19: memref<96xf32, 1>, %arg20: index, %arg21: memref<128xbf16, 1>, %arg22: memref<12288xf32, 1>, %arg23: index, %arg24: index, %arg25: f32, %arg26: memref<128xf32, 1>, %arg27: memref<64xbf16, 1>, %arg28: memref<64xf32, 1>, %arg29: memref<128xbf16, 1>, %arg30: memref<128xf32, 1>):
            %c64 = arith.constant 64 : index
            %c0_90 = arith.constant 0 : index
            %cst = arith.constant 1.000000e+00 : f32
            %c64_i32_91 = arith.constant 64 : i32
            %c0_92 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg31 = %c0_92 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
              %304 = ascvec.duplicate_scalar_nomask %cst : f32 -> <f32>
              %305 = ascvec.loadalign_brc %arg19[%arg20] : memref<96xf32, 1>, index -> <f32>
              %306 = ascvec.exp %305, %mask : <f32>
              %307 = ascvec.neg %306, %mask : <f32>
              %308 = ascvec.loadalign_unpack %arg21[%c0_90] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %309 = ascvec.loadalign_unpack %arg21[%c64] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %310 = ascvec.cast %308, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %311 = ascvec.cast %309, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %312 = ascvec.loadalign %arg22[%arg23] : memref<12288xf32, 1>, index -> <f32>
              %313 = ascvec.loadalign %arg22[%arg24] : memref<12288xf32, 1>, index -> <f32>
              %314 = ascvec.add %310, %312, %mask : <f32>
              %315 = ascvec.add %311, %313, %mask : <f32>
              %316 = ascvec.mul %307, %314, %mask : <f32>
              %317 = ascvec.mul %307, %315, %mask : <f32>
              %318 = ascvec.exp %316, %mask : <f32>
              %319 = ascvec.exp %317, %mask : <f32>
              %320 = ascvec.add_scalar %318, %cst, %mask : <f32>, f32
              %321 = ascvec.add_scalar %319, %cst, %mask : <f32>, f32
              %322 = ascvec.div %304, %320, %mask : <f32>
              %323 = ascvec.div %304, %321, %mask : <f32>
              %324 = ascvec.mul_scalar %322, %arg25, %mask : <f32>, f32
              %325 = ascvec.mul_scalar %323, %arg25, %mask : <f32>, f32
              %326 = ascvec.exp %324, %mask : <f32>
              %327 = ascvec.exp %325, %mask : <f32>
              ascvec.storealign %arg26[%c0_90], %326, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg26[%c64], %327, %mask : memref<128xf32, 1>, index, <f32>
              %328 = ascvec.loadalign_unpack %arg27[%c0_90] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
              %329 = ascvec.cast %328, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %330 = ascvec.neg %329, %mask : <f32>
              %331 = ascvec.exp %330, %mask : <f32>
              %332 = ascvec.add_scalar %331, %cst, %mask : <f32>, f32
              %333 = ascvec.div %304, %332, %mask : <f32>
              ascvec.storealign %arg28[%c0_90], %333, %mask : memref<64xf32, 1>, index, <f32>
              %334 = ascvec.loadalign_unpack %arg29[%c0_90] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %335 = ascvec.cast %334, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              ascvec.storealign %arg30[%c0_90], %335, %mask : memref<128xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %243, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %240, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %237, 0 {cannir.buffer.generated_sync}
          %250 = arith.addi %arg16, %c1_i64 : i64
          %251 = arith.cmpi slt, %250, %96#1 : i64
          %252 = scf.if %251 -> (i64) {
            %304 = arith.subi %250, %96#0 : i64
            %305 = arith.select %94, %250, %304 : i64
            %306 = arith.cmpi eq, %arg18, %c0_i64 : i64
            %307 = arith.select %306, %122, %123 : i64
            %308 = arith.select %306, %c7_i64, %c8_i64 : i64
            %309 = arith.addi %arg18, %c1_i64 overflow<nsw> : i64
            %310 = arith.remui %309, %c2_i64 : i64
            %311 = arith.minsi %305, %130 : i64
            %312 = arith.subi %130, %311 : i64
            %313 = arith.minsi %312, %c1_i64 : i64
            %314 = arith.cmpi sle, %305, %130 : i64
            %315 = arith.select %314, %313, %c0_i64 : i64
            %316 = arith.muli %305, %138 : i64
            %317 = arith.muli %316, %c2_i64 : i64
            %318 = arith.addi %146, %317 : i64
            %319 = ascscalar.bind_local_ptr %307 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %308, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%319, %alloc, %c0_i32, %c0_i32, %c0_i64, %318, %315, %c256_i64, %139, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %308, 0 {cannir.buffer.generated_sync}
            %320 = arith.select %306, %151, %152 : i64
            %321 = arith.select %306, %c9_i64, %c10_i64 : i64
            %322 = arith.addi %157, %317 : i64
            %323 = ascscalar.bind_local_ptr %320 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %321, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%323, %alloc, %c0_i32, %c0_i32, %c0_i64, %322, %315, %c256_i64, %139, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %321, 0 {cannir.buffer.generated_sync}
            %324 = arith.select %306, %160, %161 : i64
            %325 = arith.select %306, %c11_i64, %c12_i64 : i64
            %326 = arith.minsi %305, %164 : i64
            %327 = arith.subi %164, %326 : i64
            %328 = arith.minsi %327, %c1_i64 : i64
            %329 = arith.cmpi sle, %305, %164 : i64
            %330 = arith.select %329, %328, %c0_i64 : i64
            %331 = arith.muli %305, %172 : i64
            %332 = arith.muli %331, %c2_i64 : i64
            %333 = arith.addi %180, %332 : i64
            %334 = ascscalar.bind_local_ptr %324 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %325, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%334, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %333, %330, %c256_i64, %173, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %325, 0 {cannir.buffer.generated_sync}
            %335 = arith.select %306, %185, %186 : i64
            %336 = arith.select %306, %c13_i64, %c14_i64 : i64
            %337 = arith.minsi %305, %191 : i64
            %338 = arith.subi %191, %337 : i64
            %339 = arith.minsi %338, %c1_i64 : i64
            %340 = arith.cmpi sle, %305, %191 : i64
            %341 = arith.select %340, %339, %c0_i64 : i64
            %342 = arith.muli %305, %199 : i64
            %343 = arith.muli %342, %c2_i64 : i64
            %344 = arith.addi %207, %343 : i64
            %345 = ascscalar.bind_local_ptr %335 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %336, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%345, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %344, %341, %c2_i64, %200, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %336, 0 {cannir.buffer.generated_sync}
            %346 = arith.select %306, %212, %213 : i64
            %347 = arith.select %306, %c3_i64, %c4_i64 : i64
            %348 = arith.addi %218, %317 : i64
            %349 = arith.addi %348, %220 : i64
            %350 = ascscalar.bind_local_ptr %346 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %347, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%350, %alloc, %c0_i32, %c0_i32, %c0_i64, %349, %315, %c128_i64, %139, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %347, 0 {cannir.buffer.generated_sync}
            scf.yield %310 : i64
          } else {
            scf.yield %arg18 : i64
          }
          %253 = ascscalar.memref_address %alloc_17 : memref<16384xf32, 1>
          %254 = ascscalar.memref_address %alloc_18 : memref<16384xf32, 1>
          %255 = arith.select %226, %253, %254 : i64
          %256 = ascscalar.bind_local_ptr %255 : memref<16384xf32, 1>
          %257 = arith.select %226, %c1_i64, %c2_i64 : i64
          ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_13, %alloc_11, %alloc_10, %alloc_24 : memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>, memref<128xf32, 1>) {
          ^bb0(%arg19: memref<128xf32, 1>, %arg20: memref<128xf32, 1>, %arg21: memref<16384xf32, 1>, %arg22: memref<128xf32, 1>):
            %c64 = arith.constant 64 : index
            %c1_i32_90 = arith.constant 1 : i32
            %c0_i32_91 = arith.constant 0 : i32
            %c128_i32_92 = arith.constant 128 : i32
            %c0_93 = arith.constant 0 : index
            %c64_i32_94 = arith.constant 64 : i32
            %mask, %remaining_out = ascvec.update_mask %c64_i32_94 : !ascvec.mask
            %304 = ascvec.loadalign %arg19[%c0_93] : memref<128xf32, 1>, index -> <f32>
            %305 = ascvec.loadalign %arg19[%c64] : memref<128xf32, 1>, index -> <f32>
            %306 = ascvec.loadalign %arg20[%c0_93] : memref<128xf32, 1>, index -> <f32>
            %307 = ascvec.loadalign %arg20[%c64] : memref<128xf32, 1>, index -> <f32>
            %308 = ascvec.mul %304, %306, %mask : <f32>
            %309 = ascvec.mul %305, %307, %mask : <f32>
            scf.for %arg23 = %c0_i32_91 to %c64_i32_94 step %c1_i32_90  : i32 {
              %310 = arith.muli %arg23, %c128_i32_92 overflow<nsw> : i32
              %311 = arith.addi %310, %c64_i32_94 overflow<nsw> : i32
              %312 = arith.index_cast %310 : i32 to index
              %313 = ascvec.loadalign %arg21[%312] : memref<16384xf32, 1>, index -> <f32>
              %314 = arith.index_cast %311 : i32 to index
              %315 = ascvec.loadalign %arg21[%314] : memref<16384xf32, 1>, index -> <f32>
              %316 = ascvec.mul %313, %308, %mask : <f32>
              %317 = ascvec.madd_two_address %315, %309, %316, %mask : <f32>
              %318 = ascvec.reduce_sum %317, %mask : <f32>
              %319 = arith.index_cast %arg23 : i32 to index
              ascvec.storealign_1st %arg22[%319], %318 : memref<128xf32, 1>, index, <f32>
            } {cannir.loop_unroll = 4 : i64}
          } {cannir.vf_mode = "simd"}
          ascvec.vf(%alloc_15, %alloc_24, %alloc_14, %alloc_25 : memref<64xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>) {
          ^bb0(%arg19: memref<64xf32, 1>, %arg20: memref<128xf32, 1>, %arg21: memref<128xf32, 1>, %arg22: memref<128xf32, 1>):
            %c0_90 = arith.constant 0 : index
            %c64_i32_91 = arith.constant 64 : i32
            %c0_92 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg23 = %c0_92 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
              %304 = ascvec.loadalign_brc %arg19[%c0_90] : memref<64xf32, 1>, index -> <f32>
              %305 = ascvec.loadalign %arg20[%c0_90] : memref<128xf32, 1>, index -> <f32>
              %306 = ascvec.loadalign %arg21[%c0_90] : memref<128xf32, 1>, index -> <f32>
              %307 = ascvec.sub %306, %305, %mask : <f32>
              %308 = ascvec.mul %307, %304, %mask : <f32>
              ascvec.storealign %arg22[%c0_90], %308, %mask : memref<128xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.get_buf v, %257, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_13, %alloc_11, %alloc_12, %alloc_10, %alloc_25, %256, %alloc_16 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>, memref<128xf32, 1>) {
          ^bb0(%arg19: memref<128xf32, 1>, %arg20: memref<128xf32, 1>, %arg21: memref<128xf32, 1>, %arg22: memref<16384xf32, 1>, %arg23: memref<128xf32, 1>, %arg24: memref<16384xf32, 1>, %arg25: memref<128xf32, 1>):
            %c64 = arith.constant 64 : index
            %c1_i32_90 = arith.constant 1 : i32
            %c0_i32_91 = arith.constant 0 : i32
            %c128_i32_92 = arith.constant 128 : i32
            %c0_93 = arith.constant 0 : index
            %c64_i32_94 = arith.constant 64 : i32
            %mask, %remaining_out = ascvec.update_mask %c64_i32_94 : !ascvec.mask
            %304 = ascvec.loadalign %arg19[%c0_93] : memref<128xf32, 1>, index -> <f32>
            %305 = ascvec.loadalign %arg19[%c64] : memref<128xf32, 1>, index -> <f32>
            %306 = ascvec.loadalign %arg20[%c0_93] : memref<128xf32, 1>, index -> <f32>
            %307 = ascvec.loadalign %arg20[%c64] : memref<128xf32, 1>, index -> <f32>
            %308 = ascvec.loadalign %arg21[%c0_93] : memref<128xf32, 1>, index -> <f32>
            %309 = ascvec.loadalign %arg21[%c64] : memref<128xf32, 1>, index -> <f32>
            scf.for %arg26 = %c0_i32_91 to %c64_i32_94 step %c1_i32_90  : i32 {
              %310 = arith.muli %arg26, %c128_i32_92 overflow<nsw> : i32
              %311 = arith.addi %310, %c64_i32_94 overflow<nsw> : i32
              %312 = arith.index_cast %310 : i32 to index
              %313 = ascvec.loadalign %arg22[%312] : memref<16384xf32, 1>, index -> <f32>
              %314 = arith.index_cast %311 : i32 to index
              %315 = ascvec.loadalign %arg22[%314] : memref<16384xf32, 1>, index -> <f32>
              %316 = arith.index_cast %arg26 : i32 to index
              %317 = ascvec.loadalign_brc %arg23[%316] : memref<128xf32, 1>, index -> <f32>
              %318 = ascvec.mul %317, %306, %mask : <f32>
              %319 = ascvec.mul %317, %307, %mask : <f32>
              %320 = ascvec.madd_two_address %313, %304, %318, %mask : <f32>
              %321 = ascvec.madd_two_address %315, %305, %319, %mask : <f32>
              ascvec.storealign %arg22[%312], %320, %mask : memref<16384xf32, 1>, index, <f32>
              ascvec.storealign %arg22[%314], %321, %mask : memref<16384xf32, 1>, index, <f32>
              ascvec.storealign %arg24[%312], %320, %mask : memref<16384xf32, 1>, index, <f32>
              ascvec.storealign %arg24[%314], %321, %mask : memref<16384xf32, 1>, index, <f32>
              %322 = ascvec.mul %320, %308, %mask : <f32>
              %323 = ascvec.madd_two_address %321, %309, %322, %mask : <f32>
              %324 = ascvec.reduce_sum %323, %mask : <f32>
              ascvec.storealign_1st %arg25[%316], %324 : memref<128xf32, 1>, index, <f32>
            } {cannir.loop_unroll = 1 : i64}
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %257, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          %258 = ascscalar.memref_address %alloc_21 : memref<128xbf16, 1>
          %259 = ascscalar.memref_address %alloc_22 : memref<128xbf16, 1>
          %260 = arith.select %226, %258, %259 : i64
          %261 = ascscalar.bind_local_ptr %260 : memref<128xbf16, 1>
          %262 = arith.select %226, %c5_i64, %c6_i64 : i64
          ascvec.get_buf v, %262, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_16, %261 : memref<128xf32, 1>, memref<128xbf16, 1>) {
          ^bb0(%arg19: memref<128xf32, 1>, %arg20: memref<128xbf16, 1>):
            %c0_90 = arith.constant 0 : index
            %c64_i32_91 = arith.constant 64 : i32
            %c0_92 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg21 = %c0_92 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
              %304 = ascvec.loadalign %arg19[%c0_90] : memref<128xf32, 1>, index -> <f32>
              %305 = ascvec.cast %304, %mask : <f32> -> <bf16>
              ascvec.storealign_pack %arg20[%c0_90], %305, %mask {mode = "b32_to_b16"} : memref<128xbf16, 1>, index, <bf16>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %262, 0 {cannir.buffer.generated_sync}
          %263 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
          %264 = cannir.extract_stride %263 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
          %265 = cannir.extract_dim %264[0] : !cannir.stride<(-1, -1, -1, 1)>
          %266 = arith.muli %95, %265 : i64
          %267 = cannir.extract_dim %264[1] : !cannir.stride<(-1, -1, -1, 1)>
          %268 = arith.muli %91, %267 : i64
          %269 = arith.addi %266, %268 : i64
          %270 = cannir.extract_dim %264[2] : !cannir.stride<(-1, -1, -1, 1)>
          %271 = arith.muli %225, %270 : i64
          %272 = arith.addi %269, %271 : i64
          %273 = arith.muli %78, %c64_i64 overflow<nsw> : i64
          %274 = arith.addi %272, %273 : i64
          %275 = arith.muli %274, %c16_i64 : i64
          %276 = ascscalar.memref_address %alloc_4 : memref<1xbf16>
          %277 = arith.divui %275, %c8_i64 : i64
          %278 = arith.addi %276, %277 : i64
          %279 = ascscalar.bind_gm_ptr %278 : memref<1xbf16>
          %280 = ascscalar.bind_local_ptr %260 : memref<1xbf16, 1>
          %281 = cannir.extract_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
          %282 = cannir.extract_stride %281 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
          %283 = cannir.extract_dim %282[2] : !cannir.stride<(-1, -1, -1, 1)>
          %284 = arith.index_cast %283 : i64 to index
          %285 = arith.muli %284, %c2 : index
          %286 = arith.index_cast %285 : index to i64
          ascvec.get_buf mte3, %262, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %279[%c0], %280[%c0], %c1_i32, %c128_i32, %286, %c128_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
          ascvec.release_buf mte3, %262, 0 {cannir.buffer.generated_sync}
          %287 = arith.index_cast %arg16 : i64 to index
          %288 = memref.load %alloc_7[%287] : memref<1xi32>
          %289 = arith.extsi %288 : i32 to i64
          %290 = cannir.extract_physical_layout %arg7 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
          %291 = cannir.extract_stride %290 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
          %292 = cannir.extract_dim %291[0] : !cannir.stride<(-1, 16384, 128, 1)>
          %293 = arith.muli %289, %292 : i64
          %294 = arith.muli %91, %c16384_i64 : i64
          %295 = arith.addi %293, %294 : i64
          %296 = arith.muli %78, %c8192_i64 overflow<nsw> : i64
          %297 = arith.addi %295, %296 : i64
          %298 = arith.muli %297, %c32_i64 : i64
          %299 = ascscalar.memref_address %alloc_6 : memref<1xf32>
          %300 = arith.divui %298, %c8_i64 : i64
          %301 = arith.addi %299, %300 : i64
          %302 = ascscalar.bind_gm_ptr %301 : memref<1xf32>
          %303 = ascscalar.bind_local_ptr %255 : memref<1xf32, 1>
          ascvec.get_buf mte3, %257, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %302[%c0], %303[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
          ascvec.release_buf mte3, %257, 0 {cannir.buffer.generated_sync}
          scf.yield %231, %252 : i64, i64
        }
        scf.yield %223#0, %223#1, %223#1, %223#1, %223#1, %223#1 : i64, i64, i64, i64, i64, i64
      } else {
        scf.yield %57#0, %57#1, %57#1, %57#1, %57#1, %57#1 : i64, i64, i64, i64, i64, i64
      }
      %66 = arith.cmpi eq, %62, %c1_i64 : i64
      %67 = arith.cmpi sge, %21, %34 : i64
      %68 = arith.select %66, %67, %false : i1
      scf.if %68 {
        %69 = arith.subi %21, %34 : i64
        %70 = arith.remsi %69, %c2_i64 : i64
        %71 = arith.cmpi ne, %70, %c0_i64 : i64
        %72 = arith.cmpi slt, %70, %c0_i64 : i64
        %73 = arith.andi %71, %72 : i1
        %74 = arith.divsi %69, %c2_i64 : i64
        %75 = arith.extui %73 : i1 to i64
        %76 = arith.subi %74, %75 : i64
        %77 = arith.addi %60, %76 : i64
        %78 = arith.select %73, %c2_i64, %c0_i64 : i64
        %79 = arith.addi %70, %78 overflow<nsw> : i64
        %80 = arith.maxsi %2, %c1_i64 : i64
        %81 = arith.remsi %77, %80 : i64
        %82 = arith.cmpi slt, %81, %c0_i64 : i64
        %83 = arith.divsi %77, %80 : i64
        %84 = arith.extui %82 : i1 to i64
        %85 = arith.subi %83, %84 : i64
        %86 = arith.maxsi %arg14, %c1_i64 : i64
        %87 = arith.remsi %85, %86 : i64
        %88 = arith.cmpi slt, %87, %c0_i64 : i64
        %89 = arith.select %88, %86, %c0_i64 : i64
        %90 = arith.addi %87, %89 : i64
        %91 = arith.select %82, %80, %c0_i64 : i64
        %92 = arith.addi %81, %91 : i64
        %93 = arith.muli %90, %arg15 : i64
        %94 = arith.addi %93, %arg15 : i64
        %95 = arith.cmpi eq, %arg13, %c1_i64 : i64
        %96 = arith.select %95, %c0_i64, %90 : i64
        %97:2 = scf.if %95 -> (i64, i64) {
          %229 = arith.index_cast %90 : i64 to index
          %230 = memref.load %alloc_9[%229] : memref<1xi32>
          %231 = arith.addi %90, %c1_i64 : i64
          %232 = arith.index_cast %231 : i64 to index
          %233 = memref.load %alloc_9[%232] : memref<1xi32>
          %234 = arith.extsi %230 : i32 to i64
          %235 = arith.extsi %233 : i32 to i64
          scf.yield %234, %235 : i64, i64
        } else {
          scf.yield %93, %94 : i64, i64
        }
        %98 = arith.index_cast %90 : i64 to index
        %99 = memref.load %alloc_8[%98] : memref<1xi32>
        %100 = arith.cmpi sgt, %99, %c0_i32 : i32
        %101 = arith.extsi %99 : i32 to i64
        %102 = arith.addi %97#0, %101 : i64
        %103 = arith.subi %102, %c1_i64 : i64
        %104 = arith.select %100, %103, %97#0 : i64
        %105 = arith.index_cast %104 : i64 to index
        %106 = memref.load %alloc_7[%105] : memref<1xi32>
        %107 = arith.extsi %106 : i32 to i64
        %108 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<16384xf32, 1> -> memref<?xf32, 1>
        %109 = cannir.extract_physical_layout %arg6 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
        %110 = cannir.extract_stride %109 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
        %111 = cannir.extract_dim %110[0] : !cannir.stride<(-1, 16384, 128, 1)>
        %112 = arith.muli %107, %111 : i64
        %113 = arith.muli %112, %c4_i64 : i64
        %114 = arith.muli %92, %c65536_i64 : i64
        %115 = arith.addi %113, %114 : i64
        %116 = arith.muli %79, %c32768_i64 overflow<nsw> : i64
        %117 = arith.addi %115, %116 : i64
        ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%108, %alloc_5, %c0_i32, %c0_i32, %c0_i64, %117, %c1_i64, %c32768_i64, %c0_i64, %c32768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 65536 : i64}> : (memref<?xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
        %118 = arith.select %95, %97#0, %c0_i64 : i64
        %119 = arith.addi %2, %92 : i64
        %120 = arith.muli %2, %c2_i64 : i64
        %121 = arith.addi %120, %92 : i64
        %122 = arith.cmpi eq, %65#2, %c0_i64 : i64
        %123 = ascscalar.memref_address %alloc_26 : memref<128xbf16, 1>
        %124 = ascscalar.memref_address %alloc_27 : memref<128xbf16, 1>
        %125 = arith.select %122, %123, %124 : i64
        %126 = arith.select %122, %c7_i64, %c8_i64 : i64
        %127 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
        %128 = cannir.extract_shape %127 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 128)>
        %129 = cannir.extract_dim %128[2] : !cannir.shape<(-1, -1, -1, 128)>
        %130 = arith.minsi %118, %129 : i64
        %131 = arith.subi %129, %130 : i64
        %132 = arith.minsi %131, %c1_i64 : i64
        %133 = arith.cmpi sle, %118, %129 : i64
        %134 = arith.select %133, %132, %c0_i64 : i64
        %135 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
        %136 = cannir.extract_stride %135 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
        %137 = cannir.extract_dim %136[2] : !cannir.stride<(-1, -1, -1, 1)>
        %138 = arith.muli %137, %c2_i64 : i64
        %139 = cannir.extract_dim %136[0] : !cannir.stride<(-1, -1, -1, 1)>
        %140 = arith.muli %96, %139 : i64
        %141 = arith.muli %140, %c2_i64 : i64
        %142 = cannir.extract_dim %136[1] : !cannir.stride<(-1, -1, -1, 1)>
        %143 = arith.muli %92, %142 : i64
        %144 = arith.muli %143, %c2_i64 : i64
        %145 = arith.addi %141, %144 : i64
        %146 = arith.muli %118, %137 : i64
        %147 = arith.muli %146, %c2_i64 : i64
        %148 = arith.addi %145, %147 : i64
        %149 = ascscalar.bind_local_ptr %125 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %126, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%149, %alloc, %c0_i32, %c0_i32, %c0_i64, %148, %134, %c256_i64, %138, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %126, 0 {cannir.buffer.generated_sync}
        %150 = arith.cmpi eq, %65#3, %c0_i64 : i64
        %151 = ascscalar.memref_address %alloc_28 : memref<128xbf16, 1>
        %152 = ascscalar.memref_address %alloc_29 : memref<128xbf16, 1>
        %153 = arith.select %150, %151, %152 : i64
        %154 = arith.select %150, %c9_i64, %c10_i64 : i64
        %155 = arith.muli %119, %142 : i64
        %156 = arith.muli %155, %c2_i64 : i64
        %157 = arith.addi %141, %156 : i64
        %158 = arith.addi %157, %147 : i64
        %159 = ascscalar.bind_local_ptr %153 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %154, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%159, %alloc, %c0_i32, %c0_i32, %c0_i64, %158, %134, %c256_i64, %138, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %154, 0 {cannir.buffer.generated_sync}
        %160 = arith.cmpi eq, %65#4, %c0_i64 : i64
        %161 = ascscalar.memref_address %alloc_30 : memref<128xbf16, 1>
        %162 = ascscalar.memref_address %alloc_31 : memref<128xbf16, 1>
        %163 = arith.select %160, %161, %162 : i64
        %164 = arith.select %160, %c11_i64, %c12_i64 : i64
        %165 = cannir.extract_dim %1[2] : !cannir.shape<(-1, -1, -1, 128)>
        %166 = arith.minsi %118, %165 : i64
        %167 = arith.subi %165, %166 : i64
        %168 = arith.minsi %167, %c1_i64 : i64
        %169 = arith.cmpi sle, %118, %165 : i64
        %170 = arith.select %169, %168, %c0_i64 : i64
        %171 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
        %172 = cannir.extract_stride %171 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
        %173 = cannir.extract_dim %172[2] : !cannir.stride<(-1, -1, -1, 1)>
        %174 = arith.muli %173, %c2_i64 : i64
        %175 = cannir.extract_dim %172[0] : !cannir.stride<(-1, -1, -1, 1)>
        %176 = arith.muli %96, %175 : i64
        %177 = arith.muli %176, %c2_i64 : i64
        %178 = cannir.extract_dim %172[1] : !cannir.stride<(-1, -1, -1, 1)>
        %179 = arith.muli %92, %178 : i64
        %180 = arith.muli %179, %c2_i64 : i64
        %181 = arith.addi %177, %180 : i64
        %182 = arith.muli %118, %173 : i64
        %183 = arith.muli %182, %c2_i64 : i64
        %184 = arith.addi %181, %183 : i64
        %185 = ascscalar.bind_local_ptr %163 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %164, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%185, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %184, %170, %c256_i64, %174, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %164, 0 {cannir.buffer.generated_sync}
        %186 = arith.cmpi eq, %65#5, %c0_i64 : i64
        %187 = ascscalar.memref_address %alloc_32 : memref<64xbf16, 1>
        %188 = ascscalar.memref_address %alloc_33 : memref<64xbf16, 1>
        %189 = arith.select %186, %187, %188 : i64
        %190 = arith.select %186, %c13_i64, %c14_i64 : i64
        %191 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>
        %192 = cannir.extract_shape %191 : <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
        %193 = cannir.extract_dim %192[2] : !cannir.shape<(-1, -1, -1, 1)>
        %194 = arith.minsi %118, %193 : i64
        %195 = arith.subi %193, %194 : i64
        %196 = arith.minsi %195, %c1_i64 : i64
        %197 = arith.cmpi sle, %118, %193 : i64
        %198 = arith.select %197, %196, %c0_i64 : i64
        %199 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>
        %200 = cannir.extract_stride %199 : <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
        %201 = cannir.extract_dim %200[2] : !cannir.stride<(-1, -1, -1, 1)>
        %202 = arith.muli %201, %c2_i64 : i64
        %203 = cannir.extract_dim %200[0] : !cannir.stride<(-1, -1, -1, 1)>
        %204 = arith.muli %96, %203 : i64
        %205 = arith.muli %204, %c2_i64 : i64
        %206 = cannir.extract_dim %200[1] : !cannir.stride<(-1, -1, -1, 1)>
        %207 = arith.muli %92, %206 : i64
        %208 = arith.muli %207, %c2_i64 : i64
        %209 = arith.addi %205, %208 : i64
        %210 = arith.muli %118, %201 : i64
        %211 = arith.muli %210, %c2_i64 : i64
        %212 = arith.addi %209, %211 : i64
        %213 = ascscalar.bind_local_ptr %189 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %190, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%213, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %212, %198, %c2_i64, %202, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %190, 0 {cannir.buffer.generated_sync}
        %214 = arith.cmpi eq, %65#1, %c0_i64 : i64
        %215 = ascscalar.memref_address %alloc_19 : memref<128xbf16, 1>
        %216 = ascscalar.memref_address %alloc_20 : memref<128xbf16, 1>
        %217 = arith.select %214, %215, %216 : i64
        %218 = arith.select %214, %c3_i64, %c4_i64 : i64
        %219 = arith.addi %65#1, %c1_i64 overflow<nsw> : i64
        %220 = arith.remui %219, %c2_i64 : i64
        %221 = arith.muli %121, %142 : i64
        %222 = arith.muli %221, %c2_i64 : i64
        %223 = arith.addi %141, %222 : i64
        %224 = arith.addi %223, %147 : i64
        %225 = arith.muli %79, %c128_i64 overflow<nsw> : i64
        %226 = arith.addi %224, %225 : i64
        %227 = ascscalar.bind_local_ptr %217 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %218, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%227, %alloc, %c0_i32, %c0_i32, %c0_i64, %226, %134, %c128_i64, %138, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %218, 0 {cannir.buffer.generated_sync}
        %228:2 = scf.for %arg16 = %97#0 to %97#1 step %c1_i64 iter_args(%arg17 = %65#0, %arg18 = %220) -> (i64, i64)  : i64 {
          %229 = arith.subi %arg16, %97#0 : i64
          %230 = arith.select %95, %arg16, %229 : i64
          %231 = arith.cmpi eq, %arg17, %c0_i64 : i64
          %232 = arith.select %231, %123, %124 : i64
          %233 = ascscalar.bind_local_ptr %232 : memref<128xbf16, 1>
          %234 = arith.select %231, %c7_i64, %c8_i64 : i64
          %235 = arith.addi %arg17, %c1_i64 overflow<nsw> : i64
          %236 = arith.remui %235, %c2_i64 : i64
          %237 = arith.select %231, %151, %152 : i64
          %238 = ascscalar.bind_local_ptr %237 : memref<128xbf16, 1>
          %239 = arith.select %231, %c9_i64, %c10_i64 : i64
          ascvec.get_buf v, %234, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %239, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%233, %238, %alloc_12, %alloc_11, %alloc_23 : memref<128xbf16, 1>, memref<128xbf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<4xf32, 1>) {
          ^bb0(%arg19: memref<128xbf16, 1>, %arg20: memref<128xbf16, 1>, %arg21: memref<128xf32, 1>, %arg22: memref<128xf32, 1>, %arg23: memref<4xf32, 1>):
            %c3 = arith.constant 3 : index
            %c2_90 = arith.constant 2 : index
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c0_91 = arith.constant 0 : index
            %c64_i32_92 = arith.constant 64 : i32
            %c0_93 = arith.constant 0 : index
            %c1_94 = arith.constant 1 : index
            scf.for %arg24 = %c0_93 to %c1_94 step %c1_94 {
              %mask, %remaining_out = ascvec.update_mask %c64_i32_92 : !ascvec.mask
              %309 = ascvec.loadalign_unpack %arg19[%c0_91] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %310 = ascvec.loadalign_unpack %arg19[%c64] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %311 = ascvec.loadalign_unpack %arg20[%c0_91] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %312 = ascvec.loadalign_unpack %arg20[%c64] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %313 = ascvec.cast %309, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %314 = ascvec.cast %310, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %315 = ascvec.cast %311, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %316 = ascvec.cast %312, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %317 = ascvec.mul %313, %313, %mask : <f32>
              %318 = ascvec.mul %314, %314, %mask : <f32>
              %319 = ascvec.mul %315, %315, %mask : <f32>
              %320 = ascvec.mul %316, %316, %mask : <f32>
              %321 = ascvec.reduce_sum %317, %mask : <f32>
              %322 = ascvec.reduce_sum %318, %mask : <f32>
              %323 = ascvec.reduce_sum %319, %mask : <f32>
              %324 = ascvec.reduce_sum %320, %mask : <f32>
              ascvec.storealign %arg21[%c0_91], %313, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg21[%c64], %314, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg22[%c0_91], %315, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg22[%c64], %316, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign_1st %arg23[%c0_91], %321 : memref<4xf32, 1>, index, <f32>
              ascvec.storealign_1st %arg23[%c1], %322 : memref<4xf32, 1>, index, <f32>
              ascvec.storealign_1st %arg23[%c2_90], %323 : memref<4xf32, 1>, index, <f32>
              ascvec.storealign_1st %arg23[%c3], %324 : memref<4xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %239, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %234, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_23 : memref<4xf32, 1>) {
          ^bb0(%arg19: memref<4xf32, 1>):
            %c3 = arith.constant 3 : index
            %c2_90 = arith.constant 2 : index
            %c1 = arith.constant 1 : index
            %cst = arith.constant 9.99999997E-7 : f32
            %c0_91 = arith.constant 0 : index
            %c1_i32_92 = arith.constant 1 : i32
            %c0_93 = arith.constant 0 : index
            %c1_94 = arith.constant 1 : index
            scf.for %arg20 = %c0_93 to %c1_94 step %c1_94 {
              %mask, %remaining_out = ascvec.update_mask %c1_i32_92 : !ascvec.mask
              %309 = ascvec.loadalign_brc %arg19[%c0_91] : memref<4xf32, 1>, index -> <f32>
              %310 = ascvec.loadalign_brc %arg19[%c1] : memref<4xf32, 1>, index -> <f32>
              %311 = ascvec.loadalign_brc %arg19[%c2_90] : memref<4xf32, 1>, index -> <f32>
              %312 = ascvec.loadalign_brc %arg19[%c3] : memref<4xf32, 1>, index -> <f32>
              %313 = ascvec.add %309, %310, %mask : <f32>
              %314 = ascvec.add %311, %312, %mask : <f32>
              %315 = ascvec.add_scalar %313, %cst, %mask : <f32>, f32
              %316 = ascvec.add_scalar %314, %cst, %mask : <f32>, f32
              %317 = ascvec.sqrt %315, %mask : <f32>
              %318 = ascvec.sqrt %316, %mask : <f32>
              ascvec.storealign_1st %arg19[%c0_91], %317 : memref<4xf32, 1>, index, <f32>
              ascvec.storealign_1st %arg19[%c1], %318 : memref<4xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.vf(%alloc_23, %alloc_12, %alloc_11, %arg11 : memref<4xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, f32) {
          ^bb0(%arg19: memref<4xf32, 1>, %arg20: memref<128xf32, 1>, %arg21: memref<128xf32, 1>, %arg22: f32):
            %c64 = arith.constant 64 : index
            %c1 = arith.constant 1 : index
            %c0_90 = arith.constant 0 : index
            %c64_i32_91 = arith.constant 64 : i32
            %c0_92 = arith.constant 0 : index
            %c1_93 = arith.constant 1 : index
            scf.for %arg23 = %c0_92 to %c1_93 step %c1_93 {
              %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
              %309 = ascvec.loadalign_brc %arg19[%c0_90] : memref<4xf32, 1>, index -> <f32>
              %310 = ascvec.loadalign_brc %arg19[%c1] : memref<4xf32, 1>, index -> <f32>
              %311 = ascvec.duplicate %309, %mask : <f32> -> <f32>
              %312 = ascvec.duplicate %310, %mask : <f32> -> <f32>
              %313 = ascvec.loadalign %arg20[%c0_90] : memref<128xf32, 1>, index -> <f32>
              %314 = ascvec.loadalign %arg20[%c64] : memref<128xf32, 1>, index -> <f32>
              %315 = ascvec.loadalign %arg21[%c0_90] : memref<128xf32, 1>, index -> <f32>
              %316 = ascvec.loadalign %arg21[%c64] : memref<128xf32, 1>, index -> <f32>
              %317 = ascvec.div %313, %311, %mask : <f32>
              %318 = ascvec.div %314, %311, %mask : <f32>
              %319 = ascvec.div %315, %312, %mask : <f32>
              %320 = ascvec.div %316, %312, %mask : <f32>
              %321 = ascvec.mul_scalar %317, %arg22, %mask : <f32>, f32
              %322 = ascvec.mul_scalar %318, %arg22, %mask : <f32>, f32
              ascvec.storealign %arg20[%c0_90], %321, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg20[%c64], %322, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg21[%c0_90], %319, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg21[%c64], %320, %mask : memref<128xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          %240 = arith.select %231, %161, %162 : i64
          %241 = ascscalar.bind_local_ptr %240 : memref<128xbf16, 1>
          %242 = arith.select %231, %c11_i64, %c12_i64 : i64
          %243 = arith.select %231, %187, %188 : i64
          %244 = ascscalar.bind_local_ptr %243 : memref<64xbf16, 1>
          %245 = arith.select %231, %c13_i64, %c14_i64 : i64
          %246 = arith.select %231, %215, %216 : i64
          %247 = ascscalar.bind_local_ptr %246 : memref<128xbf16, 1>
          %248 = arith.select %231, %c3_i64, %c4_i64 : i64
          %249 = arith.trunci %92 : i64 to i32
          %250 = arith.index_cast %249 : i32 to index
          %251 = arith.muli %249, %c128_i32 : i32
          %252 = arith.index_cast %251 : i32 to index
          %253 = arith.addi %251, %c64_i32 : i32
          %254 = arith.index_cast %253 : i32 to index
          ascvec.get_buf v, %242, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %245, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %248, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_35, %250, %241, %alloc_34, %252, %254, %arg12, %alloc_13, %244, %alloc_15, %247, %alloc_14 : memref<96xf32, 1>, index, memref<128xbf16, 1>, memref<12288xf32, 1>, index, index, f32, memref<128xf32, 1>, memref<64xbf16, 1>, memref<64xf32, 1>, memref<128xbf16, 1>, memref<128xf32, 1>) {
          ^bb0(%arg19: memref<96xf32, 1>, %arg20: index, %arg21: memref<128xbf16, 1>, %arg22: memref<12288xf32, 1>, %arg23: index, %arg24: index, %arg25: f32, %arg26: memref<128xf32, 1>, %arg27: memref<64xbf16, 1>, %arg28: memref<64xf32, 1>, %arg29: memref<128xbf16, 1>, %arg30: memref<128xf32, 1>):
            %c64 = arith.constant 64 : index
            %c0_90 = arith.constant 0 : index
            %cst = arith.constant 1.000000e+00 : f32
            %c64_i32_91 = arith.constant 64 : i32
            %c0_92 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg31 = %c0_92 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
              %309 = ascvec.duplicate_scalar_nomask %cst : f32 -> <f32>
              %310 = ascvec.loadalign_brc %arg19[%arg20] : memref<96xf32, 1>, index -> <f32>
              %311 = ascvec.exp %310, %mask : <f32>
              %312 = ascvec.neg %311, %mask : <f32>
              %313 = ascvec.loadalign_unpack %arg21[%c0_90] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %314 = ascvec.loadalign_unpack %arg21[%c64] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %315 = ascvec.cast %313, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %316 = ascvec.cast %314, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %317 = ascvec.loadalign %arg22[%arg23] : memref<12288xf32, 1>, index -> <f32>
              %318 = ascvec.loadalign %arg22[%arg24] : memref<12288xf32, 1>, index -> <f32>
              %319 = ascvec.add %315, %317, %mask : <f32>
              %320 = ascvec.add %316, %318, %mask : <f32>
              %321 = ascvec.mul %312, %319, %mask : <f32>
              %322 = ascvec.mul %312, %320, %mask : <f32>
              %323 = ascvec.exp %321, %mask : <f32>
              %324 = ascvec.exp %322, %mask : <f32>
              %325 = ascvec.add_scalar %323, %cst, %mask : <f32>, f32
              %326 = ascvec.add_scalar %324, %cst, %mask : <f32>, f32
              %327 = ascvec.div %309, %325, %mask : <f32>
              %328 = ascvec.div %309, %326, %mask : <f32>
              %329 = ascvec.mul_scalar %327, %arg25, %mask : <f32>, f32
              %330 = ascvec.mul_scalar %328, %arg25, %mask : <f32>, f32
              %331 = ascvec.exp %329, %mask : <f32>
              %332 = ascvec.exp %330, %mask : <f32>
              ascvec.storealign %arg26[%c0_90], %331, %mask : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg26[%c64], %332, %mask : memref<128xf32, 1>, index, <f32>
              %333 = ascvec.loadalign_unpack %arg27[%c0_90] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
              %334 = ascvec.cast %333, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              %335 = ascvec.neg %334, %mask : <f32>
              %336 = ascvec.exp %335, %mask : <f32>
              %337 = ascvec.add_scalar %336, %cst, %mask : <f32>, f32
              %338 = ascvec.div %309, %337, %mask : <f32>
              ascvec.storealign %arg28[%c0_90], %338, %mask : memref<64xf32, 1>, index, <f32>
              %339 = ascvec.loadalign_unpack %arg29[%c0_90] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
              %340 = ascvec.cast %339, %mask {rounding = 6 : i32} : <bf16> -> <f32>
              ascvec.storealign %arg30[%c0_90], %340, %mask : memref<128xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %248, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %245, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %c15_i64, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %242, 0 {cannir.buffer.generated_sync}
          %255 = arith.addi %arg16, %c1_i64 : i64
          %256 = arith.cmpi slt, %255, %97#1 : i64
          %257 = scf.if %256 -> (i64) {
            %309 = arith.subi %255, %97#0 : i64
            %310 = arith.select %95, %255, %309 : i64
            %311 = arith.cmpi eq, %arg18, %c0_i64 : i64
            %312 = arith.select %311, %123, %124 : i64
            %313 = arith.select %311, %c7_i64, %c8_i64 : i64
            %314 = arith.addi %arg18, %c1_i64 overflow<nsw> : i64
            %315 = arith.remui %314, %c2_i64 : i64
            %316 = arith.minsi %310, %129 : i64
            %317 = arith.subi %129, %316 : i64
            %318 = arith.minsi %317, %c1_i64 : i64
            %319 = arith.cmpi sle, %310, %129 : i64
            %320 = arith.select %319, %318, %c0_i64 : i64
            %321 = arith.muli %310, %137 : i64
            %322 = arith.muli %321, %c2_i64 : i64
            %323 = arith.addi %145, %322 : i64
            %324 = ascscalar.bind_local_ptr %312 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %313, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%324, %alloc, %c0_i32, %c0_i32, %c0_i64, %323, %320, %c256_i64, %138, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %313, 0 {cannir.buffer.generated_sync}
            %325 = arith.select %311, %151, %152 : i64
            %326 = arith.select %311, %c9_i64, %c10_i64 : i64
            %327 = arith.addi %157, %322 : i64
            %328 = ascscalar.bind_local_ptr %325 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %326, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%328, %alloc, %c0_i32, %c0_i32, %c0_i64, %327, %320, %c256_i64, %138, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %326, 0 {cannir.buffer.generated_sync}
            %329 = arith.select %311, %161, %162 : i64
            %330 = arith.select %311, %c11_i64, %c12_i64 : i64
            %331 = arith.minsi %310, %165 : i64
            %332 = arith.subi %165, %331 : i64
            %333 = arith.minsi %332, %c1_i64 : i64
            %334 = arith.cmpi sle, %310, %165 : i64
            %335 = arith.select %334, %333, %c0_i64 : i64
            %336 = arith.muli %310, %173 : i64
            %337 = arith.muli %336, %c2_i64 : i64
            %338 = arith.addi %181, %337 : i64
            %339 = ascscalar.bind_local_ptr %329 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %330, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%339, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %338, %335, %c256_i64, %174, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %330, 0 {cannir.buffer.generated_sync}
            %340 = arith.select %311, %187, %188 : i64
            %341 = arith.select %311, %c13_i64, %c14_i64 : i64
            %342 = arith.minsi %310, %193 : i64
            %343 = arith.subi %193, %342 : i64
            %344 = arith.minsi %343, %c1_i64 : i64
            %345 = arith.cmpi sle, %310, %193 : i64
            %346 = arith.select %345, %344, %c0_i64 : i64
            %347 = arith.muli %310, %201 : i64
            %348 = arith.muli %347, %c2_i64 : i64
            %349 = arith.addi %209, %348 : i64
            %350 = ascscalar.bind_local_ptr %340 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %341, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%350, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %349, %346, %c2_i64, %202, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %341, 0 {cannir.buffer.generated_sync}
            %351 = arith.select %311, %215, %216 : i64
            %352 = arith.select %311, %c3_i64, %c4_i64 : i64
            %353 = arith.addi %223, %322 : i64
            %354 = arith.addi %353, %225 : i64
            %355 = ascscalar.bind_local_ptr %351 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %352, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%355, %alloc, %c0_i32, %c0_i32, %c0_i64, %354, %320, %c128_i64, %138, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %352, 0 {cannir.buffer.generated_sync}
            scf.yield %315 : i64
          } else {
            scf.yield %arg18 : i64
          }
          %258 = ascscalar.memref_address %alloc_17 : memref<16384xf32, 1>
          %259 = ascscalar.memref_address %alloc_18 : memref<16384xf32, 1>
          %260 = arith.select %231, %258, %259 : i64
          %261 = ascscalar.bind_local_ptr %260 : memref<16384xf32, 1>
          %262 = arith.select %231, %c1_i64, %c2_i64 : i64
          ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_13, %alloc_11, %alloc_10, %alloc_24 : memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>, memref<128xf32, 1>) {
          ^bb0(%arg19: memref<128xf32, 1>, %arg20: memref<128xf32, 1>, %arg21: memref<16384xf32, 1>, %arg22: memref<128xf32, 1>):
            %c64 = arith.constant 64 : index
            %c1_i32_90 = arith.constant 1 : i32
            %c0_i32_91 = arith.constant 0 : i32
            %c128_i32_92 = arith.constant 128 : i32
            %c0_93 = arith.constant 0 : index
            %c64_i32_94 = arith.constant 64 : i32
            %mask, %remaining_out = ascvec.update_mask %c64_i32_94 : !ascvec.mask
            %309 = ascvec.loadalign %arg19[%c0_93] : memref<128xf32, 1>, index -> <f32>
            %310 = ascvec.loadalign %arg19[%c64] : memref<128xf32, 1>, index -> <f32>
            %311 = ascvec.loadalign %arg20[%c0_93] : memref<128xf32, 1>, index -> <f32>
            %312 = ascvec.loadalign %arg20[%c64] : memref<128xf32, 1>, index -> <f32>
            %313 = ascvec.mul %309, %311, %mask : <f32>
            %314 = ascvec.mul %310, %312, %mask : <f32>
            scf.for %arg23 = %c0_i32_91 to %c64_i32_94 step %c1_i32_90  : i32 {
              %315 = arith.muli %arg23, %c128_i32_92 overflow<nsw> : i32
              %316 = arith.addi %315, %c64_i32_94 overflow<nsw> : i32
              %317 = arith.index_cast %315 : i32 to index
              %318 = ascvec.loadalign %arg21[%317] : memref<16384xf32, 1>, index -> <f32>
              %319 = arith.index_cast %316 : i32 to index
              %320 = ascvec.loadalign %arg21[%319] : memref<16384xf32, 1>, index -> <f32>
              %321 = ascvec.mul %318, %313, %mask : <f32>
              %322 = ascvec.madd_two_address %320, %314, %321, %mask : <f32>
              %323 = ascvec.reduce_sum %322, %mask : <f32>
              %324 = arith.index_cast %arg23 : i32 to index
              ascvec.storealign_1st %arg22[%324], %323 : memref<128xf32, 1>, index, <f32>
            } {cannir.loop_unroll = 4 : i64}
          } {cannir.vf_mode = "simd"}
          ascvec.vf(%alloc_15, %alloc_24, %alloc_14, %alloc_25 : memref<64xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>) {
          ^bb0(%arg19: memref<64xf32, 1>, %arg20: memref<128xf32, 1>, %arg21: memref<128xf32, 1>, %arg22: memref<128xf32, 1>):
            %c0_90 = arith.constant 0 : index
            %c64_i32_91 = arith.constant 64 : i32
            %c0_92 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg23 = %c0_92 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
              %309 = ascvec.loadalign_brc %arg19[%c0_90] : memref<64xf32, 1>, index -> <f32>
              %310 = ascvec.loadalign %arg20[%c0_90] : memref<128xf32, 1>, index -> <f32>
              %311 = ascvec.loadalign %arg21[%c0_90] : memref<128xf32, 1>, index -> <f32>
              %312 = ascvec.sub %311, %310, %mask : <f32>
              %313 = ascvec.mul %312, %309, %mask : <f32>
              ascvec.storealign %arg22[%c0_90], %313, %mask : memref<128xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.get_buf v, %262, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_13, %alloc_11, %alloc_12, %alloc_10, %alloc_25, %261, %alloc_16 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>, memref<128xf32, 1>, memref<16384xf32, 1>, memref<128xf32, 1>) {
          ^bb0(%arg19: memref<128xf32, 1>, %arg20: memref<128xf32, 1>, %arg21: memref<128xf32, 1>, %arg22: memref<16384xf32, 1>, %arg23: memref<128xf32, 1>, %arg24: memref<16384xf32, 1>, %arg25: memref<128xf32, 1>):
            %c64 = arith.constant 64 : index
            %c1_i32_90 = arith.constant 1 : i32
            %c0_i32_91 = arith.constant 0 : i32
            %c128_i32_92 = arith.constant 128 : i32
            %c0_93 = arith.constant 0 : index
            %c64_i32_94 = arith.constant 64 : i32
            %mask, %remaining_out = ascvec.update_mask %c64_i32_94 : !ascvec.mask
            %309 = ascvec.loadalign %arg19[%c0_93] : memref<128xf32, 1>, index -> <f32>
            %310 = ascvec.loadalign %arg19[%c64] : memref<128xf32, 1>, index -> <f32>
            %311 = ascvec.loadalign %arg20[%c0_93] : memref<128xf32, 1>, index -> <f32>
            %312 = ascvec.loadalign %arg20[%c64] : memref<128xf32, 1>, index -> <f32>
            %313 = ascvec.loadalign %arg21[%c0_93] : memref<128xf32, 1>, index -> <f32>
            %314 = ascvec.loadalign %arg21[%c64] : memref<128xf32, 1>, index -> <f32>
            scf.for %arg26 = %c0_i32_91 to %c64_i32_94 step %c1_i32_90  : i32 {
              %315 = arith.muli %arg26, %c128_i32_92 overflow<nsw> : i32
              %316 = arith.addi %315, %c64_i32_94 overflow<nsw> : i32
              %317 = arith.index_cast %315 : i32 to index
              %318 = ascvec.loadalign %arg22[%317] : memref<16384xf32, 1>, index -> <f32>
              %319 = arith.index_cast %316 : i32 to index
              %320 = ascvec.loadalign %arg22[%319] : memref<16384xf32, 1>, index -> <f32>
              %321 = arith.index_cast %arg26 : i32 to index
              %322 = ascvec.loadalign_brc %arg23[%321] : memref<128xf32, 1>, index -> <f32>
              %323 = ascvec.mul %322, %311, %mask : <f32>
              %324 = ascvec.mul %322, %312, %mask : <f32>
              %325 = ascvec.madd_two_address %318, %309, %323, %mask : <f32>
              %326 = ascvec.madd_two_address %320, %310, %324, %mask : <f32>
              ascvec.storealign %arg22[%317], %325, %mask : memref<16384xf32, 1>, index, <f32>
              ascvec.storealign %arg22[%319], %326, %mask : memref<16384xf32, 1>, index, <f32>
              ascvec.storealign %arg24[%317], %325, %mask : memref<16384xf32, 1>, index, <f32>
              ascvec.storealign %arg24[%319], %326, %mask : memref<16384xf32, 1>, index, <f32>
              %327 = ascvec.mul %325, %313, %mask : <f32>
              %328 = ascvec.madd_two_address %326, %314, %327, %mask : <f32>
              %329 = ascvec.reduce_sum %328, %mask : <f32>
              ascvec.storealign_1st %arg25[%321], %329 : memref<128xf32, 1>, index, <f32>
            } {cannir.loop_unroll = 1 : i64}
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %262, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          %263 = ascscalar.memref_address %alloc_21 : memref<128xbf16, 1>
          %264 = ascscalar.memref_address %alloc_22 : memref<128xbf16, 1>
          %265 = arith.select %231, %263, %264 : i64
          %266 = ascscalar.bind_local_ptr %265 : memref<128xbf16, 1>
          %267 = arith.select %231, %c5_i64, %c6_i64 : i64
          ascvec.get_buf v, %267, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_16, %266 : memref<128xf32, 1>, memref<128xbf16, 1>) {
          ^bb0(%arg19: memref<128xf32, 1>, %arg20: memref<128xbf16, 1>):
            %c0_90 = arith.constant 0 : index
            %c64_i32_91 = arith.constant 64 : i32
            %c0_92 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg21 = %c0_92 to %c1 step %c1 {
              %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
              %309 = ascvec.loadalign %arg19[%c0_90] : memref<128xf32, 1>, index -> <f32>
              %310 = ascvec.cast %309, %mask : <f32> -> <bf16>
              ascvec.storealign_pack %arg20[%c0_90], %310, %mask {mode = "b32_to_b16"} : memref<128xbf16, 1>, index, <bf16>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %267, 0 {cannir.buffer.generated_sync}
          %268 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
          %269 = cannir.extract_stride %268 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
          %270 = cannir.extract_dim %269[0] : !cannir.stride<(-1, -1, -1, 1)>
          %271 = arith.muli %96, %270 : i64
          %272 = cannir.extract_dim %269[1] : !cannir.stride<(-1, -1, -1, 1)>
          %273 = arith.muli %92, %272 : i64
          %274 = arith.addi %271, %273 : i64
          %275 = cannir.extract_dim %269[2] : !cannir.stride<(-1, -1, -1, 1)>
          %276 = arith.muli %230, %275 : i64
          %277 = arith.addi %274, %276 : i64
          %278 = arith.muli %79, %c64_i64 overflow<nsw> : i64
          %279 = arith.addi %277, %278 : i64
          %280 = arith.muli %279, %c16_i64 : i64
          %281 = ascscalar.memref_address %alloc_4 : memref<1xbf16>
          %282 = arith.divui %280, %c8_i64 : i64
          %283 = arith.addi %281, %282 : i64
          %284 = ascscalar.bind_gm_ptr %283 : memref<1xbf16>
          %285 = ascscalar.bind_local_ptr %265 : memref<1xbf16, 1>
          %286 = cannir.extract_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
          %287 = cannir.extract_stride %286 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
          %288 = cannir.extract_dim %287[2] : !cannir.stride<(-1, -1, -1, 1)>
          %289 = arith.index_cast %288 : i64 to index
          %290 = arith.muli %289, %c2 : index
          %291 = arith.index_cast %290 : index to i64
          ascvec.get_buf mte3, %267, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %284[%c0], %285[%c0], %c1_i32, %c128_i32, %291, %c128_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
          ascvec.release_buf mte3, %267, 0 {cannir.buffer.generated_sync}
          %292 = arith.index_cast %arg16 : i64 to index
          %293 = memref.load %alloc_7[%292] : memref<1xi32>
          %294 = arith.extsi %293 : i32 to i64
          %295 = cannir.extract_physical_layout %arg7 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
          %296 = cannir.extract_stride %295 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
          %297 = cannir.extract_dim %296[0] : !cannir.stride<(-1, 16384, 128, 1)>
          %298 = arith.muli %294, %297 : i64
          %299 = arith.muli %92, %c16384_i64 : i64
          %300 = arith.addi %298, %299 : i64
          %301 = arith.muli %79, %c8192_i64 overflow<nsw> : i64
          %302 = arith.addi %300, %301 : i64
          %303 = arith.muli %302, %c32_i64 : i64
          %304 = ascscalar.memref_address %alloc_6 : memref<1xf32>
          %305 = arith.divui %303, %c8_i64 : i64
          %306 = arith.addi %304, %305 : i64
          %307 = ascscalar.bind_gm_ptr %306 : memref<1xf32>
          %308 = ascscalar.bind_local_ptr %260 : memref<1xf32, 1>
          ascvec.get_buf mte3, %262, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %307[%c0], %308[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
          ascvec.release_buf mte3, %262, 0 {cannir.buffer.generated_sync}
          scf.yield %236, %257 : i64, i64
        }
      }
    } else {
      %5 = arith.cmpi sle, %3, %c16_i64 : i64
      scf.if %5 {
        %6 = arith.muli %3, %c4_i64 : i64
        %7 = ascscalar.get_block_num
        %8 = arith.cmpi sgt, %7, %6 : i64
        %9 = arith.select %8, %6, %7 : i64
        %10 = arith.addi %6, %9 : i64
        %11 = arith.subi %10, %c1_i64 : i64
        %12 = arith.remsi %11, %9 : i64
        %13 = arith.cmpi ne, %12, %c0_i64 : i64
        %14 = arith.cmpi slt, %12, %c0_i64 : i64
        %15 = arith.cmpi slt, %9, %c0_i64 : i64
        %16 = arith.xori %14, %15 : i1
        %17 = arith.andi %13, %16 : i1
        %18 = arith.divsi %11, %9 : i64
        %19 = arith.extui %17 : i1 to i64
        %20 = arith.subi %18, %19 : i64
        %21 = ascscalar.get_block_idx
        %22 = arith.muli %21, %20 : i64
        %23 = arith.addi %22, %20 : i64
        %24 = arith.cmpi sgt, %23, %6 : i64
        %25 = arith.select %24, %6, %23 : i64
        %26 = arith.cmpi slt, %21, %9 : i64
        scf.if %26 {
          %27 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>
          %28 = cannir.extract_shape %27 : <<(-1)>, <(1)>> -> <(-1)>
          %29 = cannir.extract_dim %28[0] : !cannir.shape<(-1)>
          %30 = arith.muli %29, %c4_i64 : i64
          %31 = arith.cmpi eq, %30, %c0_i64 : i64
          %32 = arith.cmpi ne, %30, %c0_i64 : i64
          %33 = arith.extui %32 : i1 to i64
          %34 = arith.select %31, %c4_i64, %30 : i64
          "ascvec.copy_gm2ub_plan"(%alloc_62, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %33, %34, %c0_i64, %34, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 119584 : i64, ub_capacity = 384 : i64}> : (memref<96xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          %35 = cannir.extract_layout %arg4 : !cannir.tensor<<gm, f32>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>
          %36 = cannir.extract_shape %35 : <<(-1, 128)>, <(128, 1)>> -> <(-1, 128)>
          %37 = cannir.extract_dim %36[0] : !cannir.shape<(-1, 128)>
          %38 = arith.muli %37, %c512_i64 : i64
          %39 = arith.cmpi eq, %38, %c0_i64 : i64
          %40 = arith.cmpi ne, %38, %c0_i64 : i64
          %41 = arith.extui %40 : i1 to i64
          %42 = arith.select %39, %c4_i64, %38 : i64
          ascvec.get_buf mte2, %c16_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%alloc_61, %alloc_3, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %41, %42, %c0_i64, %42, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 70432 : i64, ub_capacity = 49152 : i64}> : (memref<12288xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c16_i64, 0 {cannir.buffer.generated_sync}
          %43:3 = scf.for %arg16 = %22 to %25 step %c1_i64 iter_args(%arg17 = %c0_i64, %arg18 = %c0_i64, %arg19 = %c0_i64) -> (i64, i64, i64)  : i64 {
            %44 = arith.remsi %arg16, %c4_i64 : i64
            %45 = arith.cmpi ne, %44, %c0_i64 : i64
            %46 = arith.cmpi slt, %44, %c0_i64 : i64
            %47 = arith.andi %45, %46 : i1
            %48 = arith.divsi %arg16, %c4_i64 : i64
            %49 = arith.extui %47 : i1 to i64
            %50 = arith.subi %48, %49 : i64
            %51 = arith.muli %50, %c4_i64 : i64
            %52 = arith.subi %arg16, %51 : i64
            %53 = arith.maxsi %2, %c1_i64 : i64
            %54 = arith.remsi %50, %53 : i64
            %55 = arith.cmpi slt, %54, %c0_i64 : i64
            %56 = arith.divsi %50, %53 : i64
            %57 = arith.extui %55 : i1 to i64
            %58 = arith.subi %56, %57 : i64
            %59 = arith.maxsi %arg14, %c1_i64 : i64
            %60 = arith.remsi %58, %59 : i64
            %61 = arith.cmpi slt, %60, %c0_i64 : i64
            %62 = arith.select %61, %59, %c0_i64 : i64
            %63 = arith.addi %60, %62 : i64
            %64 = arith.select %55, %53, %c0_i64 : i64
            %65 = arith.addi %54, %64 : i64
            %66 = arith.muli %63, %arg15 : i64
            %67 = arith.addi %66, %arg15 : i64
            %68 = arith.cmpi eq, %arg13, %c1_i64 : i64
            %69 = arith.select %68, %c0_i64, %63 : i64
            %70:2 = scf.if %68 -> (i64, i64) {
              %222 = arith.index_cast %63 : i64 to index
              %223 = memref.load %alloc_9[%222] : memref<1xi32>
              %224 = arith.addi %63, %c1_i64 : i64
              %225 = arith.index_cast %224 : i64 to index
              %226 = memref.load %alloc_9[%225] : memref<1xi32>
              %227 = arith.extsi %223 : i32 to i64
              %228 = arith.extsi %226 : i32 to i64
              scf.yield %227, %228 : i64, i64
            } else {
              scf.yield %66, %67 : i64, i64
            }
            %71 = arith.index_cast %63 : i64 to index
            %72 = memref.load %alloc_8[%71] : memref<1xi32>
            %73 = arith.cmpi sgt, %72, %c0_i32 : i32
            %74 = arith.extsi %72 : i32 to i64
            %75 = arith.addi %70#0, %74 : i64
            %76 = arith.subi %75, %c1_i64 : i64
            %77 = arith.select %73, %76, %70#0 : i64
            %78 = arith.index_cast %77 : i64 to index
            %79 = memref.load %alloc_7[%78] : memref<1xi32>
            %80 = arith.extsi %79 : i32 to i64
            %81 = arith.cmpi eq, %arg17, %c0_i64 : i64
            %82 = ascscalar.memref_address %alloc_36 : memref<4096xf32, 1>
            %83 = ascscalar.memref_address %alloc_37 : memref<4096xf32, 1>
            %84 = arith.select %81, %82, %83 : i64
            %85 = ascscalar.bind_local_ptr %84 : memref<4096xf32, 1>
            %86 = arith.cmpi ne, %arg17, %c0_i64 : i64
            %87 = arith.extui %86 : i1 to i64
            %88 = arith.addi %arg17, %c1_i64 overflow<nsw> : i64
            %89 = arith.remui %88, %c2_i64 : i64
            %90 = arith.minsi %52, %c4_i64 : i64
            %91 = arith.muli %90, %c32_i64 : i64
            %92 = arith.subi %c128_i64, %91 : i64
            %93 = arith.minsi %92, %c32_i64 : i64
            %94 = arith.cmpi sle, %52, %c4_i64 : i64
            %95 = arith.select %94, %93, %c0_i64 : i64
            %96 = cannir.extract_physical_layout %arg6 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
            %97 = cannir.extract_stride %96 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
            %98 = cannir.extract_dim %97[0] : !cannir.stride<(-1, 16384, 128, 1)>
            %99 = arith.muli %80, %98 : i64
            %100 = arith.muli %99, %c4_i64 : i64
            %101 = arith.muli %65, %c65536_i64 : i64
            %102 = arith.addi %100, %101 : i64
            %103 = arith.muli %52, %c16384_i64 : i64
            %104 = arith.addi %102, %103 : i64
            %105 = arith.muli %95, %c512_i64 : i64
            %106 = arith.cmpi eq, %105, %c0_i64 : i64
            %107 = arith.cmpi ne, %105, %c0_i64 : i64
            %108 = arith.extui %107 : i1 to i64
            %109 = arith.select %106, %c4_i64, %105 : i64
            %110 = ascscalar.bind_local_ptr %84 : memref<1xf32, 1>
            ascvec.get_buf mte2, %87, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%110, %alloc_5, %c0_i32, %c0_i32, %c0_i64, %104, %108, %109, %c0_i64, %109, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %87, 0 {cannir.buffer.generated_sync}
            %111 = arith.select %68, %70#0, %c0_i64 : i64
            %112 = arith.addi %2, %65 : i64
            %113 = arith.muli %2, %c2_i64 : i64
            %114 = arith.addi %113, %65 : i64
            %115 = arith.cmpi eq, %arg19, %c0_i64 : i64
            %116 = ascscalar.memref_address %alloc_53 : memref<128xbf16, 1>
            %117 = ascscalar.memref_address %alloc_54 : memref<128xbf16, 1>
            %118 = arith.select %115, %116, %117 : i64
            %119 = arith.select %115, %c8_i64, %c9_i64 : i64
            %120 = arith.addi %arg19, %c1_i64 : i64
            %121 = arith.remui %120, %c2_i64 : i64
            %122 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
            %123 = cannir.extract_shape %122 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 128)>
            %124 = cannir.extract_dim %123[2] : !cannir.shape<(-1, -1, -1, 128)>
            %125 = arith.minsi %111, %124 : i64
            %126 = arith.subi %124, %125 : i64
            %127 = arith.minsi %126, %c1_i64 : i64
            %128 = arith.cmpi sle, %111, %124 : i64
            %129 = arith.select %128, %127, %c0_i64 : i64
            %130 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
            %131 = cannir.extract_stride %130 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
            %132 = cannir.extract_dim %131[2] : !cannir.stride<(-1, -1, -1, 1)>
            %133 = arith.muli %132, %c2_i64 : i64
            %134 = cannir.extract_dim %131[0] : !cannir.stride<(-1, -1, -1, 1)>
            %135 = arith.muli %69, %134 : i64
            %136 = arith.muli %135, %c2_i64 : i64
            %137 = cannir.extract_dim %131[1] : !cannir.stride<(-1, -1, -1, 1)>
            %138 = arith.muli %65, %137 : i64
            %139 = arith.muli %138, %c2_i64 : i64
            %140 = arith.addi %136, %139 : i64
            %141 = arith.muli %111, %132 : i64
            %142 = arith.muli %141, %c2_i64 : i64
            %143 = arith.addi %140, %142 : i64
            %144 = ascscalar.bind_local_ptr %118 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %119, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%144, %alloc, %c0_i32, %c0_i32, %c0_i64, %143, %129, %c256_i64, %133, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %119, 0 {cannir.buffer.generated_sync}
            %145 = ascscalar.memref_address %alloc_55 : memref<128xbf16, 1>
            %146 = ascscalar.memref_address %alloc_56 : memref<128xbf16, 1>
            %147 = arith.select %115, %145, %146 : i64
            %148 = arith.select %115, %c10_i64, %c11_i64 : i64
            %149 = arith.muli %112, %137 : i64
            %150 = arith.muli %149, %c2_i64 : i64
            %151 = arith.addi %136, %150 : i64
            %152 = arith.addi %151, %142 : i64
            %153 = ascscalar.bind_local_ptr %147 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %148, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%153, %alloc, %c0_i32, %c0_i32, %c0_i64, %152, %129, %c256_i64, %133, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %148, 0 {cannir.buffer.generated_sync}
            %154 = ascscalar.memref_address %alloc_57 : memref<128xbf16, 1>
            %155 = ascscalar.memref_address %alloc_58 : memref<128xbf16, 1>
            %156 = arith.select %115, %154, %155 : i64
            %157 = arith.select %115, %c12_i64, %c13_i64 : i64
            %158 = cannir.extract_dim %1[2] : !cannir.shape<(-1, -1, -1, 128)>
            %159 = arith.minsi %111, %158 : i64
            %160 = arith.subi %158, %159 : i64
            %161 = arith.minsi %160, %c1_i64 : i64
            %162 = arith.cmpi sle, %111, %158 : i64
            %163 = arith.select %162, %161, %c0_i64 : i64
            %164 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
            %165 = cannir.extract_stride %164 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
            %166 = cannir.extract_dim %165[2] : !cannir.stride<(-1, -1, -1, 1)>
            %167 = arith.muli %166, %c2_i64 : i64
            %168 = cannir.extract_dim %165[0] : !cannir.stride<(-1, -1, -1, 1)>
            %169 = arith.muli %69, %168 : i64
            %170 = arith.muli %169, %c2_i64 : i64
            %171 = cannir.extract_dim %165[1] : !cannir.stride<(-1, -1, -1, 1)>
            %172 = arith.muli %65, %171 : i64
            %173 = arith.muli %172, %c2_i64 : i64
            %174 = arith.addi %170, %173 : i64
            %175 = arith.muli %111, %166 : i64
            %176 = arith.muli %175, %c2_i64 : i64
            %177 = arith.addi %174, %176 : i64
            %178 = ascscalar.bind_local_ptr %156 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %157, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%178, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %177, %163, %c256_i64, %167, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %157, 0 {cannir.buffer.generated_sync}
            %179 = ascscalar.memref_address %alloc_59 : memref<64xbf16, 1>
            %180 = ascscalar.memref_address %alloc_60 : memref<64xbf16, 1>
            %181 = arith.select %115, %179, %180 : i64
            %182 = arith.select %115, %c14_i64, %c15_i64 : i64
            %183 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>
            %184 = cannir.extract_shape %183 : <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
            %185 = cannir.extract_dim %184[2] : !cannir.shape<(-1, -1, -1, 1)>
            %186 = arith.minsi %111, %185 : i64
            %187 = arith.subi %185, %186 : i64
            %188 = arith.minsi %187, %c1_i64 : i64
            %189 = arith.cmpi sle, %111, %185 : i64
            %190 = arith.select %189, %188, %c0_i64 : i64
            %191 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>
            %192 = cannir.extract_stride %191 : <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
            %193 = cannir.extract_dim %192[2] : !cannir.stride<(-1, -1, -1, 1)>
            %194 = arith.muli %193, %c2_i64 : i64
            %195 = cannir.extract_dim %192[0] : !cannir.stride<(-1, -1, -1, 1)>
            %196 = arith.muli %69, %195 : i64
            %197 = arith.muli %196, %c2_i64 : i64
            %198 = cannir.extract_dim %192[1] : !cannir.stride<(-1, -1, -1, 1)>
            %199 = arith.muli %65, %198 : i64
            %200 = arith.muli %199, %c2_i64 : i64
            %201 = arith.addi %197, %200 : i64
            %202 = arith.muli %111, %193 : i64
            %203 = arith.muli %202, %c2_i64 : i64
            %204 = arith.addi %201, %203 : i64
            %205 = ascscalar.bind_local_ptr %181 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %182, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%205, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %204, %190, %c2_i64, %194, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %182, 0 {cannir.buffer.generated_sync}
            %206 = ascscalar.memref_address %alloc_46 : memref<64xbf16, 1>
            %207 = ascscalar.memref_address %alloc_47 : memref<64xbf16, 1>
            %208 = arith.select %115, %206, %207 : i64
            %209 = arith.select %115, %c4_i64, %c5_i64 : i64
            %210 = arith.muli %114, %137 : i64
            %211 = arith.muli %210, %c2_i64 : i64
            %212 = arith.addi %136, %211 : i64
            %213 = arith.addi %212, %142 : i64
            %214 = arith.muli %52, %c64_i64 : i64
            %215 = arith.addi %213, %214 : i64
            %216 = arith.muli %95, %c2_i64 : i64
            %217 = arith.cmpi eq, %216, %c0_i64 : i64
            %218 = arith.select %217, %c0_i64, %129 : i64
            %219 = arith.select %217, %c2_i64, %216 : i64
            %220 = ascscalar.bind_local_ptr %208 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %209, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%220, %alloc, %c0_i32, %c0_i32, %c0_i64, %215, %218, %219, %133, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %209, 0 {cannir.buffer.generated_sync}
            %221:2 = scf.for %arg20 = %70#0 to %70#1 step %c1_i64 iter_args(%arg21 = %arg18, %arg22 = %121) -> (i64, i64)  : i64 {
              %222 = arith.subi %arg20, %70#0 : i64
              %223 = arith.select %68, %arg20, %222 : i64
              %224 = arith.cmpi eq, %arg21, %c0_i64 : i64
              %225 = arith.select %224, %116, %117 : i64
              %226 = ascscalar.bind_local_ptr %225 : memref<128xbf16, 1>
              %227 = arith.select %224, %c8_i64, %c9_i64 : i64
              %228 = arith.addi %arg21, %c1_i64 overflow<nsw> : i64
              %229 = arith.remui %228, %c2_i64 : i64
              %230 = arith.select %224, %145, %146 : i64
              %231 = ascscalar.bind_local_ptr %230 : memref<128xbf16, 1>
              %232 = arith.select %224, %c10_i64, %c11_i64 : i64
              ascvec.get_buf v, %227, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %232, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%226, %231, %alloc_39, %alloc_38, %alloc_50 : memref<128xbf16, 1>, memref<128xbf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<4xf32, 1>) {
              ^bb0(%arg23: memref<128xbf16, 1>, %arg24: memref<128xbf16, 1>, %arg25: memref<128xf32, 1>, %arg26: memref<128xf32, 1>, %arg27: memref<4xf32, 1>):
                %c3 = arith.constant 3 : index
                %c2_90 = arith.constant 2 : index
                %c1 = arith.constant 1 : index
                %c64 = arith.constant 64 : index
                %c0_91 = arith.constant 0 : index
                %c64_i32_92 = arith.constant 64 : i32
                %c0_93 = arith.constant 0 : index
                %c1_94 = arith.constant 1 : index
                scf.for %arg28 = %c0_93 to %c1_94 step %c1_94 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_92 : !ascvec.mask
                  %302 = ascvec.loadalign_unpack %arg23[%c0_91] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
                  %303 = ascvec.loadalign_unpack %arg23[%c64] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
                  %304 = ascvec.loadalign_unpack %arg24[%c0_91] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
                  %305 = ascvec.loadalign_unpack %arg24[%c64] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
                  %306 = ascvec.cast %302, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %307 = ascvec.cast %303, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %308 = ascvec.cast %304, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %309 = ascvec.cast %305, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %310 = ascvec.mul %306, %306, %mask : <f32>
                  %311 = ascvec.mul %307, %307, %mask : <f32>
                  %312 = ascvec.mul %308, %308, %mask : <f32>
                  %313 = ascvec.mul %309, %309, %mask : <f32>
                  %314 = ascvec.reduce_sum %310, %mask : <f32>
                  %315 = ascvec.reduce_sum %311, %mask : <f32>
                  %316 = ascvec.reduce_sum %312, %mask : <f32>
                  %317 = ascvec.reduce_sum %313, %mask : <f32>
                  ascvec.storealign %arg25[%c0_91], %306, %mask : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign %arg25[%c64], %307, %mask : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign %arg26[%c0_91], %308, %mask : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign %arg26[%c64], %309, %mask : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign_1st %arg27[%c0_91], %314 : memref<4xf32, 1>, index, <f32>
                  ascvec.storealign_1st %arg27[%c1], %315 : memref<4xf32, 1>, index, <f32>
                  ascvec.storealign_1st %arg27[%c2_90], %316 : memref<4xf32, 1>, index, <f32>
                  ascvec.storealign_1st %arg27[%c3], %317 : memref<4xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %232, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %227, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_50 : memref<4xf32, 1>) {
              ^bb0(%arg23: memref<4xf32, 1>):
                %c3 = arith.constant 3 : index
                %c2_90 = arith.constant 2 : index
                %c1 = arith.constant 1 : index
                %cst = arith.constant 9.99999997E-7 : f32
                %c0_91 = arith.constant 0 : index
                %c1_i32_92 = arith.constant 1 : i32
                %c0_93 = arith.constant 0 : index
                %c1_94 = arith.constant 1 : index
                scf.for %arg24 = %c0_93 to %c1_94 step %c1_94 {
                  %mask, %remaining_out = ascvec.update_mask %c1_i32_92 : !ascvec.mask
                  %302 = ascvec.loadalign_brc %arg23[%c0_91] : memref<4xf32, 1>, index -> <f32>
                  %303 = ascvec.loadalign_brc %arg23[%c1] : memref<4xf32, 1>, index -> <f32>
                  %304 = ascvec.loadalign_brc %arg23[%c2_90] : memref<4xf32, 1>, index -> <f32>
                  %305 = ascvec.loadalign_brc %arg23[%c3] : memref<4xf32, 1>, index -> <f32>
                  %306 = ascvec.add %302, %303, %mask : <f32>
                  %307 = ascvec.add %304, %305, %mask : <f32>
                  %308 = ascvec.add_scalar %306, %cst, %mask : <f32>, f32
                  %309 = ascvec.add_scalar %307, %cst, %mask : <f32>, f32
                  %310 = ascvec.sqrt %308, %mask : <f32>
                  %311 = ascvec.sqrt %309, %mask : <f32>
                  ascvec.storealign_1st %arg23[%c0_91], %310 : memref<4xf32, 1>, index, <f32>
                  ascvec.storealign_1st %arg23[%c1], %311 : memref<4xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.vf(%alloc_50, %alloc_39, %alloc_38, %arg11 : memref<4xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, f32) {
              ^bb0(%arg23: memref<4xf32, 1>, %arg24: memref<128xf32, 1>, %arg25: memref<128xf32, 1>, %arg26: f32):
                %c64 = arith.constant 64 : index
                %c1 = arith.constant 1 : index
                %c0_90 = arith.constant 0 : index
                %c64_i32_91 = arith.constant 64 : i32
                %c0_92 = arith.constant 0 : index
                %c1_93 = arith.constant 1 : index
                scf.for %arg27 = %c0_92 to %c1_93 step %c1_93 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
                  %302 = ascvec.loadalign_brc %arg23[%c0_90] : memref<4xf32, 1>, index -> <f32>
                  %303 = ascvec.loadalign_brc %arg23[%c1] : memref<4xf32, 1>, index -> <f32>
                  %304 = ascvec.duplicate %302, %mask : <f32> -> <f32>
                  %305 = ascvec.duplicate %303, %mask : <f32> -> <f32>
                  %306 = ascvec.loadalign %arg24[%c0_90] : memref<128xf32, 1>, index -> <f32>
                  %307 = ascvec.loadalign %arg24[%c64] : memref<128xf32, 1>, index -> <f32>
                  %308 = ascvec.loadalign %arg25[%c0_90] : memref<128xf32, 1>, index -> <f32>
                  %309 = ascvec.loadalign %arg25[%c64] : memref<128xf32, 1>, index -> <f32>
                  %310 = ascvec.div %306, %304, %mask : <f32>
                  %311 = ascvec.div %307, %304, %mask : <f32>
                  %312 = ascvec.div %308, %305, %mask : <f32>
                  %313 = ascvec.div %309, %305, %mask : <f32>
                  %314 = ascvec.mul_scalar %310, %arg26, %mask : <f32>, f32
                  %315 = ascvec.mul_scalar %311, %arg26, %mask : <f32>, f32
                  ascvec.storealign %arg24[%c0_90], %314, %mask : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign %arg24[%c64], %315, %mask : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign %arg25[%c0_90], %312, %mask : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign %arg25[%c64], %313, %mask : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              %233 = arith.select %224, %154, %155 : i64
              %234 = ascscalar.bind_local_ptr %233 : memref<128xbf16, 1>
              %235 = arith.select %224, %c12_i64, %c13_i64 : i64
              %236 = arith.select %224, %179, %180 : i64
              %237 = ascscalar.bind_local_ptr %236 : memref<64xbf16, 1>
              %238 = arith.select %224, %c14_i64, %c15_i64 : i64
              %239 = arith.select %224, %206, %207 : i64
              %240 = ascscalar.bind_local_ptr %239 : memref<64xbf16, 1>
              %241 = arith.select %224, %c4_i64, %c5_i64 : i64
              %242 = arith.trunci %65 : i64 to i32
              %243 = arith.index_cast %242 : i32 to index
              %244 = arith.muli %242, %c128_i32 : i32
              %245 = arith.index_cast %244 : i32 to index
              %246 = arith.addi %244, %c64_i32 : i32
              %247 = arith.index_cast %246 : i32 to index
              ascvec.get_buf v, %235, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %238, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %241, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_62, %243, %234, %alloc_61, %245, %247, %arg12, %alloc_40, %237, %alloc_42, %240, %alloc_41 : memref<96xf32, 1>, index, memref<128xbf16, 1>, memref<12288xf32, 1>, index, index, f32, memref<128xf32, 1>, memref<64xbf16, 1>, memref<64xf32, 1>, memref<64xbf16, 1>, memref<64xf32, 1>) {
              ^bb0(%arg23: memref<96xf32, 1>, %arg24: index, %arg25: memref<128xbf16, 1>, %arg26: memref<12288xf32, 1>, %arg27: index, %arg28: index, %arg29: f32, %arg30: memref<128xf32, 1>, %arg31: memref<64xbf16, 1>, %arg32: memref<64xf32, 1>, %arg33: memref<64xbf16, 1>, %arg34: memref<64xf32, 1>):
                %c64 = arith.constant 64 : index
                %c0_90 = arith.constant 0 : index
                %cst = arith.constant 1.000000e+00 : f32
                %c64_i32_91 = arith.constant 64 : i32
                %c0_92 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg35 = %c0_92 to %c1 step %c1 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
                  %302 = ascvec.duplicate_scalar_nomask %cst : f32 -> <f32>
                  %303 = ascvec.loadalign_brc %arg23[%arg24] : memref<96xf32, 1>, index -> <f32>
                  %304 = ascvec.exp %303, %mask : <f32>
                  %305 = ascvec.neg %304, %mask : <f32>
                  %306 = ascvec.loadalign_unpack %arg25[%c0_90] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
                  %307 = ascvec.loadalign_unpack %arg25[%c64] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
                  %308 = ascvec.cast %306, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %309 = ascvec.cast %307, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %310 = ascvec.loadalign %arg26[%arg27] : memref<12288xf32, 1>, index -> <f32>
                  %311 = ascvec.loadalign %arg26[%arg28] : memref<12288xf32, 1>, index -> <f32>
                  %312 = ascvec.add %308, %310, %mask : <f32>
                  %313 = ascvec.add %309, %311, %mask : <f32>
                  %314 = ascvec.mul %305, %312, %mask : <f32>
                  %315 = ascvec.mul %305, %313, %mask : <f32>
                  %316 = ascvec.exp %314, %mask : <f32>
                  %317 = ascvec.exp %315, %mask : <f32>
                  %318 = ascvec.add_scalar %316, %cst, %mask : <f32>, f32
                  %319 = ascvec.add_scalar %317, %cst, %mask : <f32>, f32
                  %320 = ascvec.div %302, %318, %mask : <f32>
                  %321 = ascvec.div %302, %319, %mask : <f32>
                  %322 = ascvec.mul_scalar %320, %arg29, %mask : <f32>, f32
                  %323 = ascvec.mul_scalar %321, %arg29, %mask : <f32>, f32
                  %324 = ascvec.exp %322, %mask : <f32>
                  %325 = ascvec.exp %323, %mask : <f32>
                  ascvec.storealign %arg30[%c0_90], %324, %mask : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign %arg30[%c64], %325, %mask : memref<128xf32, 1>, index, <f32>
                  %326 = ascvec.loadalign_unpack %arg31[%c0_90] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
                  %327 = ascvec.cast %326, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %328 = ascvec.neg %327, %mask : <f32>
                  %329 = ascvec.exp %328, %mask : <f32>
                  %330 = ascvec.add_scalar %329, %cst, %mask : <f32>, f32
                  %331 = ascvec.div %302, %330, %mask : <f32>
                  ascvec.storealign %arg32[%c0_90], %331, %mask : memref<64xf32, 1>, index, <f32>
                  %332 = ascvec.loadalign_unpack %arg33[%c0_90] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
                  %333 = ascvec.cast %332, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  ascvec.storealign %arg34[%c0_90], %333, %mask : memref<64xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %241, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %238, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %235, 0 {cannir.buffer.generated_sync}
              %248 = arith.addi %arg20, %c1_i64 : i64
              %249 = arith.cmpi slt, %248, %70#1 : i64
              %250 = scf.if %249 -> (i64) {
                %302 = arith.subi %248, %70#0 : i64
                %303 = arith.select %68, %248, %302 : i64
                %304 = arith.cmpi eq, %arg22, %c0_i64 : i64
                %305 = arith.select %304, %116, %117 : i64
                %306 = arith.select %304, %c8_i64, %c9_i64 : i64
                %307 = arith.addi %arg22, %c1_i64 overflow<nsw> : i64
                %308 = arith.remui %307, %c2_i64 : i64
                %309 = arith.minsi %303, %124 : i64
                %310 = arith.subi %124, %309 : i64
                %311 = arith.minsi %310, %c1_i64 : i64
                %312 = arith.cmpi sle, %303, %124 : i64
                %313 = arith.select %312, %311, %c0_i64 : i64
                %314 = arith.muli %303, %132 : i64
                %315 = arith.muli %314, %c2_i64 : i64
                %316 = arith.addi %140, %315 : i64
                %317 = ascscalar.bind_local_ptr %305 : memref<1xbf16, 1>
                ascvec.get_buf mte2, %306, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%317, %alloc, %c0_i32, %c0_i32, %c0_i64, %316, %313, %c256_i64, %133, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %306, 0 {cannir.buffer.generated_sync}
                %318 = arith.select %304, %145, %146 : i64
                %319 = arith.select %304, %c10_i64, %c11_i64 : i64
                %320 = arith.addi %151, %315 : i64
                %321 = ascscalar.bind_local_ptr %318 : memref<1xbf16, 1>
                ascvec.get_buf mte2, %319, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%321, %alloc, %c0_i32, %c0_i32, %c0_i64, %320, %313, %c256_i64, %133, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %319, 0 {cannir.buffer.generated_sync}
                %322 = arith.select %304, %154, %155 : i64
                %323 = arith.select %304, %c12_i64, %c13_i64 : i64
                %324 = arith.minsi %303, %158 : i64
                %325 = arith.subi %158, %324 : i64
                %326 = arith.minsi %325, %c1_i64 : i64
                %327 = arith.cmpi sle, %303, %158 : i64
                %328 = arith.select %327, %326, %c0_i64 : i64
                %329 = arith.muli %303, %166 : i64
                %330 = arith.muli %329, %c2_i64 : i64
                %331 = arith.addi %174, %330 : i64
                %332 = ascscalar.bind_local_ptr %322 : memref<1xbf16, 1>
                ascvec.get_buf mte2, %323, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%332, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %331, %328, %c256_i64, %167, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %323, 0 {cannir.buffer.generated_sync}
                %333 = arith.select %304, %179, %180 : i64
                %334 = arith.select %304, %c14_i64, %c15_i64 : i64
                %335 = arith.minsi %303, %185 : i64
                %336 = arith.subi %185, %335 : i64
                %337 = arith.minsi %336, %c1_i64 : i64
                %338 = arith.cmpi sle, %303, %185 : i64
                %339 = arith.select %338, %337, %c0_i64 : i64
                %340 = arith.muli %303, %193 : i64
                %341 = arith.muli %340, %c2_i64 : i64
                %342 = arith.addi %201, %341 : i64
                %343 = ascscalar.bind_local_ptr %333 : memref<1xbf16, 1>
                ascvec.get_buf mte2, %334, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%343, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %342, %339, %c2_i64, %194, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %334, 0 {cannir.buffer.generated_sync}
                %344 = arith.select %304, %206, %207 : i64
                %345 = arith.select %304, %c4_i64, %c5_i64 : i64
                %346 = arith.addi %212, %315 : i64
                %347 = arith.addi %346, %214 : i64
                %348 = arith.select %217, %c0_i64, %313 : i64
                %349 = ascscalar.bind_local_ptr %344 : memref<1xbf16, 1>
                ascvec.get_buf mte2, %345, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%349, %alloc, %c0_i32, %c0_i32, %c0_i64, %347, %348, %219, %133, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %345, 0 {cannir.buffer.generated_sync}
                scf.yield %308 : i64
              } else {
                scf.yield %arg22 : i64
              }
              %251 = ascscalar.memref_address %alloc_44 : memref<4096xf32, 1>
              %252 = ascscalar.memref_address %alloc_45 : memref<4096xf32, 1>
              %253 = arith.select %224, %251, %252 : i64
              %254 = ascscalar.bind_local_ptr %253 : memref<4096xf32, 1>
              %255 = arith.select %224, %c2_i64, %c3_i64 : i64
              ascvec.get_buf v, %87, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_40, %alloc_38, %85, %alloc_51 : memref<128xf32, 1>, memref<128xf32, 1>, memref<4096xf32, 1>, memref<64xf32, 1>) {
              ^bb0(%arg23: memref<128xf32, 1>, %arg24: memref<128xf32, 1>, %arg25: memref<4096xf32, 1>, %arg26: memref<64xf32, 1>):
                %c64 = arith.constant 64 : index
                %c1_i32_90 = arith.constant 1 : i32
                %c32_i32 = arith.constant 32 : i32
                %c0_i32_91 = arith.constant 0 : i32
                %c128_i32_92 = arith.constant 128 : i32
                %c0_93 = arith.constant 0 : index
                %c64_i32_94 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_94 : !ascvec.mask
                %302 = ascvec.loadalign %arg23[%c0_93] : memref<128xf32, 1>, index -> <f32>
                %303 = ascvec.loadalign %arg23[%c64] : memref<128xf32, 1>, index -> <f32>
                %304 = ascvec.loadalign %arg24[%c0_93] : memref<128xf32, 1>, index -> <f32>
                %305 = ascvec.loadalign %arg24[%c64] : memref<128xf32, 1>, index -> <f32>
                %306 = ascvec.mul %302, %304, %mask : <f32>
                %307 = ascvec.mul %303, %305, %mask : <f32>
                scf.for %arg27 = %c0_i32_91 to %c32_i32 step %c1_i32_90  : i32 {
                  %308 = arith.muli %arg27, %c128_i32_92 overflow<nsw> : i32
                  %309 = arith.addi %308, %c64_i32_94 overflow<nsw> : i32
                  %310 = arith.index_cast %308 : i32 to index
                  %311 = ascvec.loadalign %arg25[%310] : memref<4096xf32, 1>, index -> <f32>
                  %312 = arith.index_cast %309 : i32 to index
                  %313 = ascvec.loadalign %arg25[%312] : memref<4096xf32, 1>, index -> <f32>
                  %314 = ascvec.mul %311, %306, %mask : <f32>
                  %315 = ascvec.madd_two_address %313, %307, %314, %mask : <f32>
                  %316 = ascvec.reduce_sum %315, %mask : <f32>
                  %317 = arith.index_cast %arg27 : i32 to index
                  ascvec.storealign_1st %arg26[%317], %316 : memref<64xf32, 1>, index, <f32>
                } {cannir.loop_unroll = 4 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.vf(%alloc_42, %alloc_51, %alloc_41, %alloc_52 : memref<64xf32, 1>, memref<64xf32, 1>, memref<64xf32, 1>, memref<64xf32, 1>) {
              ^bb0(%arg23: memref<64xf32, 1>, %arg24: memref<64xf32, 1>, %arg25: memref<64xf32, 1>, %arg26: memref<64xf32, 1>):
                %c0_90 = arith.constant 0 : index
                %c64_i32_91 = arith.constant 64 : i32
                %c0_92 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg27 = %c0_92 to %c1 step %c1 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
                  %302 = ascvec.loadalign_brc %arg23[%c0_90] : memref<64xf32, 1>, index -> <f32>
                  %303 = ascvec.loadalign %arg24[%c0_90] : memref<64xf32, 1>, index -> <f32>
                  %304 = ascvec.loadalign %arg25[%c0_90] : memref<64xf32, 1>, index -> <f32>
                  %305 = ascvec.sub %304, %303, %mask : <f32>
                  %306 = ascvec.mul %305, %302, %mask : <f32>
                  ascvec.storealign %arg26[%c0_90], %306, %mask : memref<64xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %255, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_40, %alloc_38, %alloc_39, %85, %alloc_52, %254, %alloc_43 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<4096xf32, 1>, memref<64xf32, 1>, memref<4096xf32, 1>, memref<32xf32, 1>) {
              ^bb0(%arg23: memref<128xf32, 1>, %arg24: memref<128xf32, 1>, %arg25: memref<128xf32, 1>, %arg26: memref<4096xf32, 1>, %arg27: memref<64xf32, 1>, %arg28: memref<4096xf32, 1>, %arg29: memref<32xf32, 1>):
                %c64 = arith.constant 64 : index
                %c1_i32_90 = arith.constant 1 : i32
                %c32_i32 = arith.constant 32 : i32
                %c0_i32_91 = arith.constant 0 : i32
                %c128_i32_92 = arith.constant 128 : i32
                %c0_93 = arith.constant 0 : index
                %c64_i32_94 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_94 : !ascvec.mask
                %302 = ascvec.loadalign %arg23[%c0_93] : memref<128xf32, 1>, index -> <f32>
                %303 = ascvec.loadalign %arg23[%c64] : memref<128xf32, 1>, index -> <f32>
                %304 = ascvec.loadalign %arg24[%c0_93] : memref<128xf32, 1>, index -> <f32>
                %305 = ascvec.loadalign %arg24[%c64] : memref<128xf32, 1>, index -> <f32>
                %306 = ascvec.loadalign %arg25[%c0_93] : memref<128xf32, 1>, index -> <f32>
                %307 = ascvec.loadalign %arg25[%c64] : memref<128xf32, 1>, index -> <f32>
                scf.for %arg30 = %c0_i32_91 to %c32_i32 step %c1_i32_90  : i32 {
                  %308 = arith.muli %arg30, %c128_i32_92 overflow<nsw> : i32
                  %309 = arith.addi %308, %c64_i32_94 overflow<nsw> : i32
                  %310 = arith.index_cast %308 : i32 to index
                  %311 = ascvec.loadalign %arg26[%310] : memref<4096xf32, 1>, index -> <f32>
                  %312 = arith.index_cast %309 : i32 to index
                  %313 = ascvec.loadalign %arg26[%312] : memref<4096xf32, 1>, index -> <f32>
                  %314 = arith.index_cast %arg30 : i32 to index
                  %315 = ascvec.loadalign_brc %arg27[%314] : memref<64xf32, 1>, index -> <f32>
                  %316 = ascvec.mul %315, %304, %mask : <f32>
                  %317 = ascvec.mul %315, %305, %mask : <f32>
                  %318 = ascvec.madd_two_address %311, %302, %316, %mask : <f32>
                  %319 = ascvec.madd_two_address %313, %303, %317, %mask : <f32>
                  ascvec.storealign %arg26[%310], %318, %mask : memref<4096xf32, 1>, index, <f32>
                  ascvec.storealign %arg26[%312], %319, %mask : memref<4096xf32, 1>, index, <f32>
                  ascvec.storealign %arg28[%310], %318, %mask : memref<4096xf32, 1>, index, <f32>
                  ascvec.storealign %arg28[%312], %319, %mask : memref<4096xf32, 1>, index, <f32>
                  %320 = ascvec.mul %318, %306, %mask : <f32>
                  %321 = ascvec.madd_two_address %319, %307, %320, %mask : <f32>
                  %322 = ascvec.reduce_sum %321, %mask : <f32>
                  ascvec.storealign_1st %arg29[%314], %322 : memref<32xf32, 1>, index, <f32>
                } {cannir.loop_unroll = 1 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %255, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %87, 0 {cannir.buffer.generated_sync}
              %256 = ascscalar.memref_address %alloc_48 : memref<32xbf16, 1>
              %257 = ascscalar.memref_address %alloc_49 : memref<32xbf16, 1>
              %258 = arith.select %224, %256, %257 : i64
              %259 = ascscalar.bind_local_ptr %258 : memref<32xbf16, 1>
              %260 = arith.select %224, %c6_i64, %c7_i64 : i64
              ascvec.get_buf v, %260, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_43, %259 : memref<32xf32, 1>, memref<32xbf16, 1>) {
              ^bb0(%arg23: memref<32xf32, 1>, %arg24: memref<32xbf16, 1>):
                %c0_90 = arith.constant 0 : index
                %c32_i32 = arith.constant 32 : i32
                %c0_91 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg25 = %c0_91 to %c1 step %c1 {
                  %mask, %remaining_out = ascvec.update_mask %c32_i32 : !ascvec.mask
                  %302 = ascvec.loadalign %arg23[%c0_90] : memref<32xf32, 1>, index -> <f32>
                  %303 = ascvec.cast %302, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg24[%c0_90], %303, %mask {mode = "b32_to_b16"} : memref<32xbf16, 1>, index, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %260, 0 {cannir.buffer.generated_sync}
              %261 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %262 = cannir.extract_stride %261 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
              %263 = cannir.extract_dim %262[0] : !cannir.stride<(-1, -1, -1, 1)>
              %264 = arith.muli %69, %263 : i64
              %265 = cannir.extract_dim %262[1] : !cannir.stride<(-1, -1, -1, 1)>
              %266 = arith.muli %65, %265 : i64
              %267 = arith.addi %264, %266 : i64
              %268 = cannir.extract_dim %262[2] : !cannir.stride<(-1, -1, -1, 1)>
              %269 = arith.muli %223, %268 : i64
              %270 = arith.addi %267, %269 : i64
              %271 = arith.muli %52, %c32_i64 : i64
              %272 = arith.addi %270, %271 : i64
              %273 = arith.muli %272, %c16_i64 : i64
              %274 = ascscalar.memref_address %alloc_4 : memref<1xbf16>
              %275 = arith.divui %273, %c8_i64 : i64
              %276 = arith.addi %274, %275 : i64
              %277 = ascscalar.bind_gm_ptr %276 : memref<1xbf16>
              %278 = ascscalar.bind_local_ptr %258 : memref<1xbf16, 1>
              %279 = cannir.extract_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %280 = cannir.extract_stride %279 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
              %281 = cannir.extract_dim %280[2] : !cannir.stride<(-1, -1, -1, 1)>
              %282 = arith.index_cast %281 : i64 to index
              %283 = arith.muli %282, %c2 : index
              %284 = arith.index_cast %283 : index to i64
              ascvec.get_buf mte3, %260, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %277[%c0], %278[%c0], %c1_i32, %c64_i32, %284, %c64_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
              ascvec.release_buf mte3, %260, 0 {cannir.buffer.generated_sync}
              %285 = arith.index_cast %arg20 : i64 to index
              %286 = memref.load %alloc_7[%285] : memref<1xi32>
              %287 = arith.extsi %286 : i32 to i64
              %288 = cannir.extract_physical_layout %arg7 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
              %289 = cannir.extract_stride %288 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
              %290 = cannir.extract_dim %289[0] : !cannir.stride<(-1, 16384, 128, 1)>
              %291 = arith.muli %287, %290 : i64
              %292 = arith.muli %65, %c16384_i64 : i64
              %293 = arith.addi %291, %292 : i64
              %294 = arith.muli %52, %c4096_i64 : i64
              %295 = arith.addi %293, %294 : i64
              %296 = arith.muli %295, %c32_i64 : i64
              %297 = ascscalar.memref_address %alloc_6 : memref<1xf32>
              %298 = arith.divui %296, %c8_i64 : i64
              %299 = arith.addi %297, %298 : i64
              %300 = ascscalar.bind_gm_ptr %299 : memref<1xf32>
              %301 = ascscalar.bind_local_ptr %253 : memref<1xf32, 1>
              ascvec.get_buf mte3, %255, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %300[%c0], %301[%c0], %c1_i32, %c16384_i32, %c0_i64, %c16384_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
              ascvec.release_buf mte3, %255, 0 {cannir.buffer.generated_sync}
              scf.yield %229, %250 : i64, i64
            }
            scf.yield %89, %221#0, %221#1 : i64, i64, i64
          }
        }
      } else {
        %6 = arith.muli %3, %c2_i64 : i64
        %7 = ascscalar.get_block_num
        %8 = arith.cmpi sgt, %7, %6 : i64
        %9 = arith.select %8, %6, %7 : i64
        %10 = arith.addi %6, %9 : i64
        %11 = arith.subi %10, %c1_i64 : i64
        %12 = arith.remsi %11, %9 : i64
        %13 = arith.cmpi ne, %12, %c0_i64 : i64
        %14 = arith.cmpi slt, %12, %c0_i64 : i64
        %15 = arith.cmpi slt, %9, %c0_i64 : i64
        %16 = arith.xori %14, %15 : i1
        %17 = arith.andi %13, %16 : i1
        %18 = arith.divsi %11, %9 : i64
        %19 = arith.extui %17 : i1 to i64
        %20 = arith.subi %18, %19 : i64
        %21 = ascscalar.get_block_idx
        %22 = arith.muli %21, %20 : i64
        %23 = arith.addi %22, %20 : i64
        %24 = arith.cmpi sgt, %23, %6 : i64
        %25 = arith.select %24, %6, %23 : i64
        %26 = arith.cmpi slt, %21, %9 : i64
        scf.if %26 {
          %27 = cannir.extract_layout %arg3 : !cannir.tensor<<gm, f32>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>
          %28 = cannir.extract_shape %27 : <<(-1)>, <(1)>> -> <(-1)>
          %29 = cannir.extract_dim %28[0] : !cannir.shape<(-1)>
          %30 = arith.muli %29, %c4_i64 : i64
          %31 = arith.cmpi eq, %30, %c0_i64 : i64
          %32 = arith.cmpi ne, %30, %c0_i64 : i64
          %33 = arith.extui %32 : i1 to i64
          %34 = arith.select %31, %c4_i64, %30 : i64
          "ascvec.copy_gm2ub_plan"(%alloc_89, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %33, %34, %c0_i64, %34, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 185376 : i64, ub_capacity = 384 : i64}> : (memref<96xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          %35 = cannir.extract_layout %arg4 : !cannir.tensor<<gm, f32>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>
          %36 = cannir.extract_shape %35 : <<(-1, 128)>, <(128, 1)>> -> <(-1, 128)>
          %37 = cannir.extract_dim %36[0] : !cannir.shape<(-1, 128)>
          %38 = arith.muli %37, %c512_i64 : i64
          %39 = arith.cmpi eq, %38, %c0_i64 : i64
          %40 = arith.cmpi ne, %38, %c0_i64 : i64
          %41 = arith.extui %40 : i1 to i64
          %42 = arith.select %39, %c4_i64, %38 : i64
          ascvec.get_buf mte2, %c16_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%alloc_88, %alloc_3, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %41, %42, %c0_i64, %42, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 136224 : i64, ub_capacity = 49152 : i64}> : (memref<12288xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c16_i64, 0 {cannir.buffer.generated_sync}
          %43:3 = scf.for %arg16 = %22 to %25 step %c1_i64 iter_args(%arg17 = %c0_i64, %arg18 = %c0_i64, %arg19 = %c0_i64) -> (i64, i64, i64)  : i64 {
            %44 = arith.remsi %arg16, %c2_i64 : i64
            %45 = arith.cmpi ne, %44, %c0_i64 : i64
            %46 = arith.cmpi slt, %44, %c0_i64 : i64
            %47 = arith.andi %45, %46 : i1
            %48 = arith.divsi %arg16, %c2_i64 : i64
            %49 = arith.extui %47 : i1 to i64
            %50 = arith.subi %48, %49 : i64
            %51 = arith.muli %50, %c2_i64 : i64
            %52 = arith.subi %arg16, %51 : i64
            %53 = arith.maxsi %2, %c1_i64 : i64
            %54 = arith.remsi %50, %53 : i64
            %55 = arith.cmpi slt, %54, %c0_i64 : i64
            %56 = arith.divsi %50, %53 : i64
            %57 = arith.extui %55 : i1 to i64
            %58 = arith.subi %56, %57 : i64
            %59 = arith.maxsi %arg14, %c1_i64 : i64
            %60 = arith.remsi %58, %59 : i64
            %61 = arith.cmpi slt, %60, %c0_i64 : i64
            %62 = arith.select %61, %59, %c0_i64 : i64
            %63 = arith.addi %60, %62 : i64
            %64 = arith.select %55, %53, %c0_i64 : i64
            %65 = arith.addi %54, %64 : i64
            %66 = arith.muli %63, %arg15 : i64
            %67 = arith.addi %66, %arg15 : i64
            %68 = arith.cmpi eq, %arg13, %c1_i64 : i64
            %69 = arith.select %68, %c0_i64, %63 : i64
            %70:2 = scf.if %68 -> (i64, i64) {
              %222 = arith.index_cast %63 : i64 to index
              %223 = memref.load %alloc_9[%222] : memref<1xi32>
              %224 = arith.addi %63, %c1_i64 : i64
              %225 = arith.index_cast %224 : i64 to index
              %226 = memref.load %alloc_9[%225] : memref<1xi32>
              %227 = arith.extsi %223 : i32 to i64
              %228 = arith.extsi %226 : i32 to i64
              scf.yield %227, %228 : i64, i64
            } else {
              scf.yield %66, %67 : i64, i64
            }
            %71 = arith.index_cast %63 : i64 to index
            %72 = memref.load %alloc_8[%71] : memref<1xi32>
            %73 = arith.cmpi sgt, %72, %c0_i32 : i32
            %74 = arith.extsi %72 : i32 to i64
            %75 = arith.addi %70#0, %74 : i64
            %76 = arith.subi %75, %c1_i64 : i64
            %77 = arith.select %73, %76, %70#0 : i64
            %78 = arith.index_cast %77 : i64 to index
            %79 = memref.load %alloc_7[%78] : memref<1xi32>
            %80 = arith.extsi %79 : i32 to i64
            %81 = arith.cmpi eq, %arg17, %c0_i64 : i64
            %82 = ascscalar.memref_address %alloc_63 : memref<8192xf32, 1>
            %83 = ascscalar.memref_address %alloc_64 : memref<8192xf32, 1>
            %84 = arith.select %81, %82, %83 : i64
            %85 = ascscalar.bind_local_ptr %84 : memref<8192xf32, 1>
            %86 = arith.cmpi ne, %arg17, %c0_i64 : i64
            %87 = arith.extui %86 : i1 to i64
            %88 = arith.addi %arg17, %c1_i64 overflow<nsw> : i64
            %89 = arith.remui %88, %c2_i64 : i64
            %90 = arith.minsi %52, %c2_i64 : i64
            %91 = arith.muli %90, %c64_i64 : i64
            %92 = arith.subi %c128_i64, %91 : i64
            %93 = arith.minsi %92, %c64_i64 : i64
            %94 = arith.cmpi sle, %52, %c2_i64 : i64
            %95 = arith.select %94, %93, %c0_i64 : i64
            %96 = cannir.extract_physical_layout %arg6 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
            %97 = cannir.extract_stride %96 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
            %98 = cannir.extract_dim %97[0] : !cannir.stride<(-1, 16384, 128, 1)>
            %99 = arith.muli %80, %98 : i64
            %100 = arith.muli %99, %c4_i64 : i64
            %101 = arith.muli %65, %c65536_i64 : i64
            %102 = arith.addi %100, %101 : i64
            %103 = arith.muli %52, %c32768_i64 : i64
            %104 = arith.addi %102, %103 : i64
            %105 = arith.muli %95, %c512_i64 : i64
            %106 = arith.cmpi eq, %105, %c0_i64 : i64
            %107 = arith.cmpi ne, %105, %c0_i64 : i64
            %108 = arith.extui %107 : i1 to i64
            %109 = arith.select %106, %c4_i64, %105 : i64
            %110 = ascscalar.bind_local_ptr %84 : memref<1xf32, 1>
            ascvec.get_buf mte2, %87, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%110, %alloc_5, %c0_i32, %c0_i32, %c0_i64, %104, %108, %109, %c0_i64, %109, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %87, 0 {cannir.buffer.generated_sync}
            %111 = arith.select %68, %70#0, %c0_i64 : i64
            %112 = arith.addi %2, %65 : i64
            %113 = arith.muli %2, %c2_i64 : i64
            %114 = arith.addi %113, %65 : i64
            %115 = arith.cmpi eq, %arg19, %c0_i64 : i64
            %116 = ascscalar.memref_address %alloc_80 : memref<128xbf16, 1>
            %117 = ascscalar.memref_address %alloc_81 : memref<128xbf16, 1>
            %118 = arith.select %115, %116, %117 : i64
            %119 = arith.select %115, %c8_i64, %c9_i64 : i64
            %120 = arith.addi %arg19, %c1_i64 : i64
            %121 = arith.remui %120, %c2_i64 : i64
            %122 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
            %123 = cannir.extract_shape %122 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 128)>
            %124 = cannir.extract_dim %123[2] : !cannir.shape<(-1, -1, -1, 128)>
            %125 = arith.minsi %111, %124 : i64
            %126 = arith.subi %124, %125 : i64
            %127 = arith.minsi %126, %c1_i64 : i64
            %128 = arith.cmpi sle, %111, %124 : i64
            %129 = arith.select %128, %127, %c0_i64 : i64
            %130 = cannir.extract_physical_layout %arg0 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
            %131 = cannir.extract_stride %130 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
            %132 = cannir.extract_dim %131[2] : !cannir.stride<(-1, -1, -1, 1)>
            %133 = arith.muli %132, %c2_i64 : i64
            %134 = cannir.extract_dim %131[0] : !cannir.stride<(-1, -1, -1, 1)>
            %135 = arith.muli %69, %134 : i64
            %136 = arith.muli %135, %c2_i64 : i64
            %137 = cannir.extract_dim %131[1] : !cannir.stride<(-1, -1, -1, 1)>
            %138 = arith.muli %65, %137 : i64
            %139 = arith.muli %138, %c2_i64 : i64
            %140 = arith.addi %136, %139 : i64
            %141 = arith.muli %111, %132 : i64
            %142 = arith.muli %141, %c2_i64 : i64
            %143 = arith.addi %140, %142 : i64
            %144 = ascscalar.bind_local_ptr %118 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %119, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%144, %alloc, %c0_i32, %c0_i32, %c0_i64, %143, %129, %c256_i64, %133, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %119, 0 {cannir.buffer.generated_sync}
            %145 = ascscalar.memref_address %alloc_82 : memref<128xbf16, 1>
            %146 = ascscalar.memref_address %alloc_83 : memref<128xbf16, 1>
            %147 = arith.select %115, %145, %146 : i64
            %148 = arith.select %115, %c10_i64, %c11_i64 : i64
            %149 = arith.muli %112, %137 : i64
            %150 = arith.muli %149, %c2_i64 : i64
            %151 = arith.addi %136, %150 : i64
            %152 = arith.addi %151, %142 : i64
            %153 = ascscalar.bind_local_ptr %147 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %148, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%153, %alloc, %c0_i32, %c0_i32, %c0_i64, %152, %129, %c256_i64, %133, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %148, 0 {cannir.buffer.generated_sync}
            %154 = ascscalar.memref_address %alloc_84 : memref<128xbf16, 1>
            %155 = ascscalar.memref_address %alloc_85 : memref<128xbf16, 1>
            %156 = arith.select %115, %154, %155 : i64
            %157 = arith.select %115, %c12_i64, %c13_i64 : i64
            %158 = cannir.extract_dim %1[2] : !cannir.shape<(-1, -1, -1, 128)>
            %159 = arith.minsi %111, %158 : i64
            %160 = arith.subi %158, %159 : i64
            %161 = arith.minsi %160, %c1_i64 : i64
            %162 = arith.cmpi sle, %111, %158 : i64
            %163 = arith.select %162, %161, %c0_i64 : i64
            %164 = cannir.extract_physical_layout %arg1 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
            %165 = cannir.extract_stride %164 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
            %166 = cannir.extract_dim %165[2] : !cannir.stride<(-1, -1, -1, 1)>
            %167 = arith.muli %166, %c2_i64 : i64
            %168 = cannir.extract_dim %165[0] : !cannir.stride<(-1, -1, -1, 1)>
            %169 = arith.muli %69, %168 : i64
            %170 = arith.muli %169, %c2_i64 : i64
            %171 = cannir.extract_dim %165[1] : !cannir.stride<(-1, -1, -1, 1)>
            %172 = arith.muli %65, %171 : i64
            %173 = arith.muli %172, %c2_i64 : i64
            %174 = arith.addi %170, %173 : i64
            %175 = arith.muli %111, %166 : i64
            %176 = arith.muli %175, %c2_i64 : i64
            %177 = arith.addi %174, %176 : i64
            %178 = ascscalar.bind_local_ptr %156 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %157, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%178, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %177, %163, %c256_i64, %167, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %157, 0 {cannir.buffer.generated_sync}
            %179 = ascscalar.memref_address %alloc_86 : memref<64xbf16, 1>
            %180 = ascscalar.memref_address %alloc_87 : memref<64xbf16, 1>
            %181 = arith.select %115, %179, %180 : i64
            %182 = arith.select %115, %c14_i64, %c15_i64 : i64
            %183 = cannir.extract_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>
            %184 = cannir.extract_shape %183 : <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
            %185 = cannir.extract_dim %184[2] : !cannir.shape<(-1, -1, -1, 1)>
            %186 = arith.minsi %111, %185 : i64
            %187 = arith.subi %185, %186 : i64
            %188 = arith.minsi %187, %c1_i64 : i64
            %189 = arith.cmpi sle, %111, %185 : i64
            %190 = arith.select %189, %188, %c0_i64 : i64
            %191 = cannir.extract_physical_layout %arg2 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>>, identity>
            %192 = cannir.extract_stride %191 : <<(-1, -1, -1, 1)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
            %193 = cannir.extract_dim %192[2] : !cannir.stride<(-1, -1, -1, 1)>
            %194 = arith.muli %193, %c2_i64 : i64
            %195 = cannir.extract_dim %192[0] : !cannir.stride<(-1, -1, -1, 1)>
            %196 = arith.muli %69, %195 : i64
            %197 = arith.muli %196, %c2_i64 : i64
            %198 = cannir.extract_dim %192[1] : !cannir.stride<(-1, -1, -1, 1)>
            %199 = arith.muli %65, %198 : i64
            %200 = arith.muli %199, %c2_i64 : i64
            %201 = arith.addi %197, %200 : i64
            %202 = arith.muli %111, %193 : i64
            %203 = arith.muli %202, %c2_i64 : i64
            %204 = arith.addi %201, %203 : i64
            %205 = ascscalar.bind_local_ptr %181 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %182, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%205, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %204, %190, %c2_i64, %194, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %182, 0 {cannir.buffer.generated_sync}
            %206 = ascscalar.memref_address %alloc_73 : memref<64xbf16, 1>
            %207 = ascscalar.memref_address %alloc_74 : memref<64xbf16, 1>
            %208 = arith.select %115, %206, %207 : i64
            %209 = arith.select %115, %c4_i64, %c5_i64 : i64
            %210 = arith.muli %114, %137 : i64
            %211 = arith.muli %210, %c2_i64 : i64
            %212 = arith.addi %136, %211 : i64
            %213 = arith.addi %212, %142 : i64
            %214 = arith.muli %52, %c128_i64 : i64
            %215 = arith.addi %213, %214 : i64
            %216 = arith.muli %95, %c2_i64 : i64
            %217 = arith.cmpi eq, %216, %c0_i64 : i64
            %218 = arith.select %217, %c0_i64, %129 : i64
            %219 = arith.select %217, %c2_i64, %216 : i64
            %220 = ascscalar.bind_local_ptr %208 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %209, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%220, %alloc, %c0_i32, %c0_i32, %c0_i64, %215, %218, %219, %133, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %209, 0 {cannir.buffer.generated_sync}
            %221:2 = scf.for %arg20 = %70#0 to %70#1 step %c1_i64 iter_args(%arg21 = %arg18, %arg22 = %121) -> (i64, i64)  : i64 {
              %222 = arith.subi %arg20, %70#0 : i64
              %223 = arith.select %68, %arg20, %222 : i64
              %224 = arith.cmpi eq, %arg21, %c0_i64 : i64
              %225 = arith.select %224, %116, %117 : i64
              %226 = ascscalar.bind_local_ptr %225 : memref<128xbf16, 1>
              %227 = arith.select %224, %c8_i64, %c9_i64 : i64
              %228 = arith.addi %arg21, %c1_i64 overflow<nsw> : i64
              %229 = arith.remui %228, %c2_i64 : i64
              %230 = arith.select %224, %145, %146 : i64
              %231 = ascscalar.bind_local_ptr %230 : memref<128xbf16, 1>
              %232 = arith.select %224, %c10_i64, %c11_i64 : i64
              ascvec.get_buf v, %227, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %232, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%226, %231, %alloc_66, %alloc_65, %alloc_77 : memref<128xbf16, 1>, memref<128xbf16, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<4xf32, 1>) {
              ^bb0(%arg23: memref<128xbf16, 1>, %arg24: memref<128xbf16, 1>, %arg25: memref<128xf32, 1>, %arg26: memref<128xf32, 1>, %arg27: memref<4xf32, 1>):
                %c3 = arith.constant 3 : index
                %c2_90 = arith.constant 2 : index
                %c1 = arith.constant 1 : index
                %c64 = arith.constant 64 : index
                %c0_91 = arith.constant 0 : index
                %c64_i32_92 = arith.constant 64 : i32
                %c0_93 = arith.constant 0 : index
                %c1_94 = arith.constant 1 : index
                scf.for %arg28 = %c0_93 to %c1_94 step %c1_94 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_92 : !ascvec.mask
                  %302 = ascvec.loadalign_unpack %arg23[%c0_91] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
                  %303 = ascvec.loadalign_unpack %arg23[%c64] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
                  %304 = ascvec.loadalign_unpack %arg24[%c0_91] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
                  %305 = ascvec.loadalign_unpack %arg24[%c64] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
                  %306 = ascvec.cast %302, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %307 = ascvec.cast %303, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %308 = ascvec.cast %304, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %309 = ascvec.cast %305, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %310 = ascvec.mul %306, %306, %mask : <f32>
                  %311 = ascvec.mul %307, %307, %mask : <f32>
                  %312 = ascvec.mul %308, %308, %mask : <f32>
                  %313 = ascvec.mul %309, %309, %mask : <f32>
                  %314 = ascvec.reduce_sum %310, %mask : <f32>
                  %315 = ascvec.reduce_sum %311, %mask : <f32>
                  %316 = ascvec.reduce_sum %312, %mask : <f32>
                  %317 = ascvec.reduce_sum %313, %mask : <f32>
                  ascvec.storealign %arg25[%c0_91], %306, %mask : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign %arg25[%c64], %307, %mask : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign %arg26[%c0_91], %308, %mask : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign %arg26[%c64], %309, %mask : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign_1st %arg27[%c0_91], %314 : memref<4xf32, 1>, index, <f32>
                  ascvec.storealign_1st %arg27[%c1], %315 : memref<4xf32, 1>, index, <f32>
                  ascvec.storealign_1st %arg27[%c2_90], %316 : memref<4xf32, 1>, index, <f32>
                  ascvec.storealign_1st %arg27[%c3], %317 : memref<4xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %232, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %227, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_77 : memref<4xf32, 1>) {
              ^bb0(%arg23: memref<4xf32, 1>):
                %c3 = arith.constant 3 : index
                %c2_90 = arith.constant 2 : index
                %c1 = arith.constant 1 : index
                %cst = arith.constant 9.99999997E-7 : f32
                %c0_91 = arith.constant 0 : index
                %c1_i32_92 = arith.constant 1 : i32
                %c0_93 = arith.constant 0 : index
                %c1_94 = arith.constant 1 : index
                scf.for %arg24 = %c0_93 to %c1_94 step %c1_94 {
                  %mask, %remaining_out = ascvec.update_mask %c1_i32_92 : !ascvec.mask
                  %302 = ascvec.loadalign_brc %arg23[%c0_91] : memref<4xf32, 1>, index -> <f32>
                  %303 = ascvec.loadalign_brc %arg23[%c1] : memref<4xf32, 1>, index -> <f32>
                  %304 = ascvec.loadalign_brc %arg23[%c2_90] : memref<4xf32, 1>, index -> <f32>
                  %305 = ascvec.loadalign_brc %arg23[%c3] : memref<4xf32, 1>, index -> <f32>
                  %306 = ascvec.add %302, %303, %mask : <f32>
                  %307 = ascvec.add %304, %305, %mask : <f32>
                  %308 = ascvec.add_scalar %306, %cst, %mask : <f32>, f32
                  %309 = ascvec.add_scalar %307, %cst, %mask : <f32>, f32
                  %310 = ascvec.sqrt %308, %mask : <f32>
                  %311 = ascvec.sqrt %309, %mask : <f32>
                  ascvec.storealign_1st %arg23[%c0_91], %310 : memref<4xf32, 1>, index, <f32>
                  ascvec.storealign_1st %arg23[%c1], %311 : memref<4xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.vf(%alloc_77, %alloc_66, %alloc_65, %arg11 : memref<4xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, f32) {
              ^bb0(%arg23: memref<4xf32, 1>, %arg24: memref<128xf32, 1>, %arg25: memref<128xf32, 1>, %arg26: f32):
                %c64 = arith.constant 64 : index
                %c1 = arith.constant 1 : index
                %c0_90 = arith.constant 0 : index
                %c64_i32_91 = arith.constant 64 : i32
                %c0_92 = arith.constant 0 : index
                %c1_93 = arith.constant 1 : index
                scf.for %arg27 = %c0_92 to %c1_93 step %c1_93 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
                  %302 = ascvec.loadalign_brc %arg23[%c0_90] : memref<4xf32, 1>, index -> <f32>
                  %303 = ascvec.loadalign_brc %arg23[%c1] : memref<4xf32, 1>, index -> <f32>
                  %304 = ascvec.duplicate %302, %mask : <f32> -> <f32>
                  %305 = ascvec.duplicate %303, %mask : <f32> -> <f32>
                  %306 = ascvec.loadalign %arg24[%c0_90] : memref<128xf32, 1>, index -> <f32>
                  %307 = ascvec.loadalign %arg24[%c64] : memref<128xf32, 1>, index -> <f32>
                  %308 = ascvec.loadalign %arg25[%c0_90] : memref<128xf32, 1>, index -> <f32>
                  %309 = ascvec.loadalign %arg25[%c64] : memref<128xf32, 1>, index -> <f32>
                  %310 = ascvec.div %306, %304, %mask : <f32>
                  %311 = ascvec.div %307, %304, %mask : <f32>
                  %312 = ascvec.div %308, %305, %mask : <f32>
                  %313 = ascvec.div %309, %305, %mask : <f32>
                  %314 = ascvec.mul_scalar %310, %arg26, %mask : <f32>, f32
                  %315 = ascvec.mul_scalar %311, %arg26, %mask : <f32>, f32
                  ascvec.storealign %arg24[%c0_90], %314, %mask : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign %arg24[%c64], %315, %mask : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign %arg25[%c0_90], %312, %mask : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign %arg25[%c64], %313, %mask : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              %233 = arith.select %224, %154, %155 : i64
              %234 = ascscalar.bind_local_ptr %233 : memref<128xbf16, 1>
              %235 = arith.select %224, %c12_i64, %c13_i64 : i64
              %236 = arith.select %224, %179, %180 : i64
              %237 = ascscalar.bind_local_ptr %236 : memref<64xbf16, 1>
              %238 = arith.select %224, %c14_i64, %c15_i64 : i64
              %239 = arith.select %224, %206, %207 : i64
              %240 = ascscalar.bind_local_ptr %239 : memref<64xbf16, 1>
              %241 = arith.select %224, %c4_i64, %c5_i64 : i64
              %242 = arith.trunci %65 : i64 to i32
              %243 = arith.index_cast %242 : i32 to index
              %244 = arith.muli %242, %c128_i32 : i32
              %245 = arith.index_cast %244 : i32 to index
              %246 = arith.addi %244, %c64_i32 : i32
              %247 = arith.index_cast %246 : i32 to index
              ascvec.get_buf v, %235, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %238, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %241, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_89, %243, %234, %alloc_88, %245, %247, %arg12, %alloc_67, %237, %alloc_69, %240, %alloc_68 : memref<96xf32, 1>, index, memref<128xbf16, 1>, memref<12288xf32, 1>, index, index, f32, memref<128xf32, 1>, memref<64xbf16, 1>, memref<64xf32, 1>, memref<64xbf16, 1>, memref<64xf32, 1>) {
              ^bb0(%arg23: memref<96xf32, 1>, %arg24: index, %arg25: memref<128xbf16, 1>, %arg26: memref<12288xf32, 1>, %arg27: index, %arg28: index, %arg29: f32, %arg30: memref<128xf32, 1>, %arg31: memref<64xbf16, 1>, %arg32: memref<64xf32, 1>, %arg33: memref<64xbf16, 1>, %arg34: memref<64xf32, 1>):
                %c64 = arith.constant 64 : index
                %c0_90 = arith.constant 0 : index
                %cst = arith.constant 1.000000e+00 : f32
                %c64_i32_91 = arith.constant 64 : i32
                %c0_92 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg35 = %c0_92 to %c1 step %c1 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
                  %302 = ascvec.duplicate_scalar_nomask %cst : f32 -> <f32>
                  %303 = ascvec.loadalign_brc %arg23[%arg24] : memref<96xf32, 1>, index -> <f32>
                  %304 = ascvec.exp %303, %mask : <f32>
                  %305 = ascvec.neg %304, %mask : <f32>
                  %306 = ascvec.loadalign_unpack %arg25[%c0_90] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
                  %307 = ascvec.loadalign_unpack %arg25[%c64] {mode = "b16_to_b32"} : memref<128xbf16, 1>, index -> <bf16>
                  %308 = ascvec.cast %306, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %309 = ascvec.cast %307, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %310 = ascvec.loadalign %arg26[%arg27] : memref<12288xf32, 1>, index -> <f32>
                  %311 = ascvec.loadalign %arg26[%arg28] : memref<12288xf32, 1>, index -> <f32>
                  %312 = ascvec.add %308, %310, %mask : <f32>
                  %313 = ascvec.add %309, %311, %mask : <f32>
                  %314 = ascvec.mul %305, %312, %mask : <f32>
                  %315 = ascvec.mul %305, %313, %mask : <f32>
                  %316 = ascvec.exp %314, %mask : <f32>
                  %317 = ascvec.exp %315, %mask : <f32>
                  %318 = ascvec.add_scalar %316, %cst, %mask : <f32>, f32
                  %319 = ascvec.add_scalar %317, %cst, %mask : <f32>, f32
                  %320 = ascvec.div %302, %318, %mask : <f32>
                  %321 = ascvec.div %302, %319, %mask : <f32>
                  %322 = ascvec.mul_scalar %320, %arg29, %mask : <f32>, f32
                  %323 = ascvec.mul_scalar %321, %arg29, %mask : <f32>, f32
                  %324 = ascvec.exp %322, %mask : <f32>
                  %325 = ascvec.exp %323, %mask : <f32>
                  ascvec.storealign %arg30[%c0_90], %324, %mask : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign %arg30[%c64], %325, %mask : memref<128xf32, 1>, index, <f32>
                  %326 = ascvec.loadalign_unpack %arg31[%c0_90] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
                  %327 = ascvec.cast %326, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  %328 = ascvec.neg %327, %mask : <f32>
                  %329 = ascvec.exp %328, %mask : <f32>
                  %330 = ascvec.add_scalar %329, %cst, %mask : <f32>, f32
                  %331 = ascvec.div %302, %330, %mask : <f32>
                  ascvec.storealign %arg32[%c0_90], %331, %mask : memref<64xf32, 1>, index, <f32>
                  %332 = ascvec.loadalign_unpack %arg33[%c0_90] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
                  %333 = ascvec.cast %332, %mask {rounding = 6 : i32} : <bf16> -> <f32>
                  ascvec.storealign %arg34[%c0_90], %333, %mask : memref<64xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %241, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %238, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %235, 0 {cannir.buffer.generated_sync}
              %248 = arith.addi %arg20, %c1_i64 : i64
              %249 = arith.cmpi slt, %248, %70#1 : i64
              %250 = scf.if %249 -> (i64) {
                %302 = arith.subi %248, %70#0 : i64
                %303 = arith.select %68, %248, %302 : i64
                %304 = arith.cmpi eq, %arg22, %c0_i64 : i64
                %305 = arith.select %304, %116, %117 : i64
                %306 = arith.select %304, %c8_i64, %c9_i64 : i64
                %307 = arith.addi %arg22, %c1_i64 overflow<nsw> : i64
                %308 = arith.remui %307, %c2_i64 : i64
                %309 = arith.minsi %303, %124 : i64
                %310 = arith.subi %124, %309 : i64
                %311 = arith.minsi %310, %c1_i64 : i64
                %312 = arith.cmpi sle, %303, %124 : i64
                %313 = arith.select %312, %311, %c0_i64 : i64
                %314 = arith.muli %303, %132 : i64
                %315 = arith.muli %314, %c2_i64 : i64
                %316 = arith.addi %140, %315 : i64
                %317 = ascscalar.bind_local_ptr %305 : memref<1xbf16, 1>
                ascvec.get_buf mte2, %306, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%317, %alloc, %c0_i32, %c0_i32, %c0_i64, %316, %313, %c256_i64, %133, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %306, 0 {cannir.buffer.generated_sync}
                %318 = arith.select %304, %145, %146 : i64
                %319 = arith.select %304, %c10_i64, %c11_i64 : i64
                %320 = arith.addi %151, %315 : i64
                %321 = ascscalar.bind_local_ptr %318 : memref<1xbf16, 1>
                ascvec.get_buf mte2, %319, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%321, %alloc, %c0_i32, %c0_i32, %c0_i64, %320, %313, %c256_i64, %133, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %319, 0 {cannir.buffer.generated_sync}
                %322 = arith.select %304, %154, %155 : i64
                %323 = arith.select %304, %c12_i64, %c13_i64 : i64
                %324 = arith.minsi %303, %158 : i64
                %325 = arith.subi %158, %324 : i64
                %326 = arith.minsi %325, %c1_i64 : i64
                %327 = arith.cmpi sle, %303, %158 : i64
                %328 = arith.select %327, %326, %c0_i64 : i64
                %329 = arith.muli %303, %166 : i64
                %330 = arith.muli %329, %c2_i64 : i64
                %331 = arith.addi %174, %330 : i64
                %332 = ascscalar.bind_local_ptr %322 : memref<1xbf16, 1>
                ascvec.get_buf mte2, %323, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%332, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %331, %328, %c256_i64, %167, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %323, 0 {cannir.buffer.generated_sync}
                %333 = arith.select %304, %179, %180 : i64
                %334 = arith.select %304, %c14_i64, %c15_i64 : i64
                %335 = arith.minsi %303, %185 : i64
                %336 = arith.subi %185, %335 : i64
                %337 = arith.minsi %336, %c1_i64 : i64
                %338 = arith.cmpi sle, %303, %185 : i64
                %339 = arith.select %338, %337, %c0_i64 : i64
                %340 = arith.muli %303, %193 : i64
                %341 = arith.muli %340, %c2_i64 : i64
                %342 = arith.addi %201, %341 : i64
                %343 = ascscalar.bind_local_ptr %333 : memref<1xbf16, 1>
                ascvec.get_buf mte2, %334, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%343, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %342, %339, %c2_i64, %194, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %334, 0 {cannir.buffer.generated_sync}
                %344 = arith.select %304, %206, %207 : i64
                %345 = arith.select %304, %c4_i64, %c5_i64 : i64
                %346 = arith.addi %212, %315 : i64
                %347 = arith.addi %346, %214 : i64
                %348 = arith.select %217, %c0_i64, %313 : i64
                %349 = ascscalar.bind_local_ptr %344 : memref<1xbf16, 1>
                ascvec.get_buf mte2, %345, 0 {cannir.buffer.generated_sync}
                "ascvec.copy_gm2ub_plan"(%349, %alloc, %c0_i32, %c0_i32, %c0_i64, %347, %348, %219, %133, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
                ascvec.release_buf mte2, %345, 0 {cannir.buffer.generated_sync}
                scf.yield %308 : i64
              } else {
                scf.yield %arg22 : i64
              }
              %251 = ascscalar.memref_address %alloc_71 : memref<8192xf32, 1>
              %252 = ascscalar.memref_address %alloc_72 : memref<8192xf32, 1>
              %253 = arith.select %224, %251, %252 : i64
              %254 = ascscalar.bind_local_ptr %253 : memref<8192xf32, 1>
              %255 = arith.select %224, %c2_i64, %c3_i64 : i64
              ascvec.get_buf v, %87, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_67, %alloc_65, %85, %alloc_78 : memref<128xf32, 1>, memref<128xf32, 1>, memref<8192xf32, 1>, memref<64xf32, 1>) {
              ^bb0(%arg23: memref<128xf32, 1>, %arg24: memref<128xf32, 1>, %arg25: memref<8192xf32, 1>, %arg26: memref<64xf32, 1>):
                %c64 = arith.constant 64 : index
                %c1_i32_90 = arith.constant 1 : i32
                %c0_i32_91 = arith.constant 0 : i32
                %c128_i32_92 = arith.constant 128 : i32
                %c0_93 = arith.constant 0 : index
                %c64_i32_94 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_94 : !ascvec.mask
                %302 = ascvec.loadalign %arg23[%c0_93] : memref<128xf32, 1>, index -> <f32>
                %303 = ascvec.loadalign %arg23[%c64] : memref<128xf32, 1>, index -> <f32>
                %304 = ascvec.loadalign %arg24[%c0_93] : memref<128xf32, 1>, index -> <f32>
                %305 = ascvec.loadalign %arg24[%c64] : memref<128xf32, 1>, index -> <f32>
                %306 = ascvec.mul %302, %304, %mask : <f32>
                %307 = ascvec.mul %303, %305, %mask : <f32>
                scf.for %arg27 = %c0_i32_91 to %c64_i32_94 step %c1_i32_90  : i32 {
                  %308 = arith.muli %arg27, %c128_i32_92 overflow<nsw> : i32
                  %309 = arith.addi %308, %c64_i32_94 overflow<nsw> : i32
                  %310 = arith.index_cast %308 : i32 to index
                  %311 = ascvec.loadalign %arg25[%310] : memref<8192xf32, 1>, index -> <f32>
                  %312 = arith.index_cast %309 : i32 to index
                  %313 = ascvec.loadalign %arg25[%312] : memref<8192xf32, 1>, index -> <f32>
                  %314 = ascvec.mul %311, %306, %mask : <f32>
                  %315 = ascvec.madd_two_address %313, %307, %314, %mask : <f32>
                  %316 = ascvec.reduce_sum %315, %mask : <f32>
                  %317 = arith.index_cast %arg27 : i32 to index
                  ascvec.storealign_1st %arg26[%317], %316 : memref<64xf32, 1>, index, <f32>
                } {cannir.loop_unroll = 4 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.vf(%alloc_69, %alloc_78, %alloc_68, %alloc_79 : memref<64xf32, 1>, memref<64xf32, 1>, memref<64xf32, 1>, memref<64xf32, 1>) {
              ^bb0(%arg23: memref<64xf32, 1>, %arg24: memref<64xf32, 1>, %arg25: memref<64xf32, 1>, %arg26: memref<64xf32, 1>):
                %c0_90 = arith.constant 0 : index
                %c64_i32_91 = arith.constant 64 : i32
                %c0_92 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg27 = %c0_92 to %c1 step %c1 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
                  %302 = ascvec.loadalign_brc %arg23[%c0_90] : memref<64xf32, 1>, index -> <f32>
                  %303 = ascvec.loadalign %arg24[%c0_90] : memref<64xf32, 1>, index -> <f32>
                  %304 = ascvec.loadalign %arg25[%c0_90] : memref<64xf32, 1>, index -> <f32>
                  %305 = ascvec.sub %304, %303, %mask : <f32>
                  %306 = ascvec.mul %305, %302, %mask : <f32>
                  ascvec.storealign %arg26[%c0_90], %306, %mask : memref<64xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.get_buf v, %255, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_67, %alloc_65, %alloc_66, %85, %alloc_79, %254, %alloc_70 : memref<128xf32, 1>, memref<128xf32, 1>, memref<128xf32, 1>, memref<8192xf32, 1>, memref<64xf32, 1>, memref<8192xf32, 1>, memref<64xf32, 1>) {
              ^bb0(%arg23: memref<128xf32, 1>, %arg24: memref<128xf32, 1>, %arg25: memref<128xf32, 1>, %arg26: memref<8192xf32, 1>, %arg27: memref<64xf32, 1>, %arg28: memref<8192xf32, 1>, %arg29: memref<64xf32, 1>):
                %c64 = arith.constant 64 : index
                %c1_i32_90 = arith.constant 1 : i32
                %c0_i32_91 = arith.constant 0 : i32
                %c128_i32_92 = arith.constant 128 : i32
                %c0_93 = arith.constant 0 : index
                %c64_i32_94 = arith.constant 64 : i32
                %mask, %remaining_out = ascvec.update_mask %c64_i32_94 : !ascvec.mask
                %302 = ascvec.loadalign %arg23[%c0_93] : memref<128xf32, 1>, index -> <f32>
                %303 = ascvec.loadalign %arg23[%c64] : memref<128xf32, 1>, index -> <f32>
                %304 = ascvec.loadalign %arg24[%c0_93] : memref<128xf32, 1>, index -> <f32>
                %305 = ascvec.loadalign %arg24[%c64] : memref<128xf32, 1>, index -> <f32>
                %306 = ascvec.loadalign %arg25[%c0_93] : memref<128xf32, 1>, index -> <f32>
                %307 = ascvec.loadalign %arg25[%c64] : memref<128xf32, 1>, index -> <f32>
                scf.for %arg30 = %c0_i32_91 to %c64_i32_94 step %c1_i32_90  : i32 {
                  %308 = arith.muli %arg30, %c128_i32_92 overflow<nsw> : i32
                  %309 = arith.addi %308, %c64_i32_94 overflow<nsw> : i32
                  %310 = arith.index_cast %308 : i32 to index
                  %311 = ascvec.loadalign %arg26[%310] : memref<8192xf32, 1>, index -> <f32>
                  %312 = arith.index_cast %309 : i32 to index
                  %313 = ascvec.loadalign %arg26[%312] : memref<8192xf32, 1>, index -> <f32>
                  %314 = arith.index_cast %arg30 : i32 to index
                  %315 = ascvec.loadalign_brc %arg27[%314] : memref<64xf32, 1>, index -> <f32>
                  %316 = ascvec.mul %315, %304, %mask : <f32>
                  %317 = ascvec.mul %315, %305, %mask : <f32>
                  %318 = ascvec.madd_two_address %311, %302, %316, %mask : <f32>
                  %319 = ascvec.madd_two_address %313, %303, %317, %mask : <f32>
                  ascvec.storealign %arg26[%310], %318, %mask : memref<8192xf32, 1>, index, <f32>
                  ascvec.storealign %arg26[%312], %319, %mask : memref<8192xf32, 1>, index, <f32>
                  ascvec.storealign %arg28[%310], %318, %mask : memref<8192xf32, 1>, index, <f32>
                  ascvec.storealign %arg28[%312], %319, %mask : memref<8192xf32, 1>, index, <f32>
                  %320 = ascvec.mul %318, %306, %mask : <f32>
                  %321 = ascvec.madd_two_address %319, %307, %320, %mask : <f32>
                  %322 = ascvec.reduce_sum %321, %mask : <f32>
                  ascvec.storealign_1st %arg29[%314], %322 : memref<64xf32, 1>, index, <f32>
                } {cannir.loop_unroll = 1 : i64}
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %255, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %87, 0 {cannir.buffer.generated_sync}
              %256 = ascscalar.memref_address %alloc_75 : memref<64xbf16, 1>
              %257 = ascscalar.memref_address %alloc_76 : memref<64xbf16, 1>
              %258 = arith.select %224, %256, %257 : i64
              %259 = ascscalar.bind_local_ptr %258 : memref<64xbf16, 1>
              %260 = arith.select %224, %c6_i64, %c7_i64 : i64
              ascvec.get_buf v, %260, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_70, %259 : memref<64xf32, 1>, memref<64xbf16, 1>) {
              ^bb0(%arg23: memref<64xf32, 1>, %arg24: memref<64xbf16, 1>):
                %c0_90 = arith.constant 0 : index
                %c64_i32_91 = arith.constant 64 : i32
                %c0_92 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg25 = %c0_92 to %c1 step %c1 {
                  %mask, %remaining_out = ascvec.update_mask %c64_i32_91 : !ascvec.mask
                  %302 = ascvec.loadalign %arg23[%c0_90] : memref<64xf32, 1>, index -> <f32>
                  %303 = ascvec.cast %302, %mask : <f32> -> <bf16>
                  ascvec.storealign_pack %arg24[%c0_90], %303, %mask {mode = "b32_to_b16"} : memref<64xbf16, 1>, index, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %260, 0 {cannir.buffer.generated_sync}
              %261 = cannir.extract_physical_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %262 = cannir.extract_stride %261 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
              %263 = cannir.extract_dim %262[0] : !cannir.stride<(-1, -1, -1, 1)>
              %264 = arith.muli %69, %263 : i64
              %265 = cannir.extract_dim %262[1] : !cannir.stride<(-1, -1, -1, 1)>
              %266 = arith.muli %65, %265 : i64
              %267 = arith.addi %264, %266 : i64
              %268 = cannir.extract_dim %262[2] : !cannir.stride<(-1, -1, -1, 1)>
              %269 = arith.muli %223, %268 : i64
              %270 = arith.addi %267, %269 : i64
              %271 = arith.muli %52, %c64_i64 : i64
              %272 = arith.addi %270, %271 : i64
              %273 = arith.muli %272, %c16_i64 : i64
              %274 = ascscalar.memref_address %alloc_4 : memref<1xbf16>
              %275 = arith.divui %273, %c8_i64 : i64
              %276 = arith.addi %274, %275 : i64
              %277 = ascscalar.bind_gm_ptr %276 : memref<1xbf16>
              %278 = ascscalar.bind_local_ptr %258 : memref<1xbf16, 1>
              %279 = cannir.extract_layout %arg5 : !cannir.tensor<<gm, bf16>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>>, identity>
              %280 = cannir.extract_stride %279 : <<(-1, -1, -1, 128)>, <(-1, -1, -1, 1)>> -> <(-1, -1, -1, 1)>
              %281 = cannir.extract_dim %280[2] : !cannir.stride<(-1, -1, -1, 1)>
              %282 = arith.index_cast %281 : i64 to index
              %283 = arith.muli %282, %c2 : index
              %284 = arith.index_cast %283 : index to i64
              ascvec.get_buf mte3, %260, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %277[%c0], %278[%c0], %c1_i32, %c128_i32, %284, %c128_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
              ascvec.release_buf mte3, %260, 0 {cannir.buffer.generated_sync}
              %285 = arith.index_cast %arg20 : i64 to index
              %286 = memref.load %alloc_7[%285] : memref<1xi32>
              %287 = arith.extsi %286 : i32 to i64
              %288 = cannir.extract_physical_layout %arg7 : !cannir.tensor<<gm, f32>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>>, identity>
              %289 = cannir.extract_stride %288 : <<(-1, -1, 128, 128)>, <(-1, 16384, 128, 1)>> -> <(-1, 16384, 128, 1)>
              %290 = cannir.extract_dim %289[0] : !cannir.stride<(-1, 16384, 128, 1)>
              %291 = arith.muli %287, %290 : i64
              %292 = arith.muli %65, %c16384_i64 : i64
              %293 = arith.addi %291, %292 : i64
              %294 = arith.muli %52, %c8192_i64 : i64
              %295 = arith.addi %293, %294 : i64
              %296 = arith.muli %295, %c32_i64 : i64
              %297 = ascscalar.memref_address %alloc_6 : memref<1xf32>
              %298 = arith.divui %296, %c8_i64 : i64
              %299 = arith.addi %297, %298 : i64
              %300 = ascscalar.bind_gm_ptr %299 : memref<1xf32>
              %301 = ascscalar.bind_local_ptr %253 : memref<1xf32, 1>
              ascvec.get_buf mte3, %255, 0 {cannir.buffer.generated_sync}
              ascvec.copy_ub2gm %300[%c0], %301[%c0], %c1_i32, %c32768_i32, %c0_i64, %c32768_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
              ascvec.release_buf mte3, %255, 0 {cannir.buffer.generated_sync}
              scf.yield %229, %250 : i64, i64
            }
            scf.yield %89, %221#0, %221#1 : i64, i64, i64
          }
        }
      }
    }
    return
  }
}
