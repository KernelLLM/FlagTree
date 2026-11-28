module {
  func.func @run_09e7a0edf895671a922966f97f9c145a(%arg0: !cannir.tensor<<gm, ui8>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(1024, 1024)>, <(1024, 1)>>, <<(1024, 1024)>, <(1024, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(384, 1024)>, <(1024, 1)>>, <<(384, 1024)>, <(1024, 1)>>, identity>, %arg3: !cannir.tensor<<gm, ui8>, <<(576, 1024)>, <(1024, 1)>>, <<(576, 1024)>, <(1024, 1)>>, identity>, %arg4: !cannir.tensor<<gm, bf16>, <<(2, 128, 512)>, <(65536, 512, 1)>>, <<(2, 128, 512)>, <(65536, 512, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(1, 1024)>, <(1024, 1)>>, <<(1, 1024)>, <(1024, 1)>>, identity>, %arg7: !cannir.tensor<<gm, f32>, <<(1, 384)>, <(384, 1)>>, <<(1, 384)>, <(384, 1)>>, identity>, %arg8: !cannir.tensor<<gm, f32>, <<(1, 576)>, <(576, 1)>>, <<(1, 576)>, <(576, 1)>>, identity>, %arg9: !cannir.tensor<<gm, f32>, <<(1, 1024)>, <(1024, 1)>>, <<(1, 1024)>, <(1024, 1)>>, identity>, %arg10: !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, %arg11: !cannir.tensor<<gm, ui8>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, identity>, %arg12: !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg13: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg14: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg15: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg16: !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg17: !cannir.tensor<<gm, f32>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>, %arg18: !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg19: !cannir.tensor<<gm, f32>, <<(-1, 512)>, <(512, 1)>>, <<(-1, 512)>, <(512, 1)>>, identity>, %arg20: !cannir.tensor<<gm, ui8>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg21: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg22: !cannir.tensor<<gm, f32>, <<(-1, 384)>, <(384, 1)>>, <<(-1, 384)>, <(384, 1)>>, identity>, %arg23: !cannir.tensor<<gm, bf16>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, %arg24: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg25: !cannir.tensor<<gm, bf16>, <<(-1, 512)>, <(512, 1)>>, <<(-1, 512)>, <(512, 1)>>, identity>, %arg26: !cannir.tensor<<gm, bf16>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>, %arg27: !cannir.tensor<<gm, ui8>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>, %arg28: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg29: f32, %arg30: f32, %arg31: f32, %arg32: !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg33: i64) attributes {cannbotdsl.contract_digest = "09e7a0edf895671a922966f97f9c145abfe7bb24e71b13497e311e82202f04ff", cannbotdsl.public_slot_count = 34 : i64, cannbotdsl.slot_dtype_bits = [8, 8, 8, 8, 16, 32, 32, 32, 32, 32, 32, 8, 64, 32, 16, 16, 32, 32, 32, 32, 8, 32, 32, 16, 16, 16, 16, 8, 32, 32, 32, 32, 32, 64], cannbotdsl.slot_dtype_codes = [1, 1, 1, 1, 4, 2, 2, 2, 2, 2, 2, 1, 0, 2, 4, 4, 2, 2, 2, 2, 1, 2, 2, 4, 4, 4, 4, 1, 2, 2, 2, 2, 2, 0], cannbotdsl.slot_dtype_lanes = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_flags = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], cannbotdsl.slot_kinds = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2], cannbotdsl.slot_type_digests = ["d9b145762ae9f19227aa0f6759e516bd94b1706216476726f7c1b71348db2dea", "bedf2e36ab0a1a49c1ea30df81c05f58b1c98ba054d5a1af6a8b027d66cd4a61", "3df087ffb692c7d1e9711f95375979f9ce8da0d426454be20f38297d71212e86", "88e8482478baa6fea881c5f22be2c680cc2a94f74d77dafa5b86b0f7bc10e207", "acf211e93f52c0b0a29534f094bc8c996dd6988e8cfde7cbbc30b19ec2937547", "b134b54071cdca5322ca9f5d39ee9881e5a8f90b566373e7bf45d235577c8f89", "2bcea24f572a85cc6c0207a9343668b3d196cc21c575daacebe2352c75d33ec0", "041f49b0678f0f41ea1ebfa6b4304910d474ad992b40d426c78628a899d0b2b4", "76cd705669f68d7c2a87ab548e70d30e65dc51e7c617ee8c4390186742b2cd83", "2bcea24f572a85cc6c0207a9343668b3d196cc21c575daacebe2352c75d33ec0", "7b962932eb0067670c47bf914f64cc35656e412a09850be9b2bc297c5b918fc3", "36b680395402605410081e50278e4d329700d06f4e63995423f1d8b39c77c706", "c67886b9b15d5f3da963e66099f8fbcace72279ff7c2dd40a3bfd84a4bbf6543", "b134b54071cdca5322ca9f5d39ee9881e5a8f90b566373e7bf45d235577c8f89", "d58d6c5102be0390b5a14c2a1f764c9e4e2eb551033443898307364ea281fe03", "d58d6c5102be0390b5a14c2a1f764c9e4e2eb551033443898307364ea281fe03", "08db2753a9bfd3f50e744877a18587b29cf29ec6cb6e807adc0432709ff821f2", "1a16e3ee99549505c1245dacb22e335f4bb81c856f240cb63e6c46427c5d23ca", "08db2753a9bfd3f50e744877a18587b29cf29ec6cb6e807adc0432709ff821f2", "7b649b0353988674c7bd93092779d2c976c92019e5e1f42d78371bb5fd7ea5e3", "d9b145762ae9f19227aa0f6759e516bd94b1706216476726f7c1b71348db2dea", "b134b54071cdca5322ca9f5d39ee9881e5a8f90b566373e7bf45d235577c8f89", "d2ec959bd2aeb38617b54218e75dfb0fc4e1d53d2709905dad37a4ec0335087d", "2532bc93b6661eada207c27836dbaa6b192ec63e07d3f54166aa8152e353402e", "6f462fb33586c80797c53198795967a40c594e6ccb386d2279714d99ac30c454", "6e078cd2efaaf0112a1a339f9d190952242f17dc152652b2f3a6d1d9d6321bcd", "e2976be454881d83ab5ddd552ea747c077f82f6ae23d58c876a0cd9da3a07849", "aebc6e2d6e04d489dcbfbf3ed41640dfa4c7d7d9e9caffc923514bdc4bfec960", "b134b54071cdca5322ca9f5d39ee9881e5a8f90b566373e7bf45d235577c8f89", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "4a5363e2ec1eb8cefee037ab0b231653c07f9b44a406ef92ed9ba3e6f5541dbf", "08db2753a9bfd3f50e744877a18587b29cf29ec6cb6e807adc0432709ff821f2", "385f61bdceefed07379e22b8d5fa7755435cb17c17313cb87e9a56bdda596515"], cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.host} {
    cannir.kernel_launch @kernel__QuantMlaPrologKernel___call___0(%arg33) [%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13, %arg14, %arg15, %arg16, %arg17, %arg18, %arg19, %arg20, %arg21, %arg22, %arg23, %arg24, %arg25, %arg26, %arg27, %arg28, %arg29, %arg30, %arg31, %arg32] {cannbotdsl.profile = {indices = array<i64>, name = "quant_mla_prolog", op_type = "quant_mla_prolog", roles = array<i64>}} : !cannir.tensor<<gm, ui8>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(1024, 1024)>, <(1024, 1)>>, <<(1024, 1024)>, <(1024, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(384, 1024)>, <(1024, 1)>>, <<(384, 1024)>, <(1024, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(576, 1024)>, <(1024, 1)>>, <<(576, 1024)>, <(1024, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(2, 128, 512)>, <(65536, 512, 1)>>, <<(2, 128, 512)>, <(65536, 512, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1, 1024)>, <(1024, 1)>>, <<(1, 1024)>, <(1024, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1, 384)>, <(384, 1)>>, <<(1, 384)>, <(384, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1, 576)>, <(576, 1)>>, <<(1, 576)>, <(576, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1, 1024)>, <(1024, 1)>>, <<(1, 1024)>, <(1024, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, identity>, !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, 512)>, <(512, 1)>>, <<(-1, 512)>, <(512, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, !cannir.tensor<<gm, f32>, <<(-1, 384)>, <(384, 1)>>, <<(-1, 384)>, <(384, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, 512)>, <(512, 1)>>, <<(-1, 512)>, <(512, 1)>>, identity>, !cannir.tensor<<gm, bf16>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>, !cannir.tensor<<gm, ui8>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>, !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, f32, f32, f32, !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
    return
  }
  func.func @kernel__QuantMlaPrologKernel___call___0_aiv(%arg0: !cannir.tensor<<gm, ui8>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(1024, 1024)>, <(1024, 1)>>, <<(1024, 1024)>, <(1024, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(384, 1024)>, <(1024, 1)>>, <<(384, 1024)>, <(1024, 1)>>, identity>, %arg3: !cannir.tensor<<gm, ui8>, <<(576, 1024)>, <(1024, 1)>>, <<(576, 1024)>, <(1024, 1)>>, identity>, %arg4: !cannir.tensor<<gm, bf16>, <<(2, 128, 512)>, <(65536, 512, 1)>>, <<(2, 128, 512)>, <(65536, 512, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(1, 1024)>, <(1024, 1)>>, <<(1, 1024)>, <(1024, 1)>>, identity>, %arg7: !cannir.tensor<<gm, f32>, <<(1, 384)>, <(384, 1)>>, <<(1, 384)>, <(384, 1)>>, identity>, %arg8: !cannir.tensor<<gm, f32>, <<(1, 576)>, <(576, 1)>>, <<(1, 576)>, <(576, 1)>>, identity>, %arg9: !cannir.tensor<<gm, f32>, <<(1, 1024)>, <(1024, 1)>>, <<(1, 1024)>, <(1024, 1)>>, identity>, %arg10: !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, %arg11: !cannir.tensor<<gm, ui8>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, identity>, %arg12: !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg13: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg14: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg15: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg16: !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg17: !cannir.tensor<<gm, f32>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>, %arg18: !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg19: !cannir.tensor<<gm, f32>, <<(-1, 512)>, <(512, 1)>>, <<(-1, 512)>, <(512, 1)>>, identity>, %arg20: !cannir.tensor<<gm, ui8>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg21: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg22: !cannir.tensor<<gm, f32>, <<(-1, 384)>, <(384, 1)>>, <<(-1, 384)>, <(384, 1)>>, identity>, %arg23: !cannir.tensor<<gm, bf16>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, %arg24: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg25: !cannir.tensor<<gm, bf16>, <<(-1, 512)>, <(512, 1)>>, <<(-1, 512)>, <(512, 1)>>, identity>, %arg26: !cannir.tensor<<gm, bf16>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>, %arg27: !cannir.tensor<<gm, ui8>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>, %arg28: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg29: f32, %arg30: f32, %arg31: f32, %arg32: !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aiv", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__QuantMlaPrologKernel___call___0", cannir.resource.ids_prepared} {
    %c2048_i32 = arith.constant 2048 : i32
    %c4096_i32 = arith.constant 4096 : i32
    %c256_i32 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c512_i64 = arith.constant 512 : i64
    %c384_i64 = arith.constant 384 : i64
    %false = arith.constant false
    %c4096_i64 = arith.constant 4096 : i64
    %c0 = arith.constant 0 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c1024_i64 = arith.constant 1024 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c2304_i64 = arith.constant 2304 : i64
    %c32_i32 = arith.constant 32 : i32
    %c4_i32 = arith.constant 4 : i32
    %c768_i64 = arith.constant 768 : i64
    %c12288_i64 = arith.constant 12288 : i64
    %c1536_i64 = arith.constant 1536 : i64
    %c128_i32 = arith.constant 128 : i32
    %c1152_i64 = arith.constant 1152 : i64
    %c576_i32 = arith.constant 576 : i32
    %c18432_i64 = arith.constant 18432 : i64
    %c9216_i32 = arith.constant 9216 : i32
    %c9216_i64 = arith.constant 9216 : i64
    %c4608_i32 = arith.constant 4608 : i32
    %c576_i64 = arith.constant 576 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c32768_i64 = arith.constant 32768 : i64
    %c73728_i64 = arith.constant 73728 : i64
    %c36864_i64 = arith.constant 36864 : i64
    %c4608_i64 = arith.constant 4608 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %c32_i64 = arith.constant 32 : i64
    %c22_i64 = arith.constant 22 : i64
    %c21_i64 = arith.constant 21 : i64
    %c20_i64 = arith.constant 20 : i64
    %c19_i64 = arith.constant 19 : i64
    %c18_i64 = arith.constant 18 : i64
    %c17_i64 = arith.constant 17 : i64
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
    %alloc = memref.alloc() {cannir.arg_index = 5 : i64} : memref<1xf32>
    %alloc_0 = memref.alloc() {cannir.arg_index = 6 : i64} : memref<1024xf32>
    %alloc_1 = memref.alloc() {cannir.arg_index = 7 : i64} : memref<384xf32>
    %alloc_2 = memref.alloc() {cannir.arg_index = 8 : i64} : memref<576xf32>
    %alloc_3 = memref.alloc() {cannir.arg_index = 9 : i64} : memref<1024xf32>
    %alloc_4 = memref.alloc() {cannir.arg_index = 10 : i64} : memref<512xf32>
    %alloc_5 = memref.alloc() {cannir.arg_index = 11 : i64} : memref<1xui8>
    %alloc_6 = memref.alloc() {cannir.arg_index = 12 : i64} : memref<1xi64>
    %alloc_7 = memref.alloc() {cannir.arg_index = 13 : i64} : memref<1xf32>
    %alloc_8 = memref.alloc() {cannir.arg_index = 14 : i64} : memref<1xbf16>
    %alloc_9 = memref.alloc() {cannir.arg_index = 15 : i64} : memref<1xbf16>
    %alloc_10 = memref.alloc() {cannir.arg_index = 16 : i64} : memref<1xf32>
    %alloc_11 = memref.alloc() {cannir.arg_index = 17 : i64} : memref<1xf32>
    %alloc_12 = memref.alloc() {cannir.arg_index = 18 : i64} : memref<1xf32>
    %alloc_13 = memref.alloc() {cannir.arg_index = 19 : i64} : memref<1xf32>
    %alloc_14 = memref.alloc() {cannir.arg_index = 20 : i64} : memref<1xui8>
    %alloc_15 = memref.alloc() {cannir.arg_index = 21 : i64} : memref<1xf32>
    %alloc_16 = memref.alloc() {cannir.arg_index = 22 : i64} : memref<1xf32>
    %alloc_17 = memref.alloc() {cannir.arg_index = 23 : i64} : memref<1xbf16>
    %alloc_18 = memref.alloc() {cannir.arg_index = 24 : i64} : memref<1xbf16>
    %alloc_19 = memref.alloc() {cannir.arg_index = 26 : i64} : memref<1xbf16>
    %alloc_20 = memref.alloc() {cannir.arg_index = 27 : i64} : memref<1xui8>
    %alloc_21 = memref.alloc() {cannir.arg_index = 28 : i64} : memref<1xf32>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xf32, 1>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<1024xf32, 1>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<4096xf32, 1>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 24576 : i64} : memref<1024xf32, 1>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 28672 : i64} : memref<1024xf32, 1>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<128xf32, 1>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 33280 : i64} : memref<1xf32, 1>
    %alloc_29 = memref.alloc() {cannir.byte_offset = 33312 : i64} : memref<1024xui8, 1>
    %alloc_30 = memref.alloc() {cannir.byte_offset = 34336 : i64} : memref<1024xui8, 1>
    %alloc_31 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xf32, 1>
    %alloc_32 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<1024xf32, 1>
    %alloc_33 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<8192xf32, 1>
    %alloc_34 = memref.alloc() {cannir.byte_offset = 40960 : i64} : memref<1024xf32, 1>
    %alloc_35 = memref.alloc() {cannir.byte_offset = 45056 : i64} : memref<1024xf32, 1>
    %alloc_36 = memref.alloc() {cannir.byte_offset = 49152 : i64} : memref<128xf32, 1>
    %alloc_37 = memref.alloc() {cannir.byte_offset = 49664 : i64} : memref<1xf32, 1>
    %alloc_38 = memref.alloc() {cannir.byte_offset = 49696 : i64} : memref<1024xui8, 1>
    %alloc_39 = memref.alloc() {cannir.byte_offset = 50720 : i64} : memref<1024xui8, 1>
    %alloc_40 = memref.alloc() {cannir.byte_offset = 51744 : i64} : memref<8xf32, 1>
    %alloc_41 = memref.alloc() {cannir.byte_offset = 51776 : i64} : memref<1024xf32, 1>
    %alloc_42 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xf32, 1>
    %alloc_43 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<1024xf32, 1>
    %alloc_44 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<1024xf32, 1>
    %alloc_45 = memref.alloc() {cannir.byte_offset = 12288 : i64} : memref<1024xf32, 1>
    %alloc_46 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<1024xf32, 1>
    %alloc_47 = memref.alloc() {cannir.byte_offset = 20480 : i64} : memref<1024xf32, 1>
    %alloc_48 = memref.alloc() {cannir.byte_offset = 24576 : i64} : memref<128xf32, 1>
    %alloc_49 = memref.alloc() {cannir.byte_offset = 25088 : i64} : memref<8xf32, 1>
    %alloc_50 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xf32, 1>
    %alloc_51 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<1024xf32, 1>
    %alloc_52 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<1024xf32, 1>
    %alloc_53 = memref.alloc() {cannir.byte_offset = 12288 : i64} : memref<1024xf32, 1>
    %alloc_54 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<1024xf32, 1>
    %alloc_55 = memref.alloc() {cannir.byte_offset = 20480 : i64} : memref<1024xf32, 1>
    %alloc_56 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1xf32, 1>
    %alloc_57 = memref.alloc() {cannir.byte_offset = 32 : i64} : memref<1024xf32, 1>
    %alloc_58 = memref.alloc() {cannir.byte_offset = 4128 : i64} : memref<1024xf32, 1>
    %alloc_59 = memref.alloc() {cannir.byte_offset = 8224 : i64} : memref<1024xf32, 1>
    %alloc_60 = memref.alloc() {cannir.byte_offset = 12320 : i64} : memref<1024xui8, 1>
    %alloc_61 = memref.alloc() {cannir.byte_offset = 13344 : i64} : memref<1024xui8, 1>
    %alloc_62 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xf32, 1>
    %alloc_63 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<1024xf32, 1>
    %alloc_64 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<128xf32, 1>
    %alloc_65 = memref.alloc() {cannir.byte_offset = 8704 : i64} : memref<8xf32, 1>
    %alloc_66 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1xf32, 1>
    %alloc_67 = memref.alloc() {cannir.byte_offset = 32 : i64} : memref<1024xf32, 1>
    %alloc_68 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xf32, 1>
    %alloc_69 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<1024xf32, 1>
    %alloc_70 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<128xf32, 1>
    %alloc_71 = memref.alloc() {cannir.byte_offset = 8704 : i64} : memref<1xf32, 1>
    %alloc_72 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1xf32, 1>
    %alloc_73 = memref.alloc() {cannir.byte_offset = 32 : i64} : memref<1024xf32, 1>
    %alloc_74 = memref.alloc() {cannir.byte_offset = 4128 : i64} : memref<1024xf32, 1>
    %alloc_75 = memref.alloc() {cannir.byte_offset = 8224 : i64} : memref<1024xui8, 1>
    %alloc_76 = memref.alloc() {cannir.byte_offset = 9248 : i64} : memref<1024xui8, 1>
    %alloc_77 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<512xf32, 1>
    %alloc_78 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<512xf32, 1>
    %alloc_79 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<512xf32, 1>
    %alloc_80 = memref.alloc() {cannir.byte_offset = 6144 : i64} : memref<512xf32, 1>
    %alloc_81 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<512xf32, 1>
    %alloc_82 = memref.alloc() {cannir.byte_offset = 10240 : i64} : memref<512xf32, 1>
    %alloc_83 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8xf32, 1>
    %alloc_84 = memref.alloc() {cannir.byte_offset = 32 : i64} : memref<512xf32, 1>
    %alloc_85 = memref.alloc() {cannir.byte_offset = 2080 : i64} : memref<512xf32, 1>
    %alloc_86 = memref.alloc() {cannir.byte_offset = 4128 : i64} : memref<576xui8, 1>
    %alloc_87 = memref.alloc() {cannir.byte_offset = 4704 : i64} : memref<576xui8, 1>
    %alloc_88 = memref.alloc() {cannir.byte_offset = 5280 : i64} : memref<64xf32, 1>
    %alloc_89 = memref.alloc() {cannir.byte_offset = 5536 : i64} : memref<64xf32, 1>
    %alloc_90 = memref.alloc() {cannir.byte_offset = 5792 : i64} : memref<64xbf16, 1>
    %alloc_91 = memref.alloc() {cannir.byte_offset = 5920 : i64} : memref<64xbf16, 1>
    %alloc_92 = memref.alloc() {cannir.byte_offset = 6048 : i64} : memref<64xbf16, 1>
    %alloc_93 = memref.alloc() {cannir.byte_offset = 6176 : i64} : memref<64xf32, 1>
    %alloc_94 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<512xbf16, 1>
    %alloc_95 = memref.alloc() {cannir.byte_offset = 1024 : i64} : memref<512xbf16, 1>
    %alloc_96 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<576xbf16, 1>
    %alloc_97 = memref.alloc() {cannir.byte_offset = 3200 : i64} : memref<576xbf16, 1>
    %alloc_98 = memref.alloc() {cannir.byte_offset = 7232 : i64} : memref<64xbf16, 1>
    %alloc_99 = memref.alloc() {cannir.byte_offset = 7360 : i64} : memref<64xbf16, 1>
    %alloc_100 = memref.alloc() {cannir.byte_offset = 7488 : i64} : memref<128xf32, 1>
    %alloc_101 = memref.alloc() {cannir.byte_offset = 8000 : i64} : memref<1536xf32, 1>
    %alloc_102 = memref.alloc() {cannir.byte_offset = 14144 : i64} : memref<1536xf32, 1>
    %alloc_103 = memref.alloc() {cannir.byte_offset = 20288 : i64} : memref<1024xbf16, 1>
    %alloc_104 = memref.alloc() {cannir.byte_offset = 22336 : i64} : memref<1024xbf16, 1>
    %alloc_105 = memref.alloc() {cannir.byte_offset = 24384 : i64} : memref<192xf32, 1>
    %alloc_106 = memref.alloc() {cannir.byte_offset = 25152 : i64} : memref<512xbf16, 1>
    %alloc_107 = memref.alloc() {cannir.byte_offset = 26176 : i64} : memref<512xbf16, 1>
    %alloc_108 = memref.alloc() {cannir.byte_offset = 27200 : i64} : memref<512xbf16, 1>
    %alloc_109 = memref.alloc() {cannir.byte_offset = 28224 : i64} : memref<512xbf16, 1>
    %alloc_110 = memref.alloc() {cannir.byte_offset = 29248 : i64} : memref<512xf32, 1>
    %alloc_111 = memref.alloc() {cannir.byte_offset = 31296 : i64} : memref<1536xf32, 1>
    %alloc_112 = memref.alloc() {cannir.byte_offset = 37440 : i64} : memref<1536xf32, 1>
    %alloc_113 = memref.alloc() {cannir.byte_offset = 43584 : i64} : memref<1024xbf16, 1>
    %alloc_114 = memref.alloc() {cannir.byte_offset = 45632 : i64} : memref<1024xbf16, 1>
    %alloc_115 = memref.alloc() {cannir.byte_offset = 47680 : i64} : memref<192xf32, 1>
    %alloc_116 = memref.alloc() {cannir.byte_offset = 48448 : i64} : memref<512xbf16, 1>
    %alloc_117 = memref.alloc() {cannir.byte_offset = 49472 : i64} : memref<512xbf16, 1>
    %alloc_118 = memref.alloc() {cannir.byte_offset = 50496 : i64} : memref<512xbf16, 1>
    %alloc_119 = memref.alloc() {cannir.byte_offset = 51520 : i64} : memref<512xbf16, 1>
    %alloc_120 = memref.alloc() {cannir.byte_offset = 52544 : i64} : memref<512xf32, 1>
    %alloc_121 = memref.alloc() {cannir.byte_offset = 54592 : i64} : memref<8xf32, 1>
    %alloc_122 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1xf32, 1>
    %alloc_123 = memref.alloc() {cannir.byte_offset = 32 : i64} : memref<1024xf32, 1>
    %alloc_124 = memref.alloc() {cannir.byte_offset = 4128 : i64} : memref<9216xbf16, 1>
    %alloc_125 = memref.alloc() {cannir.byte_offset = 22560 : i64} : memref<9216xbf16, 1>
    %alloc_126 = memref.alloc() {cannir.byte_offset = 40992 : i64} : memref<9216xui8, 1>
    %alloc_127 = memref.alloc() {cannir.byte_offset = 50208 : i64} : memref<9216xui8, 1>
    %alloc_128 = memref.alloc() {cannir.byte_offset = 4128 : i64} : memref<4608xbf16, 1>
    %alloc_129 = memref.alloc() {cannir.byte_offset = 13344 : i64} : memref<4608xbf16, 1>
    %alloc_130 = memref.alloc() {cannir.byte_offset = 22560 : i64} : memref<4608xui8, 1>
    %alloc_131 = memref.alloc() {cannir.byte_offset = 27168 : i64} : memref<4608xui8, 1>
    %alloc_132 = memref.alloc() {cannir.byte_offset = 4128 : i64} : memref<576xbf16, 1>
    %alloc_133 = memref.alloc() {cannir.byte_offset = 5280 : i64} : memref<576xbf16, 1>
    %alloc_134 = memref.alloc() {cannir.byte_offset = 6432 : i64} : memref<576xui8, 1>
    %alloc_135 = memref.alloc() {cannir.byte_offset = 7008 : i64} : memref<576xui8, 1>
    %alloc_136 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<576xbf16, 1>
    %alloc_137 = memref.alloc() {cannir.byte_offset = 1152 : i64} : memref<576xbf16, 1>
    %alloc_138 = memref.alloc() {cannir.byte_offset = 2304 : i64} : memref<128xf32, 1>
    %alloc_139 = memref.alloc() {cannir.byte_offset = 2816 : i64} : memref<8xf32, 1>
    %alloc_140 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1xf32, 1>
    %alloc_141 = memref.alloc() {cannir.byte_offset = 32 : i64} : memref<1024xf32, 1>
    %alloc_142 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<576xbf16, 1>
    %alloc_143 = memref.alloc() {cannir.byte_offset = 1152 : i64} : memref<576xbf16, 1>
    %alloc_144 = memref.alloc() {cannir.byte_offset = 2304 : i64} : memref<128xf32, 1>
    %alloc_145 = memref.alloc() {cannir.byte_offset = 2816 : i64} : memref<1xf32, 1>
    %alloc_146 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1xf32, 1>
    %alloc_147 = memref.alloc() {cannir.byte_offset = 32 : i64} : memref<9216xbf16, 1>
    %alloc_148 = memref.alloc() {cannir.byte_offset = 18464 : i64} : memref<9216xbf16, 1>
    %alloc_149 = memref.alloc() {cannir.byte_offset = 36896 : i64} : memref<9216xui8, 1>
    %alloc_150 = memref.alloc() {cannir.byte_offset = 46112 : i64} : memref<9216xui8, 1>
    %alloc_151 = memref.alloc() {cannir.byte_offset = 32 : i64} : memref<4608xbf16, 1>
    %alloc_152 = memref.alloc() {cannir.byte_offset = 9248 : i64} : memref<4608xbf16, 1>
    %alloc_153 = memref.alloc() {cannir.byte_offset = 18464 : i64} : memref<4608xui8, 1>
    %alloc_154 = memref.alloc() {cannir.byte_offset = 23072 : i64} : memref<4608xui8, 1>
    %alloc_155 = memref.alloc() {cannir.byte_offset = 32 : i64} : memref<576xbf16, 1>
    %alloc_156 = memref.alloc() {cannir.byte_offset = 1184 : i64} : memref<576xbf16, 1>
    %alloc_157 = memref.alloc() {cannir.byte_offset = 2336 : i64} : memref<576xui8, 1>
    %alloc_158 = memref.alloc() {cannir.byte_offset = 2912 : i64} : memref<576xui8, 1>
    %0 = ascscalar.get_block_idx
    %1 = ascscalar.get_block_num
    ascvec.sync_block_wait s, %c2_i64
    %2 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, ui8>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
    %3 = cannir.extract_shape %2 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
    %4 = cannir.extract_dim %3[0] : !cannir.shape<(-1, 1024)>
    %5 = arith.muli %1, %c2_i64 : i64
    %6 = arith.cmpi sle, %5, %c128_i64 : i64
    %7 = scf.if %6 -> (i1) {
      %48 = cannir.extract_layout %arg22 : !cannir.tensor<<gm, f32>, <<(-1, 384)>, <(384, 1)>>, <<(-1, 384)>, <(384, 1)>>, identity>
      %49 = cannir.extract_shape %48 : <<(-1, 384)>, <(384, 1)>> -> <(-1, 384)>
      %50 = cannir.extract_dim %49[0] : !cannir.shape<(-1, 384)>
      %51 = arith.muli %50, %c384_i64 : i64
      %52 = arith.remsi %51, %c8_i64 : i64
      %53 = arith.cmpi ne, %52, %c0_i64 : i64
      %54 = arith.cmpi slt, %52, %c0_i64 : i64
      %55 = arith.andi %53, %54 : i1
      %56 = arith.divsi %51, %c8_i64 : i64
      %57 = arith.extui %55 : i1 to i64
      %58 = arith.subi %56, %57 : i64
      %59 = arith.cmpi sle, %5, %58 : i64
      scf.yield %59 : i1
    } else {
      scf.yield %false : i1
    }
    %8 = arith.muli %1, %c16_i64 : i64
    %9 = arith.cmpi sle, %4, %8 : i64
    %10 = arith.select %7, %9, %false : i1
    %11 = arith.cmpi sle, %4, %c4_i64 : i64
    %12 = arith.select %11, %true, %10 : i1
    scf.if %11 {
      %48 = arith.cmpi eq, %0, %c0_i64 : i64
      %49 = scf.if %48 -> (i1) {
        %50 = ascscalar.get_subblock_id
        %51 = arith.cmpi eq, %50, %c0_i64 : i64
        scf.yield %51 : i1
      } else {
        scf.yield %false : i1
      }
      scf.if %49 {
        "ascvec.copy_gm2ub_plan"(%alloc_25, %alloc_3, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c4096_i64, %c0_i64, %c4096_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 4096 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 24576 : i64, ub_capacity = 4096 : i64}> : (memref<1024xf32, 1>, memref<1024xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.get_buf mte2, %c3_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%alloc_26, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c4096_i64, %c0_i64, %c4096_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 4096 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 28672 : i64, ub_capacity = 4096 : i64}> : (memref<1024xf32, 1>, memref<1024xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c3_i64, 0 {cannir.buffer.generated_sync}
        %50 = memref.load %alloc[%c0] : memref<1xf32>
        ascvec.vf(%alloc_27 : memref<128xf32, 1>) {
        ^bb0(%arg33: memref<128xf32, 1>):
          %c64 = arith.constant 64 : index
          %c0_159 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %c0_160 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg34 = %c0_160 to %c1 step %c1 {
            %57 = ascvec.create_mask : !ascvec.mask
            %58 = ascvec.duplicate_scalar %cst, %57 : f32 -> <f32>
            ascvec.storealign %arg33[%c0_159], %58, %57 : memref<128xf32, 1>, index, <f32>
            ascvec.storealign %arg33[%c64], %58, %57 : memref<128xf32, 1>, index, <f32>
          }
        } {cannir.vf_mode = "simd"}
        %51 = cannir.extract_layout %arg20 : !cannir.tensor<<gm, ui8>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
        %52 = cannir.extract_shape %51 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
        %53 = cannir.extract_dim %52[0] : !cannir.shape<(-1, 1024)>
        %54 = scf.for %arg33 = %c0_i64 to %53 step %c1_i64 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
          %57 = arith.cmpi eq, %arg34, %c0_i64 : i64
          %58 = ascscalar.memref_address %alloc_22 : memref<1024xf32, 1>
          %59 = ascscalar.memref_address %alloc_23 : memref<1024xf32, 1>
          %60 = arith.select %57, %58, %59 : i64
          %61 = arith.cmpi ne, %arg34, %c0_i64 : i64
          %62 = arith.extui %61 : i1 to i64
          %63 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
          %64 = arith.remui %63, %c2_i64 : i64
          %65 = cannir.extract_layout %arg16 : !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
          %66 = cannir.extract_shape %65 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
          %67 = cannir.extract_dim %66[0] : !cannir.shape<(-1, 1024)>
          %68 = arith.minsi %arg33, %67 : i64
          %69 = arith.subi %67, %68 : i64
          %70 = arith.minsi %69, %c1_i64 : i64
          %71 = arith.cmpi sle, %arg33, %67 : i64
          %72 = arith.select %71, %70, %c0_i64 : i64
          %73 = arith.muli %arg33, %c4096_i64 : i64
          %74 = arith.muli %72, %c4096_i64 : i64
          %75 = arith.cmpi eq, %74, %c0_i64 : i64
          %76 = arith.cmpi ne, %74, %c0_i64 : i64
          %77 = arith.extui %76 : i1 to i64
          %78 = arith.select %75, %c4_i64, %74 : i64
          %79 = ascscalar.bind_local_ptr %60 : memref<1xf32, 1>
          ascvec.get_buf mte2, %62, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%79, %alloc_10, %c0_i32, %c0_i32, %c0_i64, %73, %77, %78, %c0_i64, %78, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %62, 0 {cannir.buffer.generated_sync}
          %80 = ascscalar.bind_local_ptr %60 : memref<1024xf32, 1>
          ascvec.get_buf v, %62, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
          %81 = arith.muli %arg33, %c1024_i64 : i64
          %82 = arith.index_cast %81 : i64 to index
          ascvec.vf(%80, %alloc_26, %50, %arg29, %alloc_27, %alloc_25, %arg30, %alloc_24, %82 : memref<1024xf32, 1>, memref<1024xf32, 1>, f32, f32, memref<128xf32, 1>, memref<1024xf32, 1>, f32, memref<4096xf32, 1>, index) {
          ^bb0(%arg35: memref<1024xf32, 1>, %arg36: memref<1024xf32, 1>, %arg37: f32, %arg38: f32, %arg39: memref<128xf32, 1>, %arg40: memref<1024xf32, 1>, %arg41: f32, %arg42: memref<4096xf32, 1>, %arg43: index):
            %c64 = arith.constant 64 : index
            %cst = arith.constant 1.000000e+00 : f32
            %cst_159 = arith.constant 1.024000e+03 : f32
            %c64_i32 = arith.constant 64 : i32
            %c1024_i32_160 = arith.constant 1024 : i32
            %c0_i32_161 = arith.constant 0 : i32
            %c0_162 = arith.constant 0 : index
            %cst_163 = arith.constant 0.000000e+00 : f32
            %83 = ascvec.create_mask : !ascvec.mask
            %84 = ascvec.duplicate_scalar %cst_163, %83 : f32 -> <f32>
            %85 = scf.for %arg44 = %c0_i32_161 to %c1024_i32_160 step %c64_i32 iter_args(%arg45 = %84) -> (!ascvec.vreg<f32>)  : i32 {
              %95 = arith.index_cast %arg44 : i32 to index
              %96 = ascvec.loadalign %arg35[%95] : memref<1024xf32, 1>, index -> <f32>
              %97 = ascvec.loadalign %arg36[%95] : memref<1024xf32, 1>, index -> <f32>
              %98 = ascvec.mul %96, %97, %83 : <f32>
              %99 = ascvec.mul_scalar %98, %arg37, %83 : <f32>, f32
              ascvec.storealign %arg35[%95], %99, %83 : memref<1024xf32, 1>, index, <f32>
              %100 = ascvec.mul %99, %99, %83 : <f32>
              %101 = ascvec.reduce_sum %100, %83 : <f32>
              %102 = ascvec.add %arg45, %101, %83 : <f32>
              scf.yield %102 : !ascvec.vreg<f32>
            }
            ascvec.mem_bar vst_vld
            %86 = ascvec.duplicate %85, %83 : <f32> -> <f32>
            %87 = ascvec.duplicate_scalar %cst_159, %83 : f32 -> <f32>
            %88 = ascvec.div %86, %87, %83 {precision_mode = true} : <f32>
            %89 = ascvec.add_scalar %88, %arg38, %83 : <f32>, f32
            %90 = ascvec.sqrt %89, %83 : <f32>
            %91 = ascvec.loadalign %arg39[%c0_162] : memref<128xf32, 1>, index -> <f32>
            %92 = ascvec.loadalign %arg39[%c64] : memref<128xf32, 1>, index -> <f32>
            %93 = ascvec.duplicate_scalar %cst, %83 : f32 -> <f32>
            %94:2 = scf.for %arg44 = %c0_i32_161 to %c1024_i32_160 step %c64_i32 iter_args(%arg45 = %91, %arg46 = %92) -> (!ascvec.vreg<f32>, !ascvec.vreg<f32>)  : i32 {
              %95 = arith.index_cast %arg44 : i32 to index
              %96 = ascvec.loadalign %arg35[%95] : memref<1024xf32, 1>, index -> <f32>
              %97 = ascvec.div %96, %90, %83 {precision_mode = true} : <f32>
              %98 = ascvec.loadalign %arg40[%95] : memref<1024xf32, 1>, index -> <f32>
              %99 = ascvec.mul %97, %98, %83 : <f32>
              %100 = ascvec.mul_scalar %99, %arg41, %83 : <f32>, f32
              %101 = arith.addi %arg43, %95 : index
              ascvec.storealign %arg42[%101], %100, %83 : memref<4096xf32, 1>, index, <f32>
              %102 = ascvec.abs %100, %83 : <f32>
              %103 = ascvec.max %arg45, %102, %83 : <f32>
              %104 = ascvec.cmp_ne %100, %100, %83 : <f32>, <f32> -> !ascvec.mask
              %105 = ascvec.select %93, %84, %104 : <f32>
              %106 = ascvec.max %arg46, %105, %83 : <f32>
              scf.yield %103, %106 : !ascvec.vreg<f32>, !ascvec.vreg<f32>
            }
            %c0_164 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg44 = %c0_164 to %c1 step %c1 {
              ascvec.storealign %arg39[%c0_162], %94#0, %83 : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg39[%c64], %94#1, %83 : memref<128xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %62, 0 {cannir.buffer.generated_sync}
          scf.yield %64 : i64
        }
        ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_27, %alloc_28 : memref<128xf32, 1>, memref<1xf32, 1>) {
        ^bb0(%arg33: memref<128xf32, 1>, %arg34: memref<1xf32, 1>):
          %c64 = arith.constant 64 : index
          %cst = arith.constant 0.000000e+00 : f32
          %c2143289344_i32 = arith.constant 2143289344 : i32
          %cst_159 = arith.constant 3.276800e+04 : f32
          %c0_160 = arith.constant 0 : index
          %57 = ascvec.create_mask : !ascvec.mask
          ascvec.mem_bar vst_vld
          %c0_161 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg35 = %c0_161 to %c1 step %c1 {
            %58 = ascvec.loadalign %arg33[%c0_160] : memref<128xf32, 1>, index -> <f32>
            %59 = ascvec.reduce_max %58, %57 : <f32>
            %60 = ascvec.duplicate %59, %57 : <f32> -> <f32>
            %61 = ascvec.loadalign %arg33[%c64] : memref<128xf32, 1>, index -> <f32>
            %62 = ascvec.reduce_max %61, %57 : <f32>
            %63 = ascvec.duplicate %62, %57 : <f32> -> <f32>
            %64 = ascvec.duplicate_scalar %cst_159, %57 : f32 -> <f32>
            %65 = ascvec.div %60, %64, %57 {precision_mode = true} : <f32>
            %66 = ascvec.duplicate_scalar %c2143289344_i32, %57 : i32 -> <ui32>
            %67 = ascvec.reinterpret %66 : <ui32> -> <f32>
            %68 = ascvec.cmp_gt_scalar %63, %cst, %57 : <f32>, f32 -> !ascvec.mask
            %69 = ascvec.select %67, %65, %68 : <f32>
            ascvec.storealign_1st %arg34[%c0_160], %69 : memref<1xf32, 1>, index, <f32>
          }
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        %55 = ascscalar.bitcast_memref %alloc_28 byte_offset = %c0_i64 : memref<1xf32, 1> -> memref<?xf32, 1>
        ascvec.get_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.copy_ub2gm %alloc_15[%c0], %55[%c0], %c1_i32, %c4_i32, %c0_i64, %c4_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
        ascvec.release_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
        %56 = scf.for %arg33 = %c0_i64 to %53 step %c1_i64 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
          %57 = arith.cmpi eq, %arg34, %c0_i64 : i64
          %58 = ascscalar.memref_address %alloc_29 : memref<1024xui8, 1>
          %59 = ascscalar.memref_address %alloc_30 : memref<1024xui8, 1>
          %60 = arith.select %57, %58, %59 : i64
          %61 = ascscalar.bind_local_ptr %60 : memref<1024xui8, 1>
          %62 = arith.select %57, %c5_i64, %c6_i64 : i64
          %63 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
          %64 = arith.remui %63, %c2_i64 : i64
          ascvec.get_buf v, %62, 0 {cannir.buffer.generated_sync}
          %65 = arith.muli %arg33, %c1024_i64 : i64
          %66 = arith.index_cast %65 : i64 to index
          ascvec.vf(%alloc_28, %alloc_24, %66, %61 : memref<1xf32, 1>, memref<4096xf32, 1>, index, memref<1024xui8, 1>) {
          ^bb0(%arg35: memref<1xf32, 1>, %arg36: memref<4096xf32, 1>, %arg37: index, %arg38: memref<1024xui8, 1>):
            %c64_i32 = arith.constant 64 : i32
            %c1024_i32_159 = arith.constant 1024 : i32
            %c0_i32_160 = arith.constant 0 : i32
            %cst = arith.constant 1.1920929E-7 : f32
            %cst_161 = arith.constant 3.276800e+04 : f32
            %cst_162 = arith.constant -3.276800e+04 : f32
            %cst_163 = arith.constant 1.000000e+00 : f32
            %cst_164 = arith.constant 0.000000e+00 : f32
            %c0_165 = arith.constant 0 : index
            %79 = ascvec.create_mask : !ascvec.mask
            ascvec.mem_bar vst_vld
            %80 = ascvec.loadalign_brc %arg35[%c0_165] : memref<1xf32, 1>, index -> <f32>
            %81 = ascvec.cmp_eq_scalar %80, %cst_164, %79 : <f32>, f32 -> !ascvec.mask
            %82 = ascvec.duplicate_scalar %cst_163, %79 : f32 -> <f32>
            %83 = ascvec.select %82, %80, %81 : <f32>
            scf.for %arg39 = %c0_i32_160 to %c1024_i32_159 step %c64_i32  : i32 {
              %84 = arith.index_cast %arg39 : i32 to index
              %85 = arith.addi %arg37, %84 : index
              %86 = ascvec.loadalign %arg36[%85] : memref<4096xf32, 1>, index -> <f32>
              %87 = ascvec.div %86, %83, %79 {precision_mode = true} : <f32>
              %88 = ascvec.duplicate_scalar %cst_164, %79 : f32 -> <f32>
              %89 = ascvec.select %88, %87, %81 : <f32>
              %90 = ascvec.cmp_eq %89, %89, %79 : <f32>, <f32> -> !ascvec.mask
              %91 = ascvec.select %89, %88, %90 : <f32>
              %92 = ascvec.max_scalar %91, %cst_162, %79 : <f32>, f32
              %93 = ascvec.min_scalar %92, %cst_161, %79 : <f32>, f32
              %94 = ascvec.abs %93, %79 : <f32>
              %95 = ascvec.cmp_lt_scalar %94, %cst, %79 : <f32>, f32 -> !ascvec.mask
              %96 = ascvec.select %88, %93, %95 : <f32>
              %97 = ascvec.cast %96, %79 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
              %98 = ascvec.reinterpret %97 : <!ascdtype.hifloat8> -> <f8E4M3FN>
              ascvec.storealign_pack %arg38[%84], %98, %79 {mode = "b32_to_b8"} : memref<1024xui8, 1>, index, <f8E4M3FN>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %62, 0 {cannir.buffer.generated_sync}
          %67 = arith.minsi %arg33, %53 : i64
          %68 = arith.subi %53, %67 : i64
          %69 = arith.minsi %68, %c1_i64 : i64
          %70 = arith.cmpi sle, %arg33, %53 : i64
          %71 = arith.select %70, %69, %c0_i64 : i64
          %72 = arith.muli %arg33, %c8192_i64 : i64
          %73 = ascscalar.memref_address %alloc_14 : memref<1xui8>
          %74 = arith.divui %72, %c8_i64 : i64
          %75 = arith.addi %73, %74 : i64
          %76 = ascscalar.bind_gm_ptr %75 : memref<1xui8>
          %77 = ascscalar.bind_local_ptr %60 : memref<1xui8, 1>
          %78 = arith.cmpi ugt, %71, %c0_i64 : i64
          scf.if %78 {
            ascvec.get_buf mte3, %62, 0 {cannir.buffer.generated_sync}
            %79 = arith.trunci %71 : i64 to i32
            ascvec.copy_ub2gm %76[%c0], %77[%c0], %79, %c1024_i32, %c1024_i64, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
            ascvec.release_buf mte3, %62, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %64 : i64
        }
      }
    } else {
      scf.if %10 {
        %48 = arith.muli %0, %c2_i64 : i64
        %49 = ascscalar.get_subblock_id
        %50 = arith.addi %48, %49 : i64
        "ascvec.copy_gm2ub_plan"(%alloc_34, %alloc_3, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c4096_i64, %c0_i64, %c4096_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 4096 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 40960 : i64, ub_capacity = 4096 : i64}> : (memref<1024xf32, 1>, memref<1024xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.get_buf mte2, %c3_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%alloc_35, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c4096_i64, %c0_i64, %c4096_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 4096 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 45056 : i64, ub_capacity = 4096 : i64}> : (memref<1024xf32, 1>, memref<1024xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c3_i64, 0 {cannir.buffer.generated_sync}
        %51 = memref.load %alloc[%c0] : memref<1xf32>
        ascvec.vf(%alloc_36 : memref<128xf32, 1>) {
        ^bb0(%arg33: memref<128xf32, 1>):
          %c64 = arith.constant 64 : index
          %c0_159 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %c0_160 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg34 = %c0_160 to %c1 step %c1 {
            %95 = ascvec.create_mask : !ascvec.mask
            %96 = ascvec.duplicate_scalar %cst, %95 : f32 -> <f32>
            ascvec.storealign %arg33[%c0_159], %96, %95 : memref<128xf32, 1>, index, <f32>
            ascvec.storealign %arg33[%c64], %96, %95 : memref<128xf32, 1>, index, <f32>
          }
        } {cannir.vf_mode = "simd"}
        %52 = cannir.extract_layout %arg20 : !cannir.tensor<<gm, ui8>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
        %53 = cannir.extract_shape %52 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
        %54 = cannir.extract_dim %53[0] : !cannir.shape<(-1, 1024)>
        %55 = scf.for %arg33 = %50 to %54 step %5 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
          %95 = arith.cmpi eq, %arg34, %c0_i64 : i64
          %96 = ascscalar.memref_address %alloc_31 : memref<1024xf32, 1>
          %97 = ascscalar.memref_address %alloc_32 : memref<1024xf32, 1>
          %98 = arith.select %95, %96, %97 : i64
          %99 = arith.cmpi ne, %arg34, %c0_i64 : i64
          %100 = arith.extui %99 : i1 to i64
          %101 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
          %102 = arith.remui %101, %c2_i64 : i64
          %103 = cannir.extract_layout %arg16 : !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
          %104 = cannir.extract_shape %103 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
          %105 = cannir.extract_dim %104[0] : !cannir.shape<(-1, 1024)>
          %106 = arith.minsi %arg33, %105 : i64
          %107 = arith.subi %105, %106 : i64
          %108 = arith.minsi %107, %c1_i64 : i64
          %109 = arith.cmpi sle, %arg33, %105 : i64
          %110 = arith.select %109, %108, %c0_i64 : i64
          %111 = arith.muli %arg33, %c4096_i64 : i64
          %112 = arith.muli %110, %c4096_i64 : i64
          %113 = arith.cmpi eq, %112, %c0_i64 : i64
          %114 = arith.cmpi ne, %112, %c0_i64 : i64
          %115 = arith.extui %114 : i1 to i64
          %116 = arith.select %113, %c4_i64, %112 : i64
          %117 = ascscalar.bind_local_ptr %98 : memref<1xf32, 1>
          ascvec.get_buf mte2, %100, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%117, %alloc_10, %c0_i32, %c0_i32, %c0_i64, %111, %115, %116, %c0_i64, %116, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %100, 0 {cannir.buffer.generated_sync}
          %118 = ascscalar.bind_local_ptr %98 : memref<1024xf32, 1>
          %119 = arith.subi %arg33, %50 : i64
          %120 = arith.remsi %119, %5 : i64
          %121 = arith.cmpi ne, %120, %c0_i64 : i64
          %122 = arith.cmpi slt, %120, %c0_i64 : i64
          %123 = arith.cmpi slt, %5, %c0_i64 : i64
          %124 = arith.xori %122, %123 : i1
          %125 = arith.andi %121, %124 : i1
          %126 = arith.divsi %119, %5 : i64
          %127 = arith.extui %125 : i1 to i64
          %128 = arith.subi %126, %127 : i64
          ascvec.get_buf v, %100, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
          %129 = arith.muli %128, %c1024_i64 : i64
          %130 = arith.index_cast %129 : i64 to index
          ascvec.vf(%118, %alloc_35, %51, %arg29, %alloc_36, %alloc_34, %arg30, %alloc_33, %130 : memref<1024xf32, 1>, memref<1024xf32, 1>, f32, f32, memref<128xf32, 1>, memref<1024xf32, 1>, f32, memref<8192xf32, 1>, index) {
          ^bb0(%arg35: memref<1024xf32, 1>, %arg36: memref<1024xf32, 1>, %arg37: f32, %arg38: f32, %arg39: memref<128xf32, 1>, %arg40: memref<1024xf32, 1>, %arg41: f32, %arg42: memref<8192xf32, 1>, %arg43: index):
            %c64 = arith.constant 64 : index
            %cst = arith.constant 1.000000e+00 : f32
            %cst_159 = arith.constant 1.024000e+03 : f32
            %c64_i32 = arith.constant 64 : i32
            %c1024_i32_160 = arith.constant 1024 : i32
            %c0_i32_161 = arith.constant 0 : i32
            %c0_162 = arith.constant 0 : index
            %cst_163 = arith.constant 0.000000e+00 : f32
            %131 = ascvec.create_mask : !ascvec.mask
            %132 = ascvec.duplicate_scalar %cst_163, %131 : f32 -> <f32>
            %133 = scf.for %arg44 = %c0_i32_161 to %c1024_i32_160 step %c64_i32 iter_args(%arg45 = %132) -> (!ascvec.vreg<f32>)  : i32 {
              %143 = arith.index_cast %arg44 : i32 to index
              %144 = ascvec.loadalign %arg35[%143] : memref<1024xf32, 1>, index -> <f32>
              %145 = ascvec.loadalign %arg36[%143] : memref<1024xf32, 1>, index -> <f32>
              %146 = ascvec.mul %144, %145, %131 : <f32>
              %147 = ascvec.mul_scalar %146, %arg37, %131 : <f32>, f32
              ascvec.storealign %arg35[%143], %147, %131 : memref<1024xf32, 1>, index, <f32>
              %148 = ascvec.mul %147, %147, %131 : <f32>
              %149 = ascvec.reduce_sum %148, %131 : <f32>
              %150 = ascvec.add %arg45, %149, %131 : <f32>
              scf.yield %150 : !ascvec.vreg<f32>
            }
            ascvec.mem_bar vst_vld
            %134 = ascvec.duplicate %133, %131 : <f32> -> <f32>
            %135 = ascvec.duplicate_scalar %cst_159, %131 : f32 -> <f32>
            %136 = ascvec.div %134, %135, %131 {precision_mode = true} : <f32>
            %137 = ascvec.add_scalar %136, %arg38, %131 : <f32>, f32
            %138 = ascvec.sqrt %137, %131 : <f32>
            %139 = ascvec.loadalign %arg39[%c0_162] : memref<128xf32, 1>, index -> <f32>
            %140 = ascvec.loadalign %arg39[%c64] : memref<128xf32, 1>, index -> <f32>
            %141 = ascvec.duplicate_scalar %cst, %131 : f32 -> <f32>
            %142:2 = scf.for %arg44 = %c0_i32_161 to %c1024_i32_160 step %c64_i32 iter_args(%arg45 = %139, %arg46 = %140) -> (!ascvec.vreg<f32>, !ascvec.vreg<f32>)  : i32 {
              %143 = arith.index_cast %arg44 : i32 to index
              %144 = ascvec.loadalign %arg35[%143] : memref<1024xf32, 1>, index -> <f32>
              %145 = ascvec.div %144, %138, %131 {precision_mode = true} : <f32>
              %146 = ascvec.loadalign %arg40[%143] : memref<1024xf32, 1>, index -> <f32>
              %147 = ascvec.mul %145, %146, %131 : <f32>
              %148 = ascvec.mul_scalar %147, %arg41, %131 : <f32>, f32
              %149 = arith.addi %arg43, %143 : index
              ascvec.storealign %arg42[%149], %148, %131 : memref<8192xf32, 1>, index, <f32>
              %150 = ascvec.abs %148, %131 : <f32>
              %151 = ascvec.max %arg45, %150, %131 : <f32>
              %152 = ascvec.cmp_ne %148, %148, %131 : <f32>, <f32> -> !ascvec.mask
              %153 = ascvec.select %141, %132, %152 : <f32>
              %154 = ascvec.max %arg46, %153, %131 : <f32>
              scf.yield %151, %154 : !ascvec.vreg<f32>, !ascvec.vreg<f32>
            }
            %c0_164 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg44 = %c0_164 to %c1 step %c1 {
              ascvec.storealign %arg39[%c0_162], %142#0, %131 : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg39[%c64], %142#1, %131 : memref<128xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c3_i64, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %100, 0 {cannir.buffer.generated_sync}
          scf.yield %102 : i64
        }
        %56 = cannir.extract_layout %arg22 : !cannir.tensor<<gm, f32>, <<(-1, 384)>, <(384, 1)>>, <<(-1, 384)>, <(384, 1)>>, identity>
        %57 = cannir.extract_shape %56 : <<(-1, 384)>, <(384, 1)>> -> <(-1, 384)>
        %58 = cannir.extract_dim %57[0] : !cannir.shape<(-1, 384)>
        %59 = arith.muli %58, %c384_i64 : i64
        %60 = arith.remsi %59, %c8_i64 : i64
        %61 = arith.cmpi ne, %60, %c0_i64 : i64
        %62 = arith.cmpi slt, %60, %c0_i64 : i64
        %63 = arith.andi %61, %62 : i1
        %64 = arith.divsi %59, %c8_i64 : i64
        %65 = arith.extui %63 : i1 to i64
        %66 = arith.subi %64, %65 : i64
        ascvec.get_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_40, %alloc_36 : memref<8xf32, 1>, memref<128xf32, 1>) {
        ^bb0(%arg33: memref<8xf32, 1>, %arg34: memref<128xf32, 1>):
          %c1 = arith.constant 1 : index
          %c64 = arith.constant 64 : index
          %c0_159 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %c8_i32 = arith.constant 8 : i32
          %mask, %remaining_out = ascvec.update_mask %c8_i32 : !ascvec.mask
          %95 = ascvec.create_mask : !ascvec.mask
          %96 = ascvec.duplicate_scalar %cst, %mask : f32 -> <f32>
          ascvec.storealign %arg33[%c0_159], %96, %mask : memref<8xf32, 1>, index, <f32>
          ascvec.mem_bar vst_vld
          %c0_160 = arith.constant 0 : index
          %c1_161 = arith.constant 1 : index
          scf.for %arg35 = %c0_160 to %c1_161 step %c1_161 {
            %97 = ascvec.loadalign %arg34[%c0_159] : memref<128xf32, 1>, index -> <f32>
            %98 = ascvec.reduce_max %97, %95 : <f32>
            %99 = ascvec.duplicate %98, %95 : <f32> -> <f32>
            %100 = ascvec.loadalign %arg34[%c64] : memref<128xf32, 1>, index -> <f32>
            %101 = ascvec.reduce_max %100, %95 : <f32>
            %102 = ascvec.duplicate %101, %95 : <f32> -> <f32>
            ascvec.storealign_1st %arg33[%c0_159], %99 : memref<8xf32, 1>, index, <f32>
            ascvec.storealign_1st %arg33[%c1], %102 : memref<8xf32, 1>, index, <f32>
          }
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
        %67 = arith.minsi %50, %66 : i64
        %68 = arith.subi %66, %67 : i64
        %69 = arith.minsi %68, %c1_i64 : i64
        %70 = arith.cmpi sle, %50, %66 : i64
        %71 = arith.select %70, %69, %c0_i64 : i64
        %72 = arith.muli %50, %c256_i64 : i64
        %73 = ascscalar.memref_address %alloc_16 : memref<1xf32>
        %74 = arith.divui %72, %c8_i64 : i64
        %75 = arith.addi %73, %74 : i64
        %76 = ascscalar.bind_gm_ptr %75 : memref<1xf32>
        %77 = ascscalar.bitcast_memref %alloc_40 byte_offset = %c0_i64 : memref<8xf32, 1> -> memref<?xf32, 1>
        %78 = arith.cmpi ugt, %71, %c0_i64 : i64
        scf.if %78 {
          ascvec.get_buf mte3, %c7_i64, 0 {cannir.buffer.generated_sync}
          %95 = arith.trunci %71 : i64 to i32
          ascvec.copy_ub2gm %76[%c0], %77[%c0], %95, %c32_i32, %c32_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
          ascvec.release_buf mte3, %c7_i64, 0 {cannir.buffer.generated_sync}
        }
        ascvec.sync_all
        ascvec.sync_block_arrive mte3, %c0_i64
        ascvec.sync_block_wait s, %c2_i64
        %79 = arith.maxsi %5, %c1_i64 : i64
        %80 = arith.divsi %66, %79 : i64
        %81 = arith.minsi %80, %c0_i64 : i64
        %82 = arith.muli %81, %5 : i64
        %83 = arith.subi %66, %82 : i64
        %84 = arith.minsi %5, %83 : i64
        %85 = arith.cmpi sge, %80, %c0_i64 : i64
        %86 = arith.select %85, %84, %c0_i64 : i64
        %87 = arith.muli %86, %c32_i64 : i64
        %88 = arith.cmpi eq, %87, %c0_i64 : i64
        %89 = arith.cmpi ne, %87, %c0_i64 : i64
        %90 = arith.extui %89 : i1 to i64
        %91 = arith.select %88, %c4_i64, %87 : i64
        ascvec.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%alloc_41, %alloc_16, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %90, %91, %c0_i64, %91, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 51776 : i64, ub_capacity = 4096 : i64}> : (memref<1024xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
        %92 = arith.trunci %5 : i64 to i32
        ascvec.get_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_41, %92, %alloc_37 : memref<1024xf32, 1>, i32, memref<1xf32, 1>) {
        ^bb0(%arg33: memref<1024xf32, 1>, %arg34: i32, %arg35: memref<1xf32, 1>):
          %c2143289344_i32 = arith.constant 2143289344 : i32
          %cst = arith.constant 3.276800e+04 : f32
          %c0_i32_159 = arith.constant 0 : i32
          %c1_i32_160 = arith.constant 1 : i32
          %c0_161 = arith.constant 0 : index
          %c8_i32 = arith.constant 8 : i32
          %cst_162 = arith.constant 0.000000e+00 : f32
          %95 = ascvec.create_mask : !ascvec.mask
          %96 = ascvec.duplicate_scalar %cst_162, %95 : f32 -> <f32>
          %97:2 = scf.for %arg36 = %c0_i32_159 to %arg34 step %c1_i32_160 iter_args(%arg37 = %96, %arg38 = %96) -> (!ascvec.vreg<f32>, !ascvec.vreg<f32>)  : i32 {
            %98 = arith.muli %arg36, %c8_i32 : i32
            %99 = arith.index_cast %98 : i32 to index
            %100 = ascvec.loadalign_brc %arg33[%99] : memref<1024xf32, 1>, index -> <f32>
            %101 = ascvec.max %arg37, %100, %95 : <f32>
            %102 = arith.addi %98, %c1_i32_160 : i32
            %103 = arith.index_cast %102 : i32 to index
            %104 = ascvec.loadalign_brc %arg33[%103] : memref<1024xf32, 1>, index -> <f32>
            %105 = ascvec.max %arg38, %104, %95 : <f32>
            scf.yield %101, %105 : !ascvec.vreg<f32>, !ascvec.vreg<f32>
          }
          %c0_163 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg36 = %c0_163 to %c1 step %c1 {
            %98 = ascvec.duplicate_scalar %cst, %95 : f32 -> <f32>
            %99 = ascvec.div %97#0, %98, %95 {precision_mode = true} : <f32>
            %100 = ascvec.duplicate_scalar %c2143289344_i32, %95 : i32 -> <ui32>
            %101 = ascvec.reinterpret %100 : <ui32> -> <f32>
            %102 = ascvec.cmp_gt_scalar %97#1, %cst_162, %95 : <f32>, f32 -> !ascvec.mask
            %103 = ascvec.select %101, %99, %102 : <f32>
            ascvec.storealign_1st %arg35[%c0_161], %103 : memref<1xf32, 1>, index, <f32>
          }
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
        %93 = arith.cmpi eq, %50, %c0_i64 : i64
        scf.if %93 {
          %95 = ascscalar.bitcast_memref %alloc_37 byte_offset = %c0_i64 : memref<1xf32, 1> -> memref<?xf32, 1>
          ascvec.get_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %alloc_15[%c0], %95[%c0], %c1_i32, %c4_i32, %c0_i64, %c4_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
          ascvec.release_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
        }
        %94 = scf.for %arg33 = %50 to %54 step %5 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
          %95 = arith.subi %arg33, %50 : i64
          %96 = arith.remsi %95, %5 : i64
          %97 = arith.cmpi ne, %96, %c0_i64 : i64
          %98 = arith.cmpi slt, %96, %c0_i64 : i64
          %99 = arith.cmpi slt, %5, %c0_i64 : i64
          %100 = arith.xori %98, %99 : i1
          %101 = arith.andi %97, %100 : i1
          %102 = arith.divsi %95, %5 : i64
          %103 = arith.extui %101 : i1 to i64
          %104 = arith.subi %102, %103 : i64
          %105 = arith.cmpi eq, %arg34, %c0_i64 : i64
          %106 = ascscalar.memref_address %alloc_38 : memref<1024xui8, 1>
          %107 = ascscalar.memref_address %alloc_39 : memref<1024xui8, 1>
          %108 = arith.select %105, %106, %107 : i64
          %109 = ascscalar.bind_local_ptr %108 : memref<1024xui8, 1>
          %110 = arith.select %105, %c5_i64, %c6_i64 : i64
          %111 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
          %112 = arith.remui %111, %c2_i64 : i64
          ascvec.get_buf v, %110, 0 {cannir.buffer.generated_sync}
          %113 = arith.muli %104, %c1024_i64 : i64
          %114 = arith.index_cast %113 : i64 to index
          ascvec.vf(%alloc_37, %alloc_33, %114, %109 : memref<1xf32, 1>, memref<8192xf32, 1>, index, memref<1024xui8, 1>) {
          ^bb0(%arg35: memref<1xf32, 1>, %arg36: memref<8192xf32, 1>, %arg37: index, %arg38: memref<1024xui8, 1>):
            %c64_i32 = arith.constant 64 : i32
            %c1024_i32_159 = arith.constant 1024 : i32
            %c0_i32_160 = arith.constant 0 : i32
            %cst = arith.constant 1.1920929E-7 : f32
            %cst_161 = arith.constant 3.276800e+04 : f32
            %cst_162 = arith.constant -3.276800e+04 : f32
            %cst_163 = arith.constant 1.000000e+00 : f32
            %cst_164 = arith.constant 0.000000e+00 : f32
            %c0_165 = arith.constant 0 : index
            %127 = ascvec.create_mask : !ascvec.mask
            ascvec.mem_bar vst_vld
            %128 = ascvec.loadalign_brc %arg35[%c0_165] : memref<1xf32, 1>, index -> <f32>
            %129 = ascvec.cmp_eq_scalar %128, %cst_164, %127 : <f32>, f32 -> !ascvec.mask
            %130 = ascvec.duplicate_scalar %cst_163, %127 : f32 -> <f32>
            %131 = ascvec.select %130, %128, %129 : <f32>
            scf.for %arg39 = %c0_i32_160 to %c1024_i32_159 step %c64_i32  : i32 {
              %132 = arith.index_cast %arg39 : i32 to index
              %133 = arith.addi %arg37, %132 : index
              %134 = ascvec.loadalign %arg36[%133] : memref<8192xf32, 1>, index -> <f32>
              %135 = ascvec.div %134, %131, %127 {precision_mode = true} : <f32>
              %136 = ascvec.duplicate_scalar %cst_164, %127 : f32 -> <f32>
              %137 = ascvec.select %136, %135, %129 : <f32>
              %138 = ascvec.cmp_eq %137, %137, %127 : <f32>, <f32> -> !ascvec.mask
              %139 = ascvec.select %137, %136, %138 : <f32>
              %140 = ascvec.max_scalar %139, %cst_162, %127 : <f32>, f32
              %141 = ascvec.min_scalar %140, %cst_161, %127 : <f32>, f32
              %142 = ascvec.abs %141, %127 : <f32>
              %143 = ascvec.cmp_lt_scalar %142, %cst, %127 : <f32>, f32 -> !ascvec.mask
              %144 = ascvec.select %136, %141, %143 : <f32>
              %145 = ascvec.cast %144, %127 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
              %146 = ascvec.reinterpret %145 : <!ascdtype.hifloat8> -> <f8E4M3FN>
              ascvec.storealign_pack %arg38[%132], %146, %127 {mode = "b32_to_b8"} : memref<1024xui8, 1>, index, <f8E4M3FN>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %110, 0 {cannir.buffer.generated_sync}
          %115 = arith.minsi %arg33, %54 : i64
          %116 = arith.subi %54, %115 : i64
          %117 = arith.minsi %116, %c1_i64 : i64
          %118 = arith.cmpi sle, %arg33, %54 : i64
          %119 = arith.select %118, %117, %c0_i64 : i64
          %120 = arith.muli %arg33, %c8192_i64 : i64
          %121 = ascscalar.memref_address %alloc_14 : memref<1xui8>
          %122 = arith.divui %120, %c8_i64 : i64
          %123 = arith.addi %121, %122 : i64
          %124 = ascscalar.bind_gm_ptr %123 : memref<1xui8>
          %125 = ascscalar.bind_local_ptr %108 : memref<1xui8, 1>
          %126 = arith.cmpi ugt, %119, %c0_i64 : i64
          scf.if %126 {
            ascvec.get_buf mte3, %110, 0 {cannir.buffer.generated_sync}
            %127 = arith.trunci %119 : i64 to i32
            ascvec.copy_ub2gm %124[%c0], %125[%c0], %127, %c1024_i32, %c1024_i64, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
            ascvec.release_buf mte3, %110, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %112 : i64
        }
      } else {
        scf.if %7 {
          "ascvec.copy_gm2ub_plan"(%alloc_46, %alloc_3, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c4096_i64, %c0_i64, %c4096_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 4096 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 16384 : i64, ub_capacity = 4096 : i64}> : (memref<1024xf32, 1>, memref<1024xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.get_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%alloc_47, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c4096_i64, %c0_i64, %c4096_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 4096 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 20480 : i64, ub_capacity = 4096 : i64}> : (memref<1024xf32, 1>, memref<1024xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
          %48 = memref.load %alloc[%c0] : memref<1xf32>
          %49 = arith.muli %0, %c2_i64 : i64
          %50 = ascscalar.get_subblock_id
          %51 = arith.addi %49, %50 : i64
          ascvec.vf(%alloc_48 : memref<128xf32, 1>) {
          ^bb0(%arg33: memref<128xf32, 1>):
            %c64 = arith.constant 64 : index
            %c0_159 = arith.constant 0 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c0_160 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg34 = %c0_160 to %c1 step %c1 {
              %79 = ascvec.create_mask : !ascvec.mask
              %80 = ascvec.duplicate_scalar %cst, %79 : f32 -> <f32>
              ascvec.storealign %arg33[%c0_159], %80, %79 : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg33[%c64], %80, %79 : memref<128xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          %52 = cannir.extract_layout %arg18 : !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
          %53 = cannir.extract_shape %52 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
          %54 = cannir.extract_dim %53[0] : !cannir.shape<(-1, 1024)>
          %55 = scf.for %arg33 = %51 to %54 step %5 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
            %79 = arith.cmpi eq, %arg34, %c0_i64 : i64
            %80 = ascscalar.memref_address %alloc_42 : memref<1024xf32, 1>
            %81 = ascscalar.memref_address %alloc_43 : memref<1024xf32, 1>
            %82 = arith.select %79, %80, %81 : i64
            %83 = arith.cmpi ne, %arg34, %c0_i64 : i64
            %84 = arith.extui %83 : i1 to i64
            %85 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
            %86 = arith.remui %85, %c2_i64 : i64
            %87 = cannir.extract_layout %arg16 : !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
            %88 = cannir.extract_shape %87 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
            %89 = cannir.extract_dim %88[0] : !cannir.shape<(-1, 1024)>
            %90 = arith.minsi %arg33, %89 : i64
            %91 = arith.subi %89, %90 : i64
            %92 = arith.minsi %91, %c1_i64 : i64
            %93 = arith.cmpi sle, %arg33, %89 : i64
            %94 = arith.select %93, %92, %c0_i64 : i64
            %95 = arith.muli %arg33, %c4096_i64 : i64
            %96 = arith.muli %94, %c4096_i64 : i64
            %97 = arith.cmpi eq, %96, %c0_i64 : i64
            %98 = arith.cmpi ne, %96, %c0_i64 : i64
            %99 = arith.extui %98 : i1 to i64
            %100 = arith.select %97, %c4_i64, %96 : i64
            %101 = ascscalar.bind_local_ptr %82 : memref<1xf32, 1>
            ascvec.get_buf mte2, %84, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%101, %alloc_10, %c0_i32, %c0_i32, %c0_i64, %95, %99, %100, %c0_i64, %100, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %84, 0 {cannir.buffer.generated_sync}
            %102 = ascscalar.bind_local_ptr %82 : memref<1024xf32, 1>
            %103 = ascscalar.memref_address %alloc_44 : memref<1024xf32, 1>
            %104 = ascscalar.memref_address %alloc_45 : memref<1024xf32, 1>
            %105 = arith.select %79, %103, %104 : i64
            %106 = ascscalar.bind_local_ptr %105 : memref<1024xf32, 1>
            %107 = arith.select %79, %c2_i64, %c3_i64 : i64
            ascvec.get_buf v, %84, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%102, %alloc_47, %48 : memref<1024xf32, 1>, memref<1024xf32, 1>, f32) {
            ^bb0(%arg35: memref<1024xf32, 1>, %arg36: memref<1024xf32, 1>, %arg37: f32):
              %c64_i32 = arith.constant 64 : i32
              %c1024_i32_159 = arith.constant 1024 : i32
              %c0_i32_160 = arith.constant 0 : i32
              %120 = ascvec.create_mask : !ascvec.mask
              scf.for %arg38 = %c0_i32_160 to %c1024_i32_159 step %c64_i32  : i32 {
                %121 = arith.index_cast %arg38 : i32 to index
                %122 = ascvec.loadalign %arg35[%121] : memref<1024xf32, 1>, index -> <f32>
                %123 = ascvec.loadalign %arg36[%121] : memref<1024xf32, 1>, index -> <f32>
                %124 = ascvec.mul %122, %123, %120 : <f32>
                %125 = ascvec.mul_scalar %124, %arg37, %120 : <f32>, f32
                ascvec.storealign %arg35[%121], %125, %120 : memref<1024xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %107, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%102, %arg29, %alloc_46, %arg30, %106 : memref<1024xf32, 1>, f32, memref<1024xf32, 1>, f32, memref<1024xf32, 1>) {
            ^bb0(%arg35: memref<1024xf32, 1>, %arg36: f32, %arg37: memref<1024xf32, 1>, %arg38: f32, %arg39: memref<1024xf32, 1>):
              %cst = arith.constant 1.024000e+03 : f32
              %c64_i32 = arith.constant 64 : i32
              %c1024_i32_159 = arith.constant 1024 : i32
              %c0_i32_160 = arith.constant 0 : i32
              %cst_161 = arith.constant 0.000000e+00 : f32
              %120 = ascvec.create_mask : !ascvec.mask
              ascvec.mem_bar vst_vld
              %121 = ascvec.duplicate_scalar %cst_161, %120 : f32 -> <f32>
              %122 = scf.for %arg40 = %c0_i32_160 to %c1024_i32_159 step %c64_i32 iter_args(%arg41 = %121) -> (!ascvec.vreg<f32>)  : i32 {
                %128 = arith.index_cast %arg40 : i32 to index
                %129 = ascvec.loadalign %arg35[%128] : memref<1024xf32, 1>, index -> <f32>
                %130 = ascvec.mul %129, %129, %120 : <f32>
                %131 = ascvec.reduce_sum %130, %120 : <f32>
                %132 = ascvec.add %arg41, %131, %120 : <f32>
                scf.yield %132 : !ascvec.vreg<f32>
              }
              %123 = ascvec.duplicate %122, %120 : <f32> -> <f32>
              %124 = ascvec.duplicate_scalar %cst, %120 : f32 -> <f32>
              %125 = ascvec.div %123, %124, %120 {precision_mode = true} : <f32>
              %126 = ascvec.add_scalar %125, %arg36, %120 : <f32>, f32
              %127 = ascvec.sqrt %126, %120 : <f32>
              scf.for %arg40 = %c0_i32_160 to %c1024_i32_159 step %c64_i32  : i32 {
                %128 = arith.index_cast %arg40 : i32 to index
                %129 = ascvec.loadalign %arg35[%128] : memref<1024xf32, 1>, index -> <f32>
                %130 = ascvec.div %129, %127, %120 {precision_mode = true} : <f32>
                %131 = ascvec.loadalign %arg37[%128] : memref<1024xf32, 1>, index -> <f32>
                %132 = ascvec.mul %130, %131, %120 : <f32>
                %133 = ascvec.mul_scalar %132, %arg38, %120 : <f32>, f32
                ascvec.storealign %arg39[%128], %133, %120 : memref<1024xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %84, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_48, %106 : memref<128xf32, 1>, memref<1024xf32, 1>) {
            ^bb0(%arg35: memref<128xf32, 1>, %arg36: memref<1024xf32, 1>):
              %c64 = arith.constant 64 : index
              %c64_i32 = arith.constant 64 : i32
              %c1024_i32_159 = arith.constant 1024 : i32
              %c0_i32_160 = arith.constant 0 : i32
              %cst = arith.constant 0.000000e+00 : f32
              %cst_161 = arith.constant 1.000000e+00 : f32
              %c0_162 = arith.constant 0 : index
              %120 = ascvec.create_mask : !ascvec.mask
              ascvec.mem_bar vst_vld
              %121 = ascvec.loadalign %arg35[%c0_162] : memref<128xf32, 1>, index -> <f32>
              %122 = ascvec.loadalign %arg35[%c64] : memref<128xf32, 1>, index -> <f32>
              %123 = ascvec.duplicate_scalar %cst_161, %120 : f32 -> <f32>
              %124 = ascvec.duplicate_scalar %cst, %120 : f32 -> <f32>
              %125:2 = scf.for %arg37 = %c0_i32_160 to %c1024_i32_159 step %c64_i32 iter_args(%arg38 = %121, %arg39 = %122) -> (!ascvec.vreg<f32>, !ascvec.vreg<f32>)  : i32 {
                %126 = arith.index_cast %arg37 : i32 to index
                %127 = ascvec.loadalign %arg36[%126] : memref<1024xf32, 1>, index -> <f32>
                %128 = ascvec.abs %127, %120 : <f32>
                %129 = ascvec.max %arg38, %128, %120 : <f32>
                %130 = ascvec.cmp_ne %127, %127, %120 : <f32>, <f32> -> !ascvec.mask
                %131 = ascvec.select %123, %124, %130 : <f32>
                %132 = ascvec.max %arg39, %131, %120 : <f32>
                scf.yield %129, %132 : !ascvec.vreg<f32>, !ascvec.vreg<f32>
              }
              %c0_163 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg37 = %c0_163 to %c1 step %c1 {
                ascvec.storealign %arg35[%c0_162], %125#0, %120 : memref<128xf32, 1>, index, <f32>
                ascvec.storealign %arg35[%c64], %125#1, %120 : memref<128xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %107, 0 {cannir.buffer.generated_sync}
            %108 = arith.minsi %arg33, %54 : i64
            %109 = arith.subi %54, %108 : i64
            %110 = arith.minsi %109, %c1_i64 : i64
            %111 = arith.cmpi sle, %arg33, %54 : i64
            %112 = arith.select %111, %110, %c0_i64 : i64
            %113 = arith.muli %arg33, %c32768_i64 : i64
            %114 = ascscalar.memref_address %alloc_12 : memref<1xf32>
            %115 = arith.divui %113, %c8_i64 : i64
            %116 = arith.addi %114, %115 : i64
            %117 = ascscalar.bind_gm_ptr %116 : memref<1xf32>
            %118 = ascscalar.bind_local_ptr %105 : memref<1xf32, 1>
            %119 = arith.cmpi ugt, %112, %c0_i64 : i64
            scf.if %119 {
              ascvec.get_buf mte3, %107, 0 {cannir.buffer.generated_sync}
              %120 = arith.trunci %112 : i64 to i32
              ascvec.copy_ub2gm %117[%c0], %118[%c0], %120, %c4096_i32, %c4096_i64, %c4096_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
              ascvec.release_buf mte3, %107, 0 {cannir.buffer.generated_sync}
            }
            scf.yield %86 : i64
          }
          %56 = cannir.extract_layout %arg22 : !cannir.tensor<<gm, f32>, <<(-1, 384)>, <(384, 1)>>, <<(-1, 384)>, <(384, 1)>>, identity>
          %57 = cannir.extract_shape %56 : <<(-1, 384)>, <(384, 1)>> -> <(-1, 384)>
          %58 = cannir.extract_dim %57[0] : !cannir.shape<(-1, 384)>
          %59 = arith.muli %58, %c384_i64 : i64
          %60 = arith.remsi %59, %c8_i64 : i64
          %61 = arith.cmpi ne, %60, %c0_i64 : i64
          %62 = arith.cmpi slt, %60, %c0_i64 : i64
          %63 = arith.andi %61, %62 : i1
          %64 = arith.divsi %59, %c8_i64 : i64
          %65 = arith.extui %63 : i1 to i64
          %66 = arith.subi %64, %65 : i64
          ascvec.get_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_49, %alloc_48 : memref<8xf32, 1>, memref<128xf32, 1>) {
          ^bb0(%arg33: memref<8xf32, 1>, %arg34: memref<128xf32, 1>):
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c0_159 = arith.constant 0 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8_i32 = arith.constant 8 : i32
            %79 = ascvec.create_mask : !ascvec.mask
            %mask, %remaining_out = ascvec.update_mask %c8_i32 : !ascvec.mask
            %80 = ascvec.duplicate_scalar %cst, %mask : f32 -> <f32>
            ascvec.storealign %arg33[%c0_159], %80, %mask : memref<8xf32, 1>, index, <f32>
            ascvec.mem_bar vst_vld
            %c0_160 = arith.constant 0 : index
            %c1_161 = arith.constant 1 : index
            scf.for %arg35 = %c0_160 to %c1_161 step %c1_161 {
              %81 = ascvec.loadalign %arg34[%c0_159] : memref<128xf32, 1>, index -> <f32>
              %82 = ascvec.reduce_max %81, %79 : <f32>
              %83 = ascvec.duplicate %82, %79 : <f32> -> <f32>
              %84 = ascvec.loadalign %arg34[%c64] : memref<128xf32, 1>, index -> <f32>
              %85 = ascvec.reduce_max %84, %79 : <f32>
              %86 = ascvec.duplicate %85, %79 : <f32> -> <f32>
              ascvec.storealign_1st %arg33[%c0_159], %83 : memref<8xf32, 1>, index, <f32>
              ascvec.storealign_1st %arg33[%c1], %86 : memref<8xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c6_i64, 0 {cannir.buffer.generated_sync}
          %67 = arith.minsi %51, %66 : i64
          %68 = arith.subi %66, %67 : i64
          %69 = arith.minsi %68, %c1_i64 : i64
          %70 = arith.cmpi sle, %51, %66 : i64
          %71 = arith.select %70, %69, %c0_i64 : i64
          %72 = arith.muli %51, %c256_i64 : i64
          %73 = ascscalar.memref_address %alloc_16 : memref<1xf32>
          %74 = arith.divui %72, %c8_i64 : i64
          %75 = arith.addi %73, %74 : i64
          %76 = ascscalar.bind_gm_ptr %75 : memref<1xf32>
          %77 = ascscalar.bitcast_memref %alloc_49 byte_offset = %c0_i64 : memref<8xf32, 1> -> memref<?xf32, 1>
          %78 = arith.cmpi ugt, %71, %c0_i64 : i64
          scf.if %78 {
            ascvec.get_buf mte3, %c6_i64, 0 {cannir.buffer.generated_sync}
            %79 = arith.trunci %71 : i64 to i32
            ascvec.copy_ub2gm %76[%c0], %77[%c0], %79, %c32_i32, %c32_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
            ascvec.release_buf mte3, %c6_i64, 0 {cannir.buffer.generated_sync}
          }
        } else {
          "ascvec.copy_gm2ub_plan"(%alloc_54, %alloc_3, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c4096_i64, %c0_i64, %c4096_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 4096 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 16384 : i64, ub_capacity = 4096 : i64}> : (memref<1024xf32, 1>, memref<1024xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.get_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%alloc_55, %alloc_0, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c4096_i64, %c0_i64, %c4096_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 4096 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 20480 : i64, ub_capacity = 4096 : i64}> : (memref<1024xf32, 1>, memref<1024xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c5_i64, 0 {cannir.buffer.generated_sync}
          %48 = memref.load %alloc[%c0] : memref<1xf32>
          %49 = arith.muli %0, %c2_i64 : i64
          %50 = ascscalar.get_subblock_id
          %51 = arith.addi %49, %50 : i64
          %52 = cannir.extract_layout %arg18 : !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
          %53 = cannir.extract_shape %52 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
          %54 = cannir.extract_dim %53[0] : !cannir.shape<(-1, 1024)>
          %55 = scf.for %arg33 = %51 to %54 step %5 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
            %56 = arith.cmpi eq, %arg34, %c0_i64 : i64
            %57 = ascscalar.memref_address %alloc_50 : memref<1024xf32, 1>
            %58 = ascscalar.memref_address %alloc_51 : memref<1024xf32, 1>
            %59 = arith.select %56, %57, %58 : i64
            %60 = arith.cmpi ne, %arg34, %c0_i64 : i64
            %61 = arith.extui %60 : i1 to i64
            %62 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
            %63 = arith.remui %62, %c2_i64 : i64
            %64 = cannir.extract_layout %arg16 : !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
            %65 = cannir.extract_shape %64 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
            %66 = cannir.extract_dim %65[0] : !cannir.shape<(-1, 1024)>
            %67 = arith.minsi %arg33, %66 : i64
            %68 = arith.subi %66, %67 : i64
            %69 = arith.minsi %68, %c1_i64 : i64
            %70 = arith.cmpi sle, %arg33, %66 : i64
            %71 = arith.select %70, %69, %c0_i64 : i64
            %72 = arith.muli %arg33, %c4096_i64 : i64
            %73 = arith.muli %71, %c4096_i64 : i64
            %74 = arith.cmpi eq, %73, %c0_i64 : i64
            %75 = arith.cmpi ne, %73, %c0_i64 : i64
            %76 = arith.extui %75 : i1 to i64
            %77 = arith.select %74, %c4_i64, %73 : i64
            %78 = ascscalar.bind_local_ptr %59 : memref<1xf32, 1>
            ascvec.get_buf mte2, %61, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%78, %alloc_10, %c0_i32, %c0_i32, %c0_i64, %72, %76, %77, %c0_i64, %77, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %61, 0 {cannir.buffer.generated_sync}
            %79 = ascscalar.bind_local_ptr %59 : memref<1024xf32, 1>
            %80 = ascscalar.memref_address %alloc_52 : memref<1024xf32, 1>
            %81 = ascscalar.memref_address %alloc_53 : memref<1024xf32, 1>
            %82 = arith.select %56, %80, %81 : i64
            %83 = ascscalar.bind_local_ptr %82 : memref<1024xf32, 1>
            %84 = arith.select %56, %c2_i64, %c3_i64 : i64
            ascvec.get_buf v, %61, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%79, %alloc_55, %48 : memref<1024xf32, 1>, memref<1024xf32, 1>, f32) {
            ^bb0(%arg35: memref<1024xf32, 1>, %arg36: memref<1024xf32, 1>, %arg37: f32):
              %c64_i32 = arith.constant 64 : i32
              %c1024_i32_159 = arith.constant 1024 : i32
              %c0_i32_160 = arith.constant 0 : i32
              %97 = ascvec.create_mask : !ascvec.mask
              scf.for %arg38 = %c0_i32_160 to %c1024_i32_159 step %c64_i32  : i32 {
                %98 = arith.index_cast %arg38 : i32 to index
                %99 = ascvec.loadalign %arg35[%98] : memref<1024xf32, 1>, index -> <f32>
                %100 = ascvec.loadalign %arg36[%98] : memref<1024xf32, 1>, index -> <f32>
                %101 = ascvec.mul %99, %100, %97 : <f32>
                %102 = ascvec.mul_scalar %101, %arg37, %97 : <f32>, f32
                ascvec.storealign %arg35[%98], %102, %97 : memref<1024xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c5_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %84, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%79, %arg29, %alloc_54, %arg30, %83 : memref<1024xf32, 1>, f32, memref<1024xf32, 1>, f32, memref<1024xf32, 1>) {
            ^bb0(%arg35: memref<1024xf32, 1>, %arg36: f32, %arg37: memref<1024xf32, 1>, %arg38: f32, %arg39: memref<1024xf32, 1>):
              %cst = arith.constant 1.024000e+03 : f32
              %c64_i32 = arith.constant 64 : i32
              %c1024_i32_159 = arith.constant 1024 : i32
              %c0_i32_160 = arith.constant 0 : i32
              %cst_161 = arith.constant 0.000000e+00 : f32
              %97 = ascvec.create_mask : !ascvec.mask
              ascvec.mem_bar vst_vld
              %98 = ascvec.duplicate_scalar %cst_161, %97 : f32 -> <f32>
              %99 = scf.for %arg40 = %c0_i32_160 to %c1024_i32_159 step %c64_i32 iter_args(%arg41 = %98) -> (!ascvec.vreg<f32>)  : i32 {
                %105 = arith.index_cast %arg40 : i32 to index
                %106 = ascvec.loadalign %arg35[%105] : memref<1024xf32, 1>, index -> <f32>
                %107 = ascvec.mul %106, %106, %97 : <f32>
                %108 = ascvec.reduce_sum %107, %97 : <f32>
                %109 = ascvec.add %arg41, %108, %97 : <f32>
                scf.yield %109 : !ascvec.vreg<f32>
              }
              %100 = ascvec.duplicate %99, %97 : <f32> -> <f32>
              %101 = ascvec.duplicate_scalar %cst, %97 : f32 -> <f32>
              %102 = ascvec.div %100, %101, %97 {precision_mode = true} : <f32>
              %103 = ascvec.add_scalar %102, %arg36, %97 : <f32>, f32
              %104 = ascvec.sqrt %103, %97 : <f32>
              scf.for %arg40 = %c0_i32_160 to %c1024_i32_159 step %c64_i32  : i32 {
                %105 = arith.index_cast %arg40 : i32 to index
                %106 = ascvec.loadalign %arg35[%105] : memref<1024xf32, 1>, index -> <f32>
                %107 = ascvec.div %106, %104, %97 {precision_mode = true} : <f32>
                %108 = ascvec.loadalign %arg37[%105] : memref<1024xf32, 1>, index -> <f32>
                %109 = ascvec.mul %107, %108, %97 : <f32>
                %110 = ascvec.mul_scalar %109, %arg38, %97 : <f32>, f32
                ascvec.storealign %arg39[%105], %110, %97 : memref<1024xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %84, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %61, 0 {cannir.buffer.generated_sync}
            %85 = arith.minsi %arg33, %54 : i64
            %86 = arith.subi %54, %85 : i64
            %87 = arith.minsi %86, %c1_i64 : i64
            %88 = arith.cmpi sle, %arg33, %54 : i64
            %89 = arith.select %88, %87, %c0_i64 : i64
            %90 = arith.muli %arg33, %c32768_i64 : i64
            %91 = ascscalar.memref_address %alloc_12 : memref<1xf32>
            %92 = arith.divui %90, %c8_i64 : i64
            %93 = arith.addi %91, %92 : i64
            %94 = ascscalar.bind_gm_ptr %93 : memref<1xf32>
            %95 = ascscalar.bind_local_ptr %82 : memref<1xf32, 1>
            %96 = arith.cmpi ugt, %89, %c0_i64 : i64
            scf.if %96 {
              ascvec.get_buf mte3, %84, 0 {cannir.buffer.generated_sync}
              %97 = arith.trunci %89 : i64 to i32
              ascvec.copy_ub2gm %94[%c0], %95[%c0], %97, %c4096_i32, %c4096_i64, %c4096_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
              ascvec.release_buf mte3, %84, 0 {cannir.buffer.generated_sync}
            }
            scf.yield %63 : i64
          }
        }
      }
    }
    scf.if %12 {
      ascvec.sync_all
      ascvec.sync_block_arrive mte3, %c8_i64
      ascvec.sync_block_wait s, %c7_i64
    } else {
      ascvec.sync_all
      ascvec.sync_block_arrive mte3, %c0_i64
      ascvec.sync_block_wait s, %c2_i64
    }
    ascvec.sync_all
    %13 = arith.cmpi sgt, %4, %c4_i64 : i64
    %14 = arith.xori %10, %true : i1
    %15 = arith.select %13, %14, %false : i1
    scf.if %15 {
      scf.if %7 {
        %48 = cannir.extract_layout %arg22 : !cannir.tensor<<gm, f32>, <<(-1, 384)>, <(384, 1)>>, <<(-1, 384)>, <(384, 1)>>, identity>
        %49 = cannir.extract_shape %48 : <<(-1, 384)>, <(384, 1)>> -> <(-1, 384)>
        %50 = cannir.extract_dim %49[0] : !cannir.shape<(-1, 384)>
        %51 = arith.muli %50, %c384_i64 : i64
        %52 = arith.remsi %51, %c8_i64 : i64
        %53 = arith.cmpi ne, %52, %c0_i64 : i64
        %54 = arith.cmpi slt, %52, %c0_i64 : i64
        %55 = arith.andi %53, %54 : i1
        %56 = arith.divsi %51, %c8_i64 : i64
        %57 = arith.extui %55 : i1 to i64
        %58 = arith.subi %56, %57 : i64
        %59 = arith.maxsi %5, %c1_i64 : i64
        %60 = arith.divsi %58, %59 : i64
        %61 = arith.minsi %60, %c0_i64 : i64
        %62 = arith.muli %61, %5 : i64
        %63 = arith.subi %58, %62 : i64
        %64 = arith.minsi %5, %63 : i64
        %65 = arith.cmpi sge, %60, %c0_i64 : i64
        %66 = arith.select %65, %64, %c0_i64 : i64
        %67 = arith.muli %66, %c32_i64 : i64
        %68 = arith.cmpi eq, %67, %c0_i64 : i64
        %69 = arith.cmpi ne, %67, %c0_i64 : i64
        %70 = arith.extui %69 : i1 to i64
        %71 = arith.select %68, %c4_i64, %67 : i64
        ascvec.get_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%alloc_57, %alloc_16, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %70, %71, %c0_i64, %71, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 32 : i64, ub_capacity = 4096 : i64}> : (memref<1024xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
        %72 = arith.trunci %5 : i64 to i32
        ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_57, %72, %alloc_56 : memref<1024xf32, 1>, i32, memref<1xf32, 1>) {
        ^bb0(%arg33: memref<1024xf32, 1>, %arg34: i32, %arg35: memref<1xf32, 1>):
          %c2143289344_i32 = arith.constant 2143289344 : i32
          %cst = arith.constant 3.276800e+04 : f32
          %c0_i32_159 = arith.constant 0 : i32
          %c1_i32_160 = arith.constant 1 : i32
          %c0_161 = arith.constant 0 : index
          %c8_i32 = arith.constant 8 : i32
          %cst_162 = arith.constant 0.000000e+00 : f32
          %82 = ascvec.create_mask : !ascvec.mask
          %83 = ascvec.duplicate_scalar %cst_162, %82 : f32 -> <f32>
          %84:2 = scf.for %arg36 = %c0_i32_159 to %arg34 step %c1_i32_160 iter_args(%arg37 = %83, %arg38 = %83) -> (!ascvec.vreg<f32>, !ascvec.vreg<f32>)  : i32 {
            %85 = arith.muli %arg36, %c8_i32 : i32
            %86 = arith.index_cast %85 : i32 to index
            %87 = ascvec.loadalign_brc %arg33[%86] : memref<1024xf32, 1>, index -> <f32>
            %88 = ascvec.max %arg37, %87, %82 : <f32>
            %89 = arith.addi %85, %c1_i32_160 : i32
            %90 = arith.index_cast %89 : i32 to index
            %91 = ascvec.loadalign_brc %arg33[%90] : memref<1024xf32, 1>, index -> <f32>
            %92 = ascvec.max %arg38, %91, %82 : <f32>
            scf.yield %88, %92 : !ascvec.vreg<f32>, !ascvec.vreg<f32>
          }
          %c0_163 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg36 = %c0_163 to %c1 step %c1 {
            %85 = ascvec.duplicate_scalar %cst, %82 : f32 -> <f32>
            %86 = ascvec.div %84#0, %85, %82 {precision_mode = true} : <f32>
            %87 = ascvec.duplicate_scalar %c2143289344_i32, %82 : i32 -> <ui32>
            %88 = ascvec.reinterpret %87 : <ui32> -> <f32>
            %89 = ascvec.cmp_gt_scalar %84#1, %cst_162, %82 : <f32>, f32 -> !ascvec.mask
            %90 = ascvec.select %88, %86, %89 : <f32>
            ascvec.storealign_1st %arg35[%c0_161], %90 : memref<1xf32, 1>, index, <f32>
          }
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
        %73 = arith.cmpi eq, %0, %c0_i64 : i64
        %74 = scf.if %73 -> (i1) {
          %82 = ascscalar.get_subblock_id
          %83 = arith.cmpi eq, %82, %c0_i64 : i64
          scf.yield %83 : i1
        } else {
          scf.yield %false : i1
        }
        scf.if %74 {
          %82 = ascscalar.bitcast_memref %alloc_56 byte_offset = %c0_i64 : memref<1xf32, 1> -> memref<?xf32, 1>
          ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %alloc_15[%c0], %82[%c0], %c1_i32, %c4_i32, %c0_i64, %c4_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
          ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
        }
        %75 = arith.muli %0, %c2_i64 : i64
        %76 = ascscalar.get_subblock_id
        %77 = arith.addi %75, %76 : i64
        %78 = cannir.extract_layout %arg18 : !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
        %79 = cannir.extract_shape %78 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
        %80 = cannir.extract_dim %79[0] : !cannir.shape<(-1, 1024)>
        %81 = scf.for %arg33 = %77 to %80 step %5 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
          %82 = arith.cmpi eq, %arg34, %c0_i64 : i64
          %83 = ascscalar.memref_address %alloc_58 : memref<1024xf32, 1>
          %84 = ascscalar.memref_address %alloc_59 : memref<1024xf32, 1>
          %85 = arith.select %82, %83, %84 : i64
          %86 = arith.select %82, %c2_i64, %c3_i64 : i64
          %87 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
          %88 = arith.remui %87, %c2_i64 : i64
          %89 = arith.minsi %arg33, %80 : i64
          %90 = arith.subi %80, %89 : i64
          %91 = arith.minsi %90, %c1_i64 : i64
          %92 = arith.cmpi sle, %arg33, %80 : i64
          %93 = arith.select %92, %91, %c0_i64 : i64
          %94 = arith.muli %arg33, %c4096_i64 : i64
          %95 = arith.muli %93, %c4096_i64 : i64
          %96 = arith.cmpi eq, %95, %c0_i64 : i64
          %97 = arith.cmpi ne, %95, %c0_i64 : i64
          %98 = arith.extui %97 : i1 to i64
          %99 = arith.select %96, %c4_i64, %95 : i64
          %100 = ascscalar.bind_local_ptr %85 : memref<1xf32, 1>
          ascvec.get_buf mte2, %86, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%100, %alloc_12, %c0_i32, %c0_i32, %c0_i64, %94, %98, %99, %c0_i64, %99, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %86, 0 {cannir.buffer.generated_sync}
          %101 = ascscalar.bind_local_ptr %85 : memref<1024xf32, 1>
          %102 = ascscalar.memref_address %alloc_60 : memref<1024xui8, 1>
          %103 = ascscalar.memref_address %alloc_61 : memref<1024xui8, 1>
          %104 = arith.select %82, %102, %103 : i64
          %105 = ascscalar.bind_local_ptr %104 : memref<1024xui8, 1>
          %106 = arith.select %82, %c4_i64, %c5_i64 : i64
          ascvec.get_buf v, %86, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %106, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_56, %101, %105 : memref<1xf32, 1>, memref<1024xf32, 1>, memref<1024xui8, 1>) {
          ^bb0(%arg35: memref<1xf32, 1>, %arg36: memref<1024xf32, 1>, %arg37: memref<1024xui8, 1>):
            %c64_i32 = arith.constant 64 : i32
            %c1024_i32_159 = arith.constant 1024 : i32
            %c0_i32_160 = arith.constant 0 : i32
            %cst = arith.constant 1.1920929E-7 : f32
            %cst_161 = arith.constant 3.276800e+04 : f32
            %cst_162 = arith.constant -3.276800e+04 : f32
            %cst_163 = arith.constant 1.000000e+00 : f32
            %cst_164 = arith.constant 0.000000e+00 : f32
            %c0_165 = arith.constant 0 : index
            %122 = ascvec.create_mask : !ascvec.mask
            ascvec.mem_bar vst_vld
            %123 = ascvec.loadalign_brc %arg35[%c0_165] : memref<1xf32, 1>, index -> <f32>
            %124 = ascvec.cmp_eq_scalar %123, %cst_164, %122 : <f32>, f32 -> !ascvec.mask
            %125 = ascvec.duplicate_scalar %cst_163, %122 : f32 -> <f32>
            %126 = ascvec.select %125, %123, %124 : <f32>
            scf.for %arg38 = %c0_i32_160 to %c1024_i32_159 step %c64_i32  : i32 {
              %127 = arith.index_cast %arg38 : i32 to index
              %128 = ascvec.loadalign %arg36[%127] : memref<1024xf32, 1>, index -> <f32>
              %129 = ascvec.div %128, %126, %122 {precision_mode = true} : <f32>
              %130 = ascvec.duplicate_scalar %cst_164, %122 : f32 -> <f32>
              %131 = ascvec.select %130, %129, %124 : <f32>
              %132 = ascvec.cmp_eq %131, %131, %122 : <f32>, <f32> -> !ascvec.mask
              %133 = ascvec.select %131, %130, %132 : <f32>
              %134 = ascvec.max_scalar %133, %cst_162, %122 : <f32>, f32
              %135 = ascvec.min_scalar %134, %cst_161, %122 : <f32>, f32
              %136 = ascvec.abs %135, %122 : <f32>
              %137 = ascvec.cmp_lt_scalar %136, %cst, %122 : <f32>, f32 -> !ascvec.mask
              %138 = ascvec.select %130, %135, %137 : <f32>
              %139 = ascvec.cast %138, %122 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
              %140 = ascvec.reinterpret %139 : <!ascdtype.hifloat8> -> <f8E4M3FN>
              ascvec.storealign_pack %arg37[%127], %140, %122 {mode = "b32_to_b8"} : memref<1024xui8, 1>, index, <f8E4M3FN>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %106, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %86, 0 {cannir.buffer.generated_sync}
          %107 = cannir.extract_layout %arg20 : !cannir.tensor<<gm, ui8>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
          %108 = cannir.extract_shape %107 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
          %109 = cannir.extract_dim %108[0] : !cannir.shape<(-1, 1024)>
          %110 = arith.minsi %arg33, %109 : i64
          %111 = arith.subi %109, %110 : i64
          %112 = arith.minsi %111, %c1_i64 : i64
          %113 = arith.cmpi sle, %arg33, %109 : i64
          %114 = arith.select %113, %112, %c0_i64 : i64
          %115 = arith.muli %arg33, %c8192_i64 : i64
          %116 = ascscalar.memref_address %alloc_14 : memref<1xui8>
          %117 = arith.divui %115, %c8_i64 : i64
          %118 = arith.addi %116, %117 : i64
          %119 = ascscalar.bind_gm_ptr %118 : memref<1xui8>
          %120 = ascscalar.bind_local_ptr %104 : memref<1xui8, 1>
          %121 = arith.cmpi ugt, %114, %c0_i64 : i64
          scf.if %121 {
            ascvec.get_buf mte3, %106, 0 {cannir.buffer.generated_sync}
            %122 = arith.trunci %114 : i64 to i32
            ascvec.copy_ub2gm %119[%c0], %120[%c0], %122, %c1024_i32, %c1024_i64, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
            ascvec.release_buf mte3, %106, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %88 : i64
        }
      } else {
        %48 = cannir.extract_layout %arg18 : !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
        %49 = cannir.extract_shape %48 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
        %50 = cannir.extract_dim %49[0] : !cannir.shape<(-1, 1024)>
        %51 = arith.cmpi sge, %50, %c16_i64 : i64
        scf.if %51 {
          %56 = arith.muli %0, %c2_i64 : i64
          %57 = ascscalar.get_subblock_id
          %58 = arith.addi %56, %57 : i64
          %59 = arith.cmpi sgt, %5, %50 : i64
          %60 = arith.select %59, %50, %5 : i64
          %61 = arith.cmpi sgt, %60, %c128_i64 : i64
          %62 = arith.select %61, %c128_i64, %60 : i64
          %63 = cannir.extract_layout %arg16 : !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
          %64 = cannir.extract_shape %63 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
          %65 = cannir.extract_dim %64[0] : !cannir.shape<(-1, 1024)>
          %66 = arith.muli %65, %c1024_i64 : i64
          %67 = arith.remsi %66, %c8_i64 : i64
          %68 = arith.cmpi ne, %67, %c0_i64 : i64
          %69 = arith.cmpi slt, %67, %c0_i64 : i64
          %70 = arith.andi %68, %69 : i1
          %71 = arith.divsi %66, %c8_i64 : i64
          %72 = arith.extui %70 : i1 to i64
          %73 = arith.subi %71, %72 : i64
          %74 = arith.cmpi slt, %58, %62 : i64
          scf.if %74 {
            ascvec.vf(%alloc_64 : memref<128xf32, 1>) {
            ^bb0(%arg33: memref<128xf32, 1>):
              %c64 = arith.constant 64 : index
              %c0_159 = arith.constant 0 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c0_160 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg34 = %c0_160 to %c1 step %c1 {
                %89 = ascvec.create_mask : !ascvec.mask
                %90 = ascvec.duplicate_scalar %cst, %89 : f32 -> <f32>
                ascvec.storealign %arg33[%c0_159], %90, %89 : memref<128xf32, 1>, index, <f32>
                ascvec.storealign %arg33[%c64], %90, %89 : memref<128xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            %76 = scf.for %arg33 = %58 to %50 step %62 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
              %89 = arith.cmpi eq, %arg34, %c0_i64 : i64
              %90 = ascscalar.memref_address %alloc_62 : memref<1024xf32, 1>
              %91 = ascscalar.memref_address %alloc_63 : memref<1024xf32, 1>
              %92 = arith.select %89, %90, %91 : i64
              %93 = arith.select %89, %c6_i64, %c7_i64 : i64
              %94 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
              %95 = arith.remui %94, %c2_i64 : i64
              %96 = arith.minsi %arg33, %50 : i64
              %97 = arith.subi %50, %96 : i64
              %98 = arith.minsi %97, %c1_i64 : i64
              %99 = arith.cmpi sle, %arg33, %50 : i64
              %100 = arith.select %99, %98, %c0_i64 : i64
              %101 = arith.muli %arg33, %c4096_i64 : i64
              %102 = arith.muli %100, %c4096_i64 : i64
              %103 = arith.cmpi eq, %102, %c0_i64 : i64
              %104 = arith.cmpi ne, %102, %c0_i64 : i64
              %105 = arith.extui %104 : i1 to i64
              %106 = arith.select %103, %c4_i64, %102 : i64
              %107 = ascscalar.bind_local_ptr %92 : memref<1xf32, 1>
              ascvec.get_buf mte2, %93, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%107, %alloc_12, %c0_i32, %c0_i32, %c0_i64, %101, %105, %106, %c0_i64, %106, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %93, 0 {cannir.buffer.generated_sync}
              %108 = ascscalar.bind_local_ptr %92 : memref<1024xf32, 1>
              ascvec.get_buf v, %93, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_64, %108 : memref<128xf32, 1>, memref<1024xf32, 1>) {
              ^bb0(%arg35: memref<128xf32, 1>, %arg36: memref<1024xf32, 1>):
                %c64 = arith.constant 64 : index
                %c64_i32 = arith.constant 64 : i32
                %c1024_i32_159 = arith.constant 1024 : i32
                %c0_i32_160 = arith.constant 0 : i32
                %cst = arith.constant 0.000000e+00 : f32
                %cst_161 = arith.constant 1.000000e+00 : f32
                %c0_162 = arith.constant 0 : index
                %109 = ascvec.create_mask : !ascvec.mask
                ascvec.mem_bar vst_vld
                %110 = ascvec.loadalign %arg35[%c0_162] : memref<128xf32, 1>, index -> <f32>
                %111 = ascvec.loadalign %arg35[%c64] : memref<128xf32, 1>, index -> <f32>
                %112 = ascvec.duplicate_scalar %cst_161, %109 : f32 -> <f32>
                %113 = ascvec.duplicate_scalar %cst, %109 : f32 -> <f32>
                %114:2 = scf.for %arg37 = %c0_i32_160 to %c1024_i32_159 step %c64_i32 iter_args(%arg38 = %110, %arg39 = %111) -> (!ascvec.vreg<f32>, !ascvec.vreg<f32>)  : i32 {
                  %115 = arith.index_cast %arg37 : i32 to index
                  %116 = ascvec.loadalign %arg36[%115] : memref<1024xf32, 1>, index -> <f32>
                  %117 = ascvec.abs %116, %109 : <f32>
                  %118 = ascvec.max %arg38, %117, %109 : <f32>
                  %119 = ascvec.cmp_ne %116, %116, %109 : <f32>, <f32> -> !ascvec.mask
                  %120 = ascvec.select %112, %113, %119 : <f32>
                  %121 = ascvec.max %arg39, %120, %109 : <f32>
                  scf.yield %118, %121 : !ascvec.vreg<f32>, !ascvec.vreg<f32>
                }
                %c0_163 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg37 = %c0_163 to %c1 step %c1 {
                  ascvec.storealign %arg35[%c0_162], %114#0, %109 : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign %arg35[%c64], %114#1, %109 : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %93, 0 {cannir.buffer.generated_sync}
              scf.yield %95 : i64
            }
            ascvec.get_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_65, %alloc_64 : memref<8xf32, 1>, memref<128xf32, 1>) {
            ^bb0(%arg33: memref<8xf32, 1>, %arg34: memref<128xf32, 1>):
              %c1 = arith.constant 1 : index
              %c64 = arith.constant 64 : index
              %c0_159 = arith.constant 0 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c8_i32 = arith.constant 8 : i32
              %89 = ascvec.create_mask : !ascvec.mask
              %mask, %remaining_out = ascvec.update_mask %c8_i32 : !ascvec.mask
              %90 = ascvec.duplicate_scalar %cst, %mask : f32 -> <f32>
              ascvec.storealign %arg33[%c0_159], %90, %mask : memref<8xf32, 1>, index, <f32>
              ascvec.mem_bar vst_vld
              %c0_160 = arith.constant 0 : index
              %c1_161 = arith.constant 1 : index
              scf.for %arg35 = %c0_160 to %c1_161 step %c1_161 {
                %91 = ascvec.loadalign %arg34[%c0_159] : memref<128xf32, 1>, index -> <f32>
                %92 = ascvec.reduce_max %91, %89 : <f32>
                %93 = ascvec.duplicate %92, %89 : <f32> -> <f32>
                %94 = ascvec.loadalign %arg34[%c64] : memref<128xf32, 1>, index -> <f32>
                %95 = ascvec.reduce_max %94, %89 : <f32>
                %96 = ascvec.duplicate %95, %89 : <f32> -> <f32>
                ascvec.storealign_1st %arg33[%c0_159], %93 : memref<8xf32, 1>, index, <f32>
                ascvec.storealign_1st %arg33[%c1], %96 : memref<8xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
            %77 = arith.minsi %58, %73 : i64
            %78 = arith.subi %73, %77 : i64
            %79 = arith.minsi %78, %c1_i64 : i64
            %80 = arith.cmpi sle, %58, %73 : i64
            %81 = arith.select %80, %79, %c0_i64 : i64
            %82 = arith.muli %58, %c256_i64 : i64
            %83 = ascscalar.memref_address %alloc_10 : memref<1xf32>
            %84 = arith.divui %82, %c8_i64 : i64
            %85 = arith.addi %83, %84 : i64
            %86 = ascscalar.bind_gm_ptr %85 : memref<1xf32>
            %87 = ascscalar.bitcast_memref %alloc_65 byte_offset = %c0_i64 : memref<8xf32, 1> -> memref<?xf32, 1>
            %88 = arith.cmpi ugt, %81, %c0_i64 : i64
            scf.if %88 {
              ascvec.get_buf mte3, %c8_i64, 0 {cannir.buffer.generated_sync}
              %89 = arith.trunci %81 : i64 to i32
              ascvec.copy_ub2gm %86[%c0], %87[%c0], %89, %c32_i32, %c32_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
              ascvec.release_buf mte3, %c8_i64, 0 {cannir.buffer.generated_sync}
            }
          }
          ascvec.sync_all
          ascvec.sync_block_arrive mte3, %c0_i64
          ascvec.sync_block_wait s, %c2_i64
          %75 = arith.cmpi eq, %58, %c0_i64 : i64
          scf.if %75 {
            %76 = arith.maxsi %62, %c1_i64 : i64
            %77 = arith.divsi %73, %76 : i64
            %78 = arith.minsi %77, %c0_i64 : i64
            %79 = arith.muli %78, %62 : i64
            %80 = arith.subi %73, %79 : i64
            %81 = arith.minsi %62, %80 : i64
            %82 = arith.cmpi sge, %77, %c0_i64 : i64
            %83 = arith.select %82, %81, %c0_i64 : i64
            %84 = arith.muli %83, %c32_i64 : i64
            %85 = arith.cmpi eq, %84, %c0_i64 : i64
            %86 = arith.cmpi ne, %84, %c0_i64 : i64
            %87 = arith.extui %86 : i1 to i64
            %88 = arith.select %85, %c4_i64, %84 : i64
            ascvec.get_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%alloc_67, %alloc_10, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %87, %88, %c0_i64, %88, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 32 : i64, ub_capacity = 4096 : i64}> : (memref<1024xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
            %89 = arith.trunci %62 : i64 to i32
            ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_67, %89, %alloc_66 : memref<1024xf32, 1>, i32, memref<1xf32, 1>) {
            ^bb0(%arg33: memref<1024xf32, 1>, %arg34: i32, %arg35: memref<1xf32, 1>):
              %c2143289344_i32 = arith.constant 2143289344 : i32
              %cst = arith.constant 3.276800e+04 : f32
              %c0_i32_159 = arith.constant 0 : i32
              %c1_i32_160 = arith.constant 1 : i32
              %c0_161 = arith.constant 0 : index
              %c8_i32 = arith.constant 8 : i32
              %cst_162 = arith.constant 0.000000e+00 : f32
              %91 = ascvec.create_mask : !ascvec.mask
              %92 = ascvec.duplicate_scalar %cst_162, %91 : f32 -> <f32>
              %93:2 = scf.for %arg36 = %c0_i32_159 to %arg34 step %c1_i32_160 iter_args(%arg37 = %92, %arg38 = %92) -> (!ascvec.vreg<f32>, !ascvec.vreg<f32>)  : i32 {
                %94 = arith.muli %arg36, %c8_i32 : i32
                %95 = arith.index_cast %94 : i32 to index
                %96 = ascvec.loadalign_brc %arg33[%95] : memref<1024xf32, 1>, index -> <f32>
                %97 = ascvec.max %arg37, %96, %91 : <f32>
                %98 = arith.addi %94, %c1_i32_160 : i32
                %99 = arith.index_cast %98 : i32 to index
                %100 = ascvec.loadalign_brc %arg33[%99] : memref<1024xf32, 1>, index -> <f32>
                %101 = ascvec.max %arg38, %100, %91 : <f32>
                scf.yield %97, %101 : !ascvec.vreg<f32>, !ascvec.vreg<f32>
              }
              %c0_163 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg36 = %c0_163 to %c1 step %c1 {
                %94 = ascvec.duplicate_scalar %cst, %91 : f32 -> <f32>
                %95 = ascvec.div %93#0, %94, %91 {precision_mode = true} : <f32>
                %96 = ascvec.duplicate_scalar %c2143289344_i32, %91 : i32 -> <ui32>
                %97 = ascvec.reinterpret %96 : <ui32> -> <f32>
                %98 = ascvec.cmp_gt_scalar %93#1, %cst_162, %91 : <f32>, f32 -> !ascvec.mask
                %99 = ascvec.select %97, %95, %98 : <f32>
                ascvec.storealign_1st %arg35[%c0_161], %99 : memref<1xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
            %90 = ascscalar.bitcast_memref %alloc_66 byte_offset = %c0_i64 : memref<1xf32, 1> -> memref<?xf32, 1>
            ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
            ascvec.copy_ub2gm %alloc_15[%c0], %90[%c0], %c1_i32, %c4_i32, %c0_i64, %c4_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
            ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
          }
        } else {
          %56 = arith.cmpi eq, %0, %c0_i64 : i64
          %57 = scf.if %56 -> (i1) {
            %58 = ascscalar.get_subblock_id
            %59 = arith.cmpi eq, %58, %c0_i64 : i64
            scf.yield %59 : i1
          } else {
            scf.yield %false : i1
          }
          scf.if %57 {
            ascvec.vf(%alloc_70 : memref<128xf32, 1>) {
            ^bb0(%arg33: memref<128xf32, 1>):
              %c64 = arith.constant 64 : index
              %c0_159 = arith.constant 0 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c0_160 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg34 = %c0_160 to %c1 step %c1 {
                %60 = ascvec.create_mask : !ascvec.mask
                %61 = ascvec.duplicate_scalar %cst, %60 : f32 -> <f32>
                ascvec.storealign %arg33[%c0_159], %61, %60 : memref<128xf32, 1>, index, <f32>
                ascvec.storealign %arg33[%c64], %61, %60 : memref<128xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            %58 = scf.for %arg33 = %c0_i64 to %50 step %c1_i64 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
              %60 = arith.cmpi eq, %arg34, %c0_i64 : i64
              %61 = ascscalar.memref_address %alloc_68 : memref<1024xf32, 1>
              %62 = ascscalar.memref_address %alloc_69 : memref<1024xf32, 1>
              %63 = arith.select %60, %61, %62 : i64
              %64 = arith.select %60, %c2_i64, %c3_i64 : i64
              %65 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
              %66 = arith.remui %65, %c2_i64 : i64
              %67 = arith.minsi %arg33, %50 : i64
              %68 = arith.subi %50, %67 : i64
              %69 = arith.minsi %68, %c1_i64 : i64
              %70 = arith.cmpi sle, %arg33, %50 : i64
              %71 = arith.select %70, %69, %c0_i64 : i64
              %72 = arith.muli %arg33, %c4096_i64 : i64
              %73 = arith.muli %71, %c4096_i64 : i64
              %74 = arith.cmpi eq, %73, %c0_i64 : i64
              %75 = arith.cmpi ne, %73, %c0_i64 : i64
              %76 = arith.extui %75 : i1 to i64
              %77 = arith.select %74, %c4_i64, %73 : i64
              %78 = ascscalar.bind_local_ptr %63 : memref<1xf32, 1>
              ascvec.get_buf mte2, %64, 0 {cannir.buffer.generated_sync}
              "ascvec.copy_gm2ub_plan"(%78, %alloc_12, %c0_i32, %c0_i32, %c0_i64, %72, %76, %77, %c0_i64, %77, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
              ascvec.release_buf mte2, %64, 0 {cannir.buffer.generated_sync}
              %79 = ascscalar.bind_local_ptr %63 : memref<1024xf32, 1>
              ascvec.get_buf v, %64, 0 {cannir.buffer.generated_sync}
              ascvec.vf(%alloc_70, %79 : memref<128xf32, 1>, memref<1024xf32, 1>) {
              ^bb0(%arg35: memref<128xf32, 1>, %arg36: memref<1024xf32, 1>):
                %c64 = arith.constant 64 : index
                %c64_i32 = arith.constant 64 : i32
                %c1024_i32_159 = arith.constant 1024 : i32
                %c0_i32_160 = arith.constant 0 : i32
                %cst = arith.constant 0.000000e+00 : f32
                %cst_161 = arith.constant 1.000000e+00 : f32
                %c0_162 = arith.constant 0 : index
                %80 = ascvec.create_mask : !ascvec.mask
                ascvec.mem_bar vst_vld
                %81 = ascvec.loadalign %arg35[%c0_162] : memref<128xf32, 1>, index -> <f32>
                %82 = ascvec.loadalign %arg35[%c64] : memref<128xf32, 1>, index -> <f32>
                %83 = ascvec.duplicate_scalar %cst_161, %80 : f32 -> <f32>
                %84 = ascvec.duplicate_scalar %cst, %80 : f32 -> <f32>
                %85:2 = scf.for %arg37 = %c0_i32_160 to %c1024_i32_159 step %c64_i32 iter_args(%arg38 = %81, %arg39 = %82) -> (!ascvec.vreg<f32>, !ascvec.vreg<f32>)  : i32 {
                  %86 = arith.index_cast %arg37 : i32 to index
                  %87 = ascvec.loadalign %arg36[%86] : memref<1024xf32, 1>, index -> <f32>
                  %88 = ascvec.abs %87, %80 : <f32>
                  %89 = ascvec.max %arg38, %88, %80 : <f32>
                  %90 = ascvec.cmp_ne %87, %87, %80 : <f32>, <f32> -> !ascvec.mask
                  %91 = ascvec.select %83, %84, %90 : <f32>
                  %92 = ascvec.max %arg39, %91, %80 : <f32>
                  scf.yield %89, %92 : !ascvec.vreg<f32>, !ascvec.vreg<f32>
                }
                %c0_163 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg37 = %c0_163 to %c1 step %c1 {
                  ascvec.storealign %arg35[%c0_162], %85#0, %80 : memref<128xf32, 1>, index, <f32>
                  ascvec.storealign %arg35[%c64], %85#1, %80 : memref<128xf32, 1>, index, <f32>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %64, 0 {cannir.buffer.generated_sync}
              scf.yield %66 : i64
            }
            ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_70, %alloc_71 : memref<128xf32, 1>, memref<1xf32, 1>) {
            ^bb0(%arg33: memref<128xf32, 1>, %arg34: memref<1xf32, 1>):
              %c64 = arith.constant 64 : index
              %cst = arith.constant 0.000000e+00 : f32
              %c2143289344_i32 = arith.constant 2143289344 : i32
              %cst_159 = arith.constant 3.276800e+04 : f32
              %c0_160 = arith.constant 0 : index
              %60 = ascvec.create_mask : !ascvec.mask
              ascvec.mem_bar vst_vld
              %c0_161 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg35 = %c0_161 to %c1 step %c1 {
                %61 = ascvec.loadalign %arg33[%c0_160] : memref<128xf32, 1>, index -> <f32>
                %62 = ascvec.reduce_max %61, %60 : <f32>
                %63 = ascvec.duplicate %62, %60 : <f32> -> <f32>
                %64 = ascvec.loadalign %arg33[%c64] : memref<128xf32, 1>, index -> <f32>
                %65 = ascvec.reduce_max %64, %60 : <f32>
                %66 = ascvec.duplicate %65, %60 : <f32> -> <f32>
                %67 = ascvec.duplicate_scalar %cst_159, %60 : f32 -> <f32>
                %68 = ascvec.div %63, %67, %60 {precision_mode = true} : <f32>
                %69 = ascvec.duplicate_scalar %c2143289344_i32, %60 : i32 -> <ui32>
                %70 = ascvec.reinterpret %69 : <ui32> -> <f32>
                %71 = ascvec.cmp_gt_scalar %66, %cst, %60 : <f32>, f32 -> !ascvec.mask
                %72 = ascvec.select %70, %68, %71 : <f32>
                ascvec.storealign_1st %arg34[%c0_160], %72 : memref<1xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
            %59 = ascscalar.bitcast_memref %alloc_71 byte_offset = %c0_i64 : memref<1xf32, 1> -> memref<?xf32, 1>
            ascvec.get_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
            ascvec.copy_ub2gm %alloc_15[%c0], %59[%c0], %c1_i32, %c4_i32, %c0_i64, %c4_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
            ascvec.release_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
          }
        }
        ascvec.sync_all
        ascvec.sync_block_arrive mte3, %c0_i64
        ascvec.sync_block_wait s, %c2_i64
        ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%alloc_72, %alloc_15, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c4_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 4 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
        %52 = arith.muli %0, %c2_i64 : i64
        %53 = ascscalar.get_subblock_id
        %54 = arith.addi %52, %53 : i64
        %55 = scf.for %arg33 = %54 to %50 step %5 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
          %56 = arith.cmpi eq, %arg34, %c0_i64 : i64
          %57 = ascscalar.memref_address %alloc_73 : memref<1024xf32, 1>
          %58 = ascscalar.memref_address %alloc_74 : memref<1024xf32, 1>
          %59 = arith.select %56, %57, %58 : i64
          %60 = arith.select %56, %c1_i64, %c2_i64 : i64
          %61 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
          %62 = arith.remui %61, %c2_i64 : i64
          %63 = arith.minsi %arg33, %50 : i64
          %64 = arith.subi %50, %63 : i64
          %65 = arith.minsi %64, %c1_i64 : i64
          %66 = arith.cmpi sle, %arg33, %50 : i64
          %67 = arith.select %66, %65, %c0_i64 : i64
          %68 = arith.muli %arg33, %c4096_i64 : i64
          %69 = arith.muli %67, %c4096_i64 : i64
          %70 = arith.cmpi eq, %69, %c0_i64 : i64
          %71 = arith.cmpi ne, %69, %c0_i64 : i64
          %72 = arith.extui %71 : i1 to i64
          %73 = arith.select %70, %c4_i64, %69 : i64
          %74 = ascscalar.bind_local_ptr %59 : memref<1xf32, 1>
          ascvec.get_buf mte2, %60, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%74, %alloc_12, %c0_i32, %c0_i32, %c0_i64, %68, %72, %73, %c0_i64, %73, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %60, 0 {cannir.buffer.generated_sync}
          %75 = ascscalar.bind_local_ptr %59 : memref<1024xf32, 1>
          %76 = ascscalar.memref_address %alloc_75 : memref<1024xui8, 1>
          %77 = ascscalar.memref_address %alloc_76 : memref<1024xui8, 1>
          %78 = arith.select %56, %76, %77 : i64
          %79 = ascscalar.bind_local_ptr %78 : memref<1024xui8, 1>
          %80 = arith.select %56, %c3_i64, %c4_i64 : i64
          ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %60, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %80, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_72, %75, %79 : memref<1xf32, 1>, memref<1024xf32, 1>, memref<1024xui8, 1>) {
          ^bb0(%arg35: memref<1xf32, 1>, %arg36: memref<1024xf32, 1>, %arg37: memref<1024xui8, 1>):
            %c64_i32 = arith.constant 64 : i32
            %c1024_i32_159 = arith.constant 1024 : i32
            %c0_i32_160 = arith.constant 0 : i32
            %cst = arith.constant 1.1920929E-7 : f32
            %cst_161 = arith.constant 3.276800e+04 : f32
            %cst_162 = arith.constant -3.276800e+04 : f32
            %cst_163 = arith.constant 1.000000e+00 : f32
            %cst_164 = arith.constant 0.000000e+00 : f32
            %c0_165 = arith.constant 0 : index
            %96 = ascvec.create_mask : !ascvec.mask
            ascvec.mem_bar vst_vld
            %97 = ascvec.loadalign_brc %arg35[%c0_165] : memref<1xf32, 1>, index -> <f32>
            %98 = ascvec.cmp_eq_scalar %97, %cst_164, %96 : <f32>, f32 -> !ascvec.mask
            %99 = ascvec.duplicate_scalar %cst_163, %96 : f32 -> <f32>
            %100 = ascvec.select %99, %97, %98 : <f32>
            scf.for %arg38 = %c0_i32_160 to %c1024_i32_159 step %c64_i32  : i32 {
              %101 = arith.index_cast %arg38 : i32 to index
              %102 = ascvec.loadalign %arg36[%101] : memref<1024xf32, 1>, index -> <f32>
              %103 = ascvec.div %102, %100, %96 {precision_mode = true} : <f32>
              %104 = ascvec.duplicate_scalar %cst_164, %96 : f32 -> <f32>
              %105 = ascvec.select %104, %103, %98 : <f32>
              %106 = ascvec.cmp_eq %105, %105, %96 : <f32>, <f32> -> !ascvec.mask
              %107 = ascvec.select %105, %104, %106 : <f32>
              %108 = ascvec.max_scalar %107, %cst_162, %96 : <f32>, f32
              %109 = ascvec.min_scalar %108, %cst_161, %96 : <f32>, f32
              %110 = ascvec.abs %109, %96 : <f32>
              %111 = ascvec.cmp_lt_scalar %110, %cst, %96 : <f32>, f32 -> !ascvec.mask
              %112 = ascvec.select %104, %109, %111 : <f32>
              %113 = ascvec.cast %112, %96 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
              %114 = ascvec.reinterpret %113 : <!ascdtype.hifloat8> -> <f8E4M3FN>
              ascvec.storealign_pack %arg37[%101], %114, %96 {mode = "b32_to_b8"} : memref<1024xui8, 1>, index, <f8E4M3FN>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %80, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %60, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          %81 = cannir.extract_layout %arg20 : !cannir.tensor<<gm, ui8>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
          %82 = cannir.extract_shape %81 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
          %83 = cannir.extract_dim %82[0] : !cannir.shape<(-1, 1024)>
          %84 = arith.minsi %arg33, %83 : i64
          %85 = arith.subi %83, %84 : i64
          %86 = arith.minsi %85, %c1_i64 : i64
          %87 = arith.cmpi sle, %arg33, %83 : i64
          %88 = arith.select %87, %86, %c0_i64 : i64
          %89 = arith.muli %arg33, %c8192_i64 : i64
          %90 = ascscalar.memref_address %alloc_14 : memref<1xui8>
          %91 = arith.divui %89, %c8_i64 : i64
          %92 = arith.addi %90, %91 : i64
          %93 = ascscalar.bind_gm_ptr %92 : memref<1xui8>
          %94 = ascscalar.bind_local_ptr %78 : memref<1xui8, 1>
          %95 = arith.cmpi ugt, %88, %c0_i64 : i64
          scf.if %95 {
            ascvec.get_buf mte3, %80, 0 {cannir.buffer.generated_sync}
            %96 = arith.trunci %88 : i64 to i32
            ascvec.copy_ub2gm %93[%c0], %94[%c0], %96, %c1024_i32, %c1024_i64, %c1024_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
            ascvec.release_buf mte3, %80, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %62 : i64
        }
      }
      ascvec.sync_all
      ascvec.sync_block_arrive mte3, %c0_i64
      ascvec.sync_block_wait s, %c2_i64
    }
    "ascvec.copy_gm2ub_plan"(%alloc_81, %alloc_4, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c2048_i64, %c0_i64, %c2048_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 2048 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 8192 : i64, ub_capacity = 2048 : i64}> : (memref<512xf32, 1>, memref<512xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
    ascvec.get_buf mte2, %c14_i64, 0 {cannir.buffer.generated_sync}
    "ascvec.copy_gm2ub_plan"(%alloc_82, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c2048_i64, %c0_i64, %c2048_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 2304 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 10240 : i64, ub_capacity = 2048 : i64}> : (memref<512xf32, 1>, memref<576xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
    ascvec.release_buf mte2, %c14_i64, 0 {cannir.buffer.generated_sync}
    %16 = memref.load %alloc[%c0] : memref<1xf32>
    %17 = arith.muli %0, %c2_i64 : i64
    %18 = ascscalar.get_subblock_id
    %19 = arith.addi %17, %18 : i64
    %20 = cannir.extract_layout %arg19 : !cannir.tensor<<gm, f32>, <<(-1, 512)>, <(512, 1)>>, <<(-1, 512)>, <(512, 1)>>, identity>
    %21 = cannir.extract_shape %20 : <<(-1, 512)>, <(512, 1)>> -> <(-1, 512)>
    %22 = cannir.extract_dim %21[0] : !cannir.shape<(-1, 512)>
    %23 = scf.for %arg33 = %19 to %22 step %5 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
      %48 = arith.cmpi eq, %arg34, %c0_i64 : i64
      %49 = ascscalar.memref_address %alloc_77 : memref<512xf32, 1>
      %50 = ascscalar.memref_address %alloc_78 : memref<512xf32, 1>
      %51 = arith.select %48, %49, %50 : i64
      %52 = arith.select %48, %c9_i64, %c10_i64 : i64
      %53 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
      %54 = arith.remui %53, %c2_i64 : i64
      %55 = cannir.extract_layout %arg17 : !cannir.tensor<<gm, f32>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>
      %56 = cannir.extract_shape %55 : <<(-1, 576)>, <(576, 1)>> -> <(-1, 576)>
      %57 = cannir.extract_dim %56[0] : !cannir.shape<(-1, 576)>
      %58 = arith.minsi %arg33, %57 : i64
      %59 = arith.subi %57, %58 : i64
      %60 = arith.minsi %59, %c1_i64 : i64
      %61 = arith.cmpi sle, %arg33, %57 : i64
      %62 = arith.select %61, %60, %c0_i64 : i64
      %63 = arith.muli %arg33, %c2304_i64 : i64
      %64 = ascscalar.bind_local_ptr %51 : memref<1xf32, 1>
      ascvec.get_buf mte2, %52, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%64, %alloc_11, %c0_i32, %c0_i32, %c0_i64, %63, %62, %c2048_i64, %c2304_i64, %c2048_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %52, 0 {cannir.buffer.generated_sync}
      %65 = ascscalar.bind_local_ptr %51 : memref<512xf32, 1>
      %66 = ascscalar.memref_address %alloc_79 : memref<512xf32, 1>
      %67 = ascscalar.memref_address %alloc_80 : memref<512xf32, 1>
      %68 = arith.select %48, %66, %67 : i64
      %69 = ascscalar.bind_local_ptr %68 : memref<512xf32, 1>
      %70 = arith.select %48, %c11_i64, %c12_i64 : i64
      ascvec.get_buf v, %52, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %c14_i64, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%65, %alloc_82, %16 : memref<512xf32, 1>, memref<512xf32, 1>, f32) {
      ^bb0(%arg35: memref<512xf32, 1>, %arg36: memref<512xf32, 1>, %arg37: f32):
        %c64_i32 = arith.constant 64 : i32
        %c512_i32 = arith.constant 512 : i32
        %c0_i32_159 = arith.constant 0 : i32
        %83 = ascvec.create_mask : !ascvec.mask
        scf.for %arg38 = %c0_i32_159 to %c512_i32 step %c64_i32  : i32 {
          %84 = arith.index_cast %arg38 : i32 to index
          %85 = ascvec.loadalign %arg35[%84] : memref<512xf32, 1>, index -> <f32>
          %86 = ascvec.loadalign %arg36[%84] : memref<512xf32, 1>, index -> <f32>
          %87 = ascvec.mul %85, %86, %83 : <f32>
          %88 = ascvec.mul_scalar %87, %arg37, %83 : <f32>, f32
          ascvec.storealign %arg35[%84], %88, %83 : memref<512xf32, 1>, index, <f32>
        }
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c14_i64, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %70, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%65, %arg29, %alloc_81, %arg31, %69 : memref<512xf32, 1>, f32, memref<512xf32, 1>, f32, memref<512xf32, 1>) {
      ^bb0(%arg35: memref<512xf32, 1>, %arg36: f32, %arg37: memref<512xf32, 1>, %arg38: f32, %arg39: memref<512xf32, 1>):
        %cst = arith.constant 5.120000e+02 : f32
        %c64_i32 = arith.constant 64 : i32
        %c512_i32 = arith.constant 512 : i32
        %c0_i32_159 = arith.constant 0 : i32
        %cst_160 = arith.constant 0.000000e+00 : f32
        %83 = ascvec.create_mask : !ascvec.mask
        ascvec.mem_bar vst_vld
        %84 = ascvec.duplicate_scalar %cst_160, %83 : f32 -> <f32>
        %85 = scf.for %arg40 = %c0_i32_159 to %c512_i32 step %c64_i32 iter_args(%arg41 = %84) -> (!ascvec.vreg<f32>)  : i32 {
          %91 = arith.index_cast %arg40 : i32 to index
          %92 = ascvec.loadalign %arg35[%91] : memref<512xf32, 1>, index -> <f32>
          %93 = ascvec.mul %92, %92, %83 : <f32>
          %94 = ascvec.reduce_sum %93, %83 : <f32>
          %95 = ascvec.add %arg41, %94, %83 : <f32>
          scf.yield %95 : !ascvec.vreg<f32>
        }
        %86 = ascvec.duplicate %85, %83 : <f32> -> <f32>
        %87 = ascvec.duplicate_scalar %cst, %83 : f32 -> <f32>
        %88 = ascvec.div %86, %87, %83 {precision_mode = true} : <f32>
        %89 = ascvec.add_scalar %88, %arg36, %83 : <f32>, f32
        %90 = ascvec.sqrt %89, %83 : <f32>
        scf.for %arg40 = %c0_i32_159 to %c512_i32 step %c64_i32  : i32 {
          %91 = arith.index_cast %arg40 : i32 to index
          %92 = ascvec.loadalign %arg35[%91] : memref<512xf32, 1>, index -> <f32>
          %93 = ascvec.div %92, %90, %83 {precision_mode = true} : <f32>
          %94 = ascvec.loadalign %arg37[%91] : memref<512xf32, 1>, index -> <f32>
          %95 = ascvec.mul %93, %94, %83 : <f32>
          %96 = ascvec.mul_scalar %95, %arg38, %83 : <f32>, f32
          ascvec.storealign %arg39[%91], %96, %83 : memref<512xf32, 1>, index, <f32>
        }
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %70, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %52, 0 {cannir.buffer.generated_sync}
      %71 = arith.minsi %arg33, %22 : i64
      %72 = arith.subi %22, %71 : i64
      %73 = arith.minsi %72, %c1_i64 : i64
      %74 = arith.cmpi sle, %arg33, %22 : i64
      %75 = arith.select %74, %73, %c0_i64 : i64
      %76 = arith.muli %arg33, %c16384_i64 : i64
      %77 = ascscalar.memref_address %alloc_13 : memref<1xf32>
      %78 = arith.divui %76, %c8_i64 : i64
      %79 = arith.addi %77, %78 : i64
      %80 = ascscalar.bind_gm_ptr %79 : memref<1xf32>
      %81 = ascscalar.bind_local_ptr %68 : memref<1xf32, 1>
      %82 = arith.cmpi ugt, %75, %c0_i64 : i64
      scf.if %82 {
        ascvec.get_buf mte3, %70, 0 {cannir.buffer.generated_sync}
        %83 = arith.trunci %75 : i64 to i32
        ascvec.copy_ub2gm %80[%c0], %81[%c0], %83, %c2048_i32, %c2048_i64, %c2048_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<1xf32, 1>
        ascvec.release_buf mte3, %70, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %54 : i64
    }
    ascvec.sync_all
    scf.if %6 {
      %48 = cannir.extract_layout %arg16 : !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
      %49 = cannir.extract_shape %48 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
      %50 = cannir.extract_dim %49[0] : !cannir.shape<(-1, 1024)>
      %51 = arith.muli %50, %c1024_i64 : i64
      %52 = arith.remsi %51, %c8_i64 : i64
      %53 = arith.cmpi ne, %52, %c0_i64 : i64
      %54 = arith.cmpi slt, %52, %c0_i64 : i64
      %55 = arith.andi %53, %54 : i1
      %56 = arith.divsi %51, %c8_i64 : i64
      %57 = arith.extui %55 : i1 to i64
      %58 = arith.subi %56, %57 : i64
      ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%alloc_83 : memref<8xf32, 1>) {
      ^bb0(%arg33: memref<8xf32, 1>):
        %c0_159 = arith.constant 0 : index
        %c8_i32 = arith.constant 8 : i32
        %cst = arith.constant 0.000000e+00 : f32
        %c0_160 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg34 = %c0_160 to %c1 step %c1 {
          %71 = ascvec.create_mask : !ascvec.mask
          %72 = ascvec.duplicate_scalar %cst, %71 : f32 -> <f32>
          %mask, %remaining_out = ascvec.update_mask %c8_i32 : !ascvec.mask
          ascvec.storealign %arg33[%c0_159], %72, %mask : memref<8xf32, 1>, index, <f32>
        }
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
      %59 = arith.minsi %19, %58 : i64
      %60 = arith.subi %58, %59 : i64
      %61 = arith.minsi %60, %c1_i64 : i64
      %62 = arith.cmpi sle, %19, %58 : i64
      %63 = arith.select %62, %61, %c0_i64 : i64
      %64 = arith.muli %19, %c256_i64 : i64
      %65 = ascscalar.memref_address %alloc_10 : memref<1xf32>
      %66 = arith.divui %64, %c8_i64 : i64
      %67 = arith.addi %65, %66 : i64
      %68 = ascscalar.bind_gm_ptr %67 : memref<1xf32>
      %69 = ascscalar.bitcast_memref %alloc_83 byte_offset = %c0_i64 : memref<8xf32, 1> -> memref<?xf32, 1>
      %70 = arith.cmpi ugt, %63, %c0_i64 : i64
      scf.if %70 {
        ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
        %71 = arith.trunci %63 : i64 to i32
        ascvec.copy_ub2gm %68[%c0], %69[%c0], %71, %c32_i32, %c32_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
        ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
      }
    }
    ascvec.sync_all
    %24 = memref.load %alloc_7[%c0] : memref<1xf32>
    ascvec.get_buf mte2, %c9_i64, 0 {cannir.buffer.generated_sync}
    "ascvec.copy_gm2ub_plan"(%alloc_93, %alloc_2, %c0_i32, %c0_i32, %c0_i64, %c2048_i64, %c1_i64, %c256_i64, %c0_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 2304 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 6176 : i64, ub_capacity = 256 : i64}> : (memref<64xf32, 1>, memref<576xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
    ascvec.release_buf mte2, %c9_i64, 0 {cannir.buffer.generated_sync}
    %25 = memref.load %alloc[%c0] : memref<1xf32>
    %26 = scf.for %arg33 = %19 to %22 step %5 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
      %48 = arith.index_cast %arg33 : i64 to index
      %49 = memref.load %alloc_6[%48] : memref<1xi64>
      %50 = arith.cmpi sge, %49, %c0_i64 : i64
      %51 = scf.if %50 -> (i1) {
        %53 = cannir.extract_layout %arg11 : !cannir.tensor<<gm, ui8>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, identity>
        %54 = cannir.extract_shape %53 : <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>> -> <(-1, -1, 1, 576)>
        %55 = cannir.extract_dim %54[0] : !cannir.shape<(-1, -1, 1, 576)>
        %56 = cannir.extract_dim %54[1] : !cannir.shape<(-1, -1, 1, 576)>
        %57 = arith.muli %55, %56 : i64
        %58 = arith.cmpi slt, %49, %57 : i64
        scf.yield %58 : i1
      } else {
        scf.yield %false : i1
      }
      %52 = scf.if %51 -> (i64) {
        %53 = arith.cmpi eq, %arg34, %c0_i64 : i64
        %54 = ascscalar.memref_address %alloc_84 : memref<512xf32, 1>
        %55 = ascscalar.memref_address %alloc_85 : memref<512xf32, 1>
        %56 = arith.select %53, %54, %55 : i64
        %57 = arith.select %53, %c1_i64, %c2_i64 : i64
        %58 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
        %59 = arith.remui %58, %c2_i64 : i64
        %60 = arith.minsi %arg33, %22 : i64
        %61 = arith.subi %22, %60 : i64
        %62 = arith.minsi %61, %c1_i64 : i64
        %63 = arith.cmpi sle, %arg33, %22 : i64
        %64 = arith.select %63, %62, %c0_i64 : i64
        %65 = arith.muli %arg33, %c2048_i64 : i64
        %66 = arith.muli %64, %c2048_i64 : i64
        %67 = arith.cmpi eq, %66, %c0_i64 : i64
        %68 = arith.cmpi ne, %66, %c0_i64 : i64
        %69 = arith.extui %68 : i1 to i64
        %70 = arith.select %67, %c4_i64, %66 : i64
        %71 = ascscalar.bind_local_ptr %56 : memref<1xf32, 1>
        ascvec.get_buf mte2, %57, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%71, %alloc_13, %c0_i32, %c0_i32, %c0_i64, %65, %69, %70, %c0_i64, %70, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %57, 0 {cannir.buffer.generated_sync}
        %72 = ascscalar.bind_local_ptr %56 : memref<512xf32, 1>
        %73 = ascscalar.memref_address %alloc_86 : memref<576xui8, 1>
        %74 = ascscalar.memref_address %alloc_87 : memref<576xui8, 1>
        %75 = arith.select %53, %73, %74 : i64
        %76 = ascscalar.bind_local_ptr %75 : memref<576xui8, 1>
        %77 = arith.select %53, %c3_i64, %c4_i64 : i64
        ascvec.get_buf v, %57, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %77, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%72, %24, %76 : memref<512xf32, 1>, f32, memref<576xui8, 1>) {
        ^bb0(%arg35: memref<512xf32, 1>, %arg36: f32, %arg37: memref<576xui8, 1>):
          %c64_i32 = arith.constant 64 : i32
          %c512_i32 = arith.constant 512 : i32
          %c0_i32_159 = arith.constant 0 : i32
          %cst = arith.constant 1.1920929E-7 : f32
          %cst_160 = arith.constant 3.276800e+04 : f32
          %cst_161 = arith.constant -3.276800e+04 : f32
          %cst_162 = arith.constant 0.000000e+00 : f32
          %135 = ascvec.create_mask : !ascvec.mask
          scf.for %arg38 = %c0_i32_159 to %c512_i32 step %c64_i32  : i32 {
            %136 = arith.index_cast %arg38 : i32 to index
            %137 = ascvec.loadalign %arg35[%136] : memref<512xf32, 1>, index -> <f32>
            %138 = ascvec.mul_scalar %137, %arg36, %135 : <f32>, f32
            %139 = ascvec.duplicate_scalar %cst_162, %135 : f32 -> <f32>
            %140 = ascvec.cmp_eq %138, %138, %135 : <f32>, <f32> -> !ascvec.mask
            %141 = ascvec.select %138, %139, %140 : <f32>
            %142 = ascvec.max_scalar %141, %cst_161, %135 : <f32>, f32
            %143 = ascvec.min_scalar %142, %cst_160, %135 : <f32>, f32
            %144 = ascvec.abs %143, %135 : <f32>
            %145 = ascvec.cmp_lt_scalar %144, %cst, %135 : <f32>, f32 -> !ascvec.mask
            %146 = ascvec.select %139, %143, %145 : <f32>
            %147 = ascvec.cast %146, %135 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
            %148 = ascvec.reinterpret %147 : <!ascdtype.hifloat8> -> <f8E4M3FN>
            ascvec.storealign_pack %arg37[%136], %148, %135 {mode = "b32_to_b8"} : memref<576xui8, 1>, index, <f8E4M3FN>
          }
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %57, 0 {cannir.buffer.generated_sync}
        %78 = ascscalar.memref_address %alloc_88 : memref<64xf32, 1>
        %79 = ascscalar.memref_address %alloc_89 : memref<64xf32, 1>
        %80 = arith.select %53, %78, %79 : i64
        %81 = arith.select %53, %c5_i64, %c6_i64 : i64
        %82 = cannir.extract_layout %arg17 : !cannir.tensor<<gm, f32>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>
        %83 = cannir.extract_shape %82 : <<(-1, 576)>, <(576, 1)>> -> <(-1, 576)>
        %84 = cannir.extract_dim %83[0] : !cannir.shape<(-1, 576)>
        %85 = arith.minsi %arg33, %84 : i64
        %86 = arith.subi %84, %85 : i64
        %87 = arith.minsi %86, %c1_i64 : i64
        %88 = arith.cmpi sle, %arg33, %84 : i64
        %89 = arith.select %88, %87, %c0_i64 : i64
        %90 = arith.muli %arg33, %c2304_i64 : i64
        %91 = arith.addi %90, %c2048_i64 : i64
        %92 = ascscalar.bind_local_ptr %80 : memref<1xf32, 1>
        ascvec.get_buf mte2, %81, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%92, %alloc_11, %c0_i32, %c0_i32, %c0_i64, %91, %89, %c256_i64, %c2304_i64, %c256_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %81, 0 {cannir.buffer.generated_sync}
        %93 = ascscalar.bind_local_ptr %80 : memref<64xf32, 1>
        ascvec.get_buf v, %81, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c9_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%93, %alloc_93, %25 : memref<64xf32, 1>, memref<64xf32, 1>, f32) {
        ^bb0(%arg35: memref<64xf32, 1>, %arg36: memref<64xf32, 1>, %arg37: f32):
          %c0_159 = arith.constant 0 : index
          %c0_160 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg38 = %c0_160 to %c1 step %c1 {
            %135 = ascvec.create_mask : !ascvec.mask
            %136 = ascvec.loadalign %arg35[%c0_159] : memref<64xf32, 1>, index -> <f32>
            %137 = ascvec.loadalign %arg36[%c0_159] : memref<64xf32, 1>, index -> <f32>
            %138 = ascvec.mul %136, %137, %135 : <f32>
            %139 = ascvec.mul_scalar %138, %arg37, %135 : <f32>, f32
            ascvec.storealign %arg35[%c0_159], %139, %135 : memref<64xf32, 1>, index, <f32>
          }
          ascvec.mem_bar vst_vld
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c9_i64, 0 {cannir.buffer.generated_sync}
        %94 = cannir.extract_layout %arg14 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
        %95 = cannir.extract_shape %94 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
        %96 = cannir.extract_dim %95[0] : !cannir.shape<(-1, 64)>
        %97 = arith.minsi %arg33, %96 : i64
        %98 = arith.subi %96, %97 : i64
        %99 = arith.minsi %98, %c1_i64 : i64
        %100 = arith.cmpi sle, %arg33, %96 : i64
        %101 = arith.select %100, %99, %c0_i64 : i64
        %102 = arith.muli %arg33, %c128_i64 : i64
        %103 = arith.muli %101, %c128_i64 : i64
        %104 = arith.cmpi eq, %103, %c0_i64 : i64
        %105 = arith.cmpi ne, %103, %c0_i64 : i64
        %106 = arith.extui %105 : i1 to i64
        %107 = arith.select %104, %c2_i64, %103 : i64
        ascvec.get_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%alloc_90, %alloc_8, %c0_i32, %c0_i32, %c0_i64, %102, %106, %107, %c0_i64, %107, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 5792 : i64, ub_capacity = 128 : i64}> : (memref<64xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c7_i64, 0 {cannir.buffer.generated_sync}
        %108 = cannir.extract_layout %arg15 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
        %109 = cannir.extract_shape %108 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
        %110 = cannir.extract_dim %109[0] : !cannir.shape<(-1, 64)>
        %111 = arith.minsi %arg33, %110 : i64
        %112 = arith.subi %110, %111 : i64
        %113 = arith.minsi %112, %c1_i64 : i64
        %114 = arith.cmpi sle, %arg33, %110 : i64
        %115 = arith.select %114, %113, %c0_i64 : i64
        %116 = arith.muli %115, %c128_i64 : i64
        %117 = arith.cmpi eq, %116, %c0_i64 : i64
        %118 = arith.cmpi ne, %116, %c0_i64 : i64
        %119 = arith.extui %118 : i1 to i64
        %120 = arith.select %117, %c2_i64, %116 : i64
        ascvec.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%alloc_91, %alloc_9, %c0_i32, %c0_i32, %c0_i64, %102, %119, %120, %c0_i64, %120, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 5920 : i64, ub_capacity = 128 : i64}> : (memref<64xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
        ascvec.get_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%93, %alloc_90, %alloc_91, %alloc_92 : memref<64xf32, 1>, memref<64xbf16, 1>, memref<64xbf16, 1>, memref<64xbf16, 1>) {
        ^bb0(%arg35: memref<64xf32, 1>, %arg36: memref<64xbf16, 1>, %arg37: memref<64xbf16, 1>, %arg38: memref<64xbf16, 1>):
          %c32 = arith.constant 32 : index
          %cst = arith.constant 3.38953139E+38 : f32
          %cst_159 = arith.constant -3.38953139E+38 : f32
          %cst_160 = arith.constant 0.000000e+00 : f32
          %c1_i32_161 = arith.constant 1 : i32
          %c0_i32_162 = arith.constant 0 : i32
          %c0_163 = arith.constant 0 : index
          %c32_i32_164 = arith.constant 32 : i32
          %c0_165 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg39 = %c0_165 to %c1 step %c1 {
            %135 = ascvec.create_mask : !ascvec.mask
            %mask, %remaining_out = ascvec.update_mask %c32_i32_164 : !ascvec.mask
            %136 = ascvec.loadalign %arg35[%c0_163] : memref<64xf32, 1>, index -> <f32>
            %137 = ascvec.arange %c0_i32_162 : i32 -> <ui32>
            %138 = ascvec.mul_scalar %137, %c1_i32_161, %135 : <ui32>, i32
            %139 = ascvec.add_scalar %138, %c32_i32_164, %135 : <ui32>, i32
            %140 = ascvec.gather_reg %136, %138 : <f32>, <ui32> -> <f32>
            %141 = ascvec.gather_reg %136, %139 : <f32>, <ui32> -> <f32>
            %142 = ascvec.loadalign_unpack %arg36[%c0_163] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
            %143 = ascvec.cast %142, %135 {rounding = 6 : i32} : <bf16> -> <f32>
            %144 = ascvec.loadalign_unpack %arg37[%c0_163] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
            %145 = ascvec.cast %144, %135 {rounding = 6 : i32} : <bf16> -> <f32>
            %146 = ascvec.add_scalar %137, %c32_i32_164, %135 : <ui32>, i32
            %147 = ascvec.gather_reg %143, %146 : <f32>, <ui32> -> <f32>
            %148 = ascvec.gather_reg %145, %146 : <f32>, <ui32> -> <f32>
            %149 = ascvec.mul %140, %145, %mask : <f32>
            %150 = ascvec.mul %141, %143, %mask : <f32>
            %151 = ascvec.sub %149, %150, %mask : <f32>
            %152 = ascvec.mul %140, %147, %mask : <f32>
            %153 = ascvec.mul %141, %148, %mask : <f32>
            %154 = ascvec.add %152, %153, %mask : <f32>
            %155 = ascvec.duplicate_scalar %cst_160, %mask : f32 -> <f32>
            %156 = ascvec.cmp_eq %151, %151, %mask : <f32>, <f32> -> !ascvec.mask
            %157 = ascvec.select %151, %155, %156 : <f32>
            %158 = ascvec.max_scalar %157, %cst_159, %mask : <f32>, f32
            %159 = ascvec.min_scalar %158, %cst, %mask : <f32>, f32
            %160 = ascvec.cmp_eq %154, %154, %mask : <f32>, <f32> -> !ascvec.mask
            %161 = ascvec.select %154, %155, %160 : <f32>
            %162 = ascvec.max_scalar %161, %cst_159, %mask : <f32>, f32
            %163 = ascvec.min_scalar %162, %cst, %mask : <f32>, f32
            %164 = ascvec.cast %159, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
            %165 = ascvec.cast %163, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
            ascvec.storealign_pack %arg38[%c0_163], %164, %mask {mode = "b32_to_b16"} : memref<64xbf16, 1>, index, <bf16>
            ascvec.storealign_pack %arg38[%c32], %165, %mask {mode = "b32_to_b16"} : memref<64xbf16, 1>, index, <bf16>
          }
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %c7_i64, 0 {cannir.buffer.generated_sync}
        ascvec.release_buf v, %81, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%alloc_92, %24, %76 : memref<64xbf16, 1>, f32, memref<576xui8, 1>) {
        ^bb0(%arg35: memref<64xbf16, 1>, %arg36: f32, %arg37: memref<576xui8, 1>):
          %c512 = arith.constant 512 : index
          %cst = arith.constant 1.1920929E-7 : f32
          %cst_159 = arith.constant 3.276800e+04 : f32
          %cst_160 = arith.constant -3.276800e+04 : f32
          %cst_161 = arith.constant 0.000000e+00 : f32
          %c0_162 = arith.constant 0 : index
          %c0_163 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg38 = %c0_163 to %c1 step %c1 {
            %135 = ascvec.create_mask : !ascvec.mask
            %136 = ascvec.loadalign_unpack %arg35[%c0_162] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
            %137 = ascvec.cast %136, %135 {rounding = 6 : i32} : <bf16> -> <f32>
            %138 = ascvec.mul_scalar %137, %arg36, %135 : <f32>, f32
            %139 = ascvec.duplicate_scalar %cst_161, %135 : f32 -> <f32>
            %140 = ascvec.cmp_eq %138, %138, %135 : <f32>, <f32> -> !ascvec.mask
            %141 = ascvec.select %138, %139, %140 : <f32>
            %142 = ascvec.max_scalar %141, %cst_160, %135 : <f32>, f32
            %143 = ascvec.min_scalar %142, %cst_159, %135 : <f32>, f32
            %144 = ascvec.abs %143, %135 : <f32>
            %145 = ascvec.cmp_lt_scalar %144, %cst, %135 : <f32>, f32 -> !ascvec.mask
            %146 = ascvec.select %139, %143, %145 : <f32>
            %147 = ascvec.cast %146, %135 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
            %148 = ascvec.reinterpret %147 : <!ascdtype.hifloat8> -> <f8E4M3FN>
            ascvec.storealign_pack %arg37[%c512], %148, %135 {mode = "b32_to_b8"} : memref<576xui8, 1>, index, <f8E4M3FN>
          }
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %77, 0 {cannir.buffer.generated_sync}
        %121 = cannir.extract_layout %arg11 : !cannir.tensor<<gm, ui8>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, identity>
        %122 = cannir.extract_shape %121 : <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>> -> <(-1, -1, 1, 576)>
        %123 = cannir.extract_dim %122[1] : !cannir.shape<(-1, -1, 1, 576)>
        %124 = arith.remsi %49, %123 : i64
        %125 = arith.cmpi ne, %124, %c0_i64 : i64
        %126 = arith.cmpi slt, %124, %c0_i64 : i64
        %127 = arith.cmpi slt, %123, %c0_i64 : i64
        %128 = arith.xori %126, %127 : i1
        %129 = arith.andi %125, %128 : i1
        %130 = arith.divsi %49, %123 : i64
        %131 = arith.extui %129 : i1 to i64
        %132 = arith.subi %130, %131 : i64
        %133 = arith.select %129, %123, %c0_i64 : i64
        %134 = arith.addi %124, %133 : i64
        ascvec.get_buf mte3, %77, 0 {cannir.buffer.generated_sync}
        scf.for %arg35 = %c0_i64 to %c18_i64 step %c1_i64  : i64 {
          %135 = arith.muli %123, %c576_i64 : i64
          %136 = arith.muli %132, %135 : i64
          %137 = arith.muli %123, %c32_i64 : i64
          %138 = arith.muli %arg35, %137 : i64
          %139 = arith.addi %136, %138 : i64
          %140 = arith.muli %134, %c32_i64 : i64
          %141 = arith.addi %139, %140 : i64
          %142 = arith.muli %141, %c8_i64 : i64
          %143 = ascscalar.memref_address %alloc_5 : memref<1xui8>
          %144 = arith.divui %142, %c8_i64 : i64
          %145 = arith.addi %143, %144 : i64
          %146 = ascscalar.bind_gm_ptr %145 : memref<1xui8>
          %147 = arith.muli %arg35, %c256_i64 overflow<nsw> : i64
          %148 = arith.divui %147, %c8_i64 : i64
          %149 = arith.addi %75, %148 : i64
          %150 = ascscalar.bind_local_ptr %149 : memref<1xui8, 1>
          ascvec.copy_ub2gm %146[%c0], %150[%c0], %c1_i32, %c32_i32, %c0_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
        }
        ascvec.release_buf mte3, %77, 0 {cannir.buffer.generated_sync}
        scf.yield %59 : i64
      } else {
        scf.yield %arg34 : i64
      }
      scf.yield %52 : i64
    }
    ascvec.sync_all
    ascvec.sync_block_arrive mte3, %c0_i64
    ascvec.sync_block_wait s, %c2_i64
    %27 = cannir.extract_layout %arg25 : !cannir.tensor<<gm, bf16>, <<(-1, 512)>, <(512, 1)>>, <<(-1, 512)>, <(512, 1)>>, identity>
    %28 = cannir.extract_shape %27 : <<(-1, 512)>, <(512, 1)>> -> <(-1, 512)>
    %29 = cannir.extract_dim %28[0] : !cannir.shape<(-1, 512)>
    %30 = arith.remsi %29, %c2_i64 : i64
    %31 = arith.cmpi ne, %30, %c0_i64 : i64
    %32 = arith.cmpi slt, %30, %c0_i64 : i64
    %33 = arith.andi %31, %32 : i1
    %34 = arith.divsi %29, %c2_i64 : i64
    %35 = arith.extui %33 : i1 to i64
    %36 = arith.subi %34, %35 : i64
    scf.if %6 {
      ascvec.vf(%alloc_100 : memref<128xf32, 1>) {
      ^bb0(%arg33: memref<128xf32, 1>):
        %c64 = arith.constant 64 : index
        %c0_159 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c0_160 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg34 = %c0_160 to %c1 step %c1 {
          %48 = ascvec.create_mask : !ascvec.mask
          %49 = ascvec.duplicate_scalar %cst, %48 : f32 -> <f32>
          ascvec.storealign %arg33[%c0_159], %49, %48 : memref<128xf32, 1>, index, <f32>
          ascvec.storealign %arg33[%c64], %49, %48 : memref<128xf32, 1>, index, <f32>
        }
      } {cannir.vf_mode = "simd"}
    }
    %37 = arith.addi %36, %c15_i64 : i64
    %38 = arith.remsi %37, %c16_i64 : i64
    %39 = arith.cmpi ne, %38, %c0_i64 : i64
    %40 = arith.cmpi slt, %38, %c0_i64 : i64
    %41 = arith.andi %39, %40 : i1
    %42 = arith.divsi %37, %c16_i64 : i64
    %43 = arith.extui %41 : i1 to i64
    %44 = arith.subi %42, %43 : i64
    %45 = arith.muli %44, %c2_i64 : i64
    %46 = arith.cmpi slt, %0, %45 : i64
    scf.if %46 {
      %48 = arith.remsi %0, %c2_i64 : i64
      %49 = arith.cmpi ne, %48, %c0_i64 : i64
      %50 = arith.cmpi slt, %48, %c0_i64 : i64
      %51 = arith.andi %49, %50 : i1
      %52 = arith.divsi %0, %c2_i64 : i64
      %53 = arith.extui %51 : i1 to i64
      %54 = arith.subi %52, %53 : i64
      %55 = arith.muli %54, %c16_i64 : i64
      %56 = arith.addi %55, %c16_i64 : i64
      %57 = cannir.extract_layout %arg22 : !cannir.tensor<<gm, f32>, <<(-1, 384)>, <(384, 1)>>, <<(-1, 384)>, <(384, 1)>>, identity>
      %58 = cannir.extract_shape %57 : <<(-1, 384)>, <(384, 1)>> -> <(-1, 384)>
      %59 = cannir.extract_dim %58[0] : !cannir.shape<(-1, 384)>
      %60 = arith.minsi %56, %59 : i64
      %61 = arith.select %51, %c2_i64, %c0_i64 : i64
      %62 = arith.addi %48, %61 overflow<nsw> : i64
      %63 = arith.remsi %55, %c8_i64 : i64
      %64 = arith.cmpi ne, %63, %c0_i64 : i64
      %65 = arith.cmpi slt, %63, %c0_i64 : i64
      %66 = arith.andi %64, %65 : i1
      %67 = arith.divsi %55, %c8_i64 : i64
      %68 = arith.extui %66 : i1 to i64
      %69 = arith.subi %67, %68 : i64
      %70 = arith.addi %69, %18 : i64
      %71 = arith.addi %60, %c7_i64 : i64
      %72 = arith.remsi %71, %c8_i64 : i64
      %73 = arith.cmpi ne, %72, %c0_i64 : i64
      %74 = arith.cmpi slt, %72, %c0_i64 : i64
      %75 = arith.andi %73, %74 : i1
      %76 = arith.divsi %71, %c8_i64 : i64
      %77 = arith.extui %75 : i1 to i64
      %78 = arith.subi %76, %77 : i64
      %79 = arith.cmpi slt, %70, %78 : i64
      scf.if %79 {
        %80 = arith.muli %62, %c768_i64 overflow<nsw> : i64
        ascvec.get_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%alloc_105, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %80, %c1_i64, %c768_i64, %c0_i64, %c768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 1536 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 24384 : i64, ub_capacity = 768 : i64}> : (memref<192xf32, 1>, memref<384xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %c8_i64, 0 {cannir.buffer.generated_sync}
        %81 = memref.load %alloc_15[%c0] : memref<1xf32>
        %82 = scf.for %arg33 = %70 to %78 step %c2_i64 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
          %83 = arith.muli %arg33, %c8_i64 : i64
          %84 = arith.subi %60, %83 : i64
          %85 = arith.minsi %84, %c8_i64 : i64
          %86 = arith.cmpi eq, %arg34, %c0_i64 : i64
          %87 = ascscalar.memref_address %alloc_101 : memref<1536xf32, 1>
          %88 = ascscalar.memref_address %alloc_102 : memref<1536xf32, 1>
          %89 = arith.select %86, %87, %88 : i64
          %90 = arith.select %86, %c4_i64, %c5_i64 : i64
          %91 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
          %92 = arith.remui %91, %c2_i64 : i64
          %93 = arith.divsi %59, %c8_i64 : i64
          %94 = arith.minsi %arg33, %93 : i64
          %95 = arith.muli %94, %c8_i64 : i64
          %96 = arith.subi %59, %95 : i64
          %97 = arith.minsi %96, %c8_i64 : i64
          %98 = arith.cmpi sle, %arg33, %93 : i64
          %99 = arith.select %98, %97, %c0_i64 : i64
          %100 = arith.muli %arg33, %c12288_i64 : i64
          %101 = arith.addi %100, %80 : i64
          %102 = ascscalar.bind_local_ptr %89 : memref<1xf32, 1>
          ascvec.get_buf mte2, %90, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%102, %alloc_16, %c0_i32, %c0_i32, %c0_i64, %101, %99, %c768_i64, %c1536_i64, %c768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %90, 0 {cannir.buffer.generated_sync}
          %103 = ascscalar.bind_local_ptr %89 : memref<1536xf32, 1>
          %104 = ascscalar.memref_address %alloc_103 : memref<1024xbf16, 1>
          %105 = ascscalar.memref_address %alloc_104 : memref<1024xbf16, 1>
          %106 = arith.select %86, %104, %105 : i64
          %107 = ascscalar.bind_local_ptr %106 : memref<1024xbf16, 1>
          %108 = arith.select %86, %c6_i64, %c7_i64 : i64
          %109 = arith.trunci %85 : i64 to i32
          ascvec.get_buf v, %90, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %108, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%103, %alloc_105, %81, %107, %109 : memref<1536xf32, 1>, memref<192xf32, 1>, f32, memref<1024xbf16, 1>, i32) {
          ^bb0(%arg35: memref<1536xf32, 1>, %arg36: memref<192xf32, 1>, %arg37: f32, %arg38: memref<1024xbf16, 1>, %arg39: i32):
            %c1_i32_159 = arith.constant 1 : i32
            %c64_i32 = arith.constant 64 : i32
            %c0_i32_160 = arith.constant 0 : i32
            %c128_i32_161 = arith.constant 128 : i32
            %c192_i32 = arith.constant 192 : i32
            %164 = ascvec.create_mask : !ascvec.mask
            scf.for %arg40 = %c0_i32_160 to %arg39 step %c1_i32_159  : i32 {
              scf.for %arg41 = %c0_i32_160 to %c128_i32_161 step %c64_i32  : i32 {
                %165 = arith.muli %arg40, %c192_i32 : i32
                %166 = arith.addi %165, %arg41 : i32
                %167 = arith.index_cast %166 : i32 to index
                %168 = ascvec.loadalign %arg35[%167] : memref<1536xf32, 1>, index -> <f32>
                %169 = arith.index_cast %arg41 : i32 to index
                %170 = ascvec.loadalign %arg36[%169] : memref<192xf32, 1>, index -> <f32>
                %171 = ascvec.mul %168, %170, %164 : <f32>
                %172 = ascvec.mul_scalar %171, %arg37, %164 : <f32>, f32
                %173 = ascvec.cast %172, %164 {reg_layout = 0 : i32} : <f32> -> <bf16>
                %174 = arith.muli %arg40, %c128_i32_161 : i32
                %175 = arith.addi %174, %arg41 : i32
                %176 = arith.index_cast %175 : i32 to index
                ascvec.storealign_pack %arg38[%176], %173, %164 {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
              }
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %108, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %c8_i64, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %90, 0 {cannir.buffer.generated_sync}
          %110 = arith.muli %arg33, %c2048_i64 : i64
          %111 = arith.muli %62, %c128_i64 overflow<nsw> : i64
          %112 = arith.addi %110, %111 : i64
          %113 = arith.muli %112, %c16_i64 : i64
          %114 = ascscalar.memref_address %alloc_17 : memref<1xbf16>
          %115 = arith.divui %113, %c8_i64 : i64
          %116 = arith.addi %114, %115 : i64
          %117 = ascscalar.bind_gm_ptr %116 : memref<1xbf16>
          %118 = ascscalar.bind_local_ptr %106 : memref<1xbf16, 1>
          %119 = arith.cmpi ugt, %99, %c0_i64 : i64
          scf.if %119 {
            ascvec.get_buf mte3, %108, 0 {cannir.buffer.generated_sync}
            %164 = arith.trunci %99 : i64 to i32
            ascvec.copy_ub2gm %117[%c0], %118[%c0], %164, %c256_i32, %c512_i64, %c256_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
            ascvec.release_buf mte3, %108, 0 {cannir.buffer.generated_sync}
          }
          %120 = cannir.extract_layout %arg14 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
          %121 = cannir.extract_shape %120 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
          %122 = cannir.extract_dim %121[0] : !cannir.shape<(-1, 64)>
          %123 = arith.divsi %122, %c8_i64 : i64
          %124 = arith.minsi %arg33, %123 : i64
          %125 = arith.muli %124, %c8_i64 : i64
          %126 = arith.subi %122, %125 : i64
          %127 = arith.minsi %126, %c8_i64 : i64
          %128 = arith.cmpi sle, %arg33, %123 : i64
          %129 = arith.select %128, %127, %c0_i64 : i64
          %130 = arith.muli %arg33, %c1024_i64 : i64
          %131 = arith.muli %129, %c128_i64 : i64
          %132 = arith.cmpi eq, %131, %c0_i64 : i64
          %133 = arith.cmpi ne, %131, %c0_i64 : i64
          %134 = arith.extui %133 : i1 to i64
          %135 = arith.select %132, %c2_i64, %131 : i64
          ascvec.get_buf mte2, %c11_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%alloc_108, %alloc_8, %c0_i32, %c0_i32, %c0_i64, %130, %134, %135, %c0_i64, %135, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 27200 : i64, ub_capacity = 1024 : i64}> : (memref<512xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c11_i64, 0 {cannir.buffer.generated_sync}
          %136 = cannir.extract_layout %arg15 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
          %137 = cannir.extract_shape %136 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
          %138 = cannir.extract_dim %137[0] : !cannir.shape<(-1, 64)>
          %139 = arith.divsi %138, %c8_i64 : i64
          %140 = arith.minsi %arg33, %139 : i64
          %141 = arith.muli %140, %c8_i64 : i64
          %142 = arith.subi %138, %141 : i64
          %143 = arith.minsi %142, %c8_i64 : i64
          %144 = arith.cmpi sle, %arg33, %139 : i64
          %145 = arith.select %144, %143, %c0_i64 : i64
          %146 = arith.muli %145, %c128_i64 : i64
          %147 = arith.cmpi eq, %146, %c0_i64 : i64
          %148 = arith.cmpi ne, %146, %c0_i64 : i64
          %149 = arith.extui %148 : i1 to i64
          %150 = arith.select %147, %c2_i64, %146 : i64
          ascvec.get_buf mte2, %c12_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%alloc_109, %alloc_9, %c0_i32, %c0_i32, %c0_i64, %130, %149, %150, %c0_i64, %150, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 28224 : i64, ub_capacity = 1024 : i64}> : (memref<512xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c12_i64, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %90, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%103, %alloc_105, %81, %alloc_110, %109 : memref<1536xf32, 1>, memref<192xf32, 1>, f32, memref<512xf32, 1>, i32) {
          ^bb0(%arg35: memref<1536xf32, 1>, %arg36: memref<192xf32, 1>, %arg37: f32, %arg38: memref<512xf32, 1>, %arg39: i32):
            %c128 = arith.constant 128 : index
            %c1_i32_159 = arith.constant 1 : i32
            %c0_i32_160 = arith.constant 0 : i32
            %c64_i32 = arith.constant 64 : i32
            %c128_i32_161 = arith.constant 128 : i32
            %c192_i32 = arith.constant 192 : i32
            %164 = ascvec.create_mask : !ascvec.mask
            scf.for %arg40 = %c0_i32_160 to %arg39 step %c1_i32_159  : i32 {
              %165 = arith.muli %arg40, %c192_i32 : i32
              %166 = arith.addi %165, %c128_i32_161 : i32
              %167 = arith.index_cast %166 : i32 to index
              %168 = ascvec.loadalign %arg35[%167] : memref<1536xf32, 1>, index -> <f32>
              %169 = ascvec.loadalign %arg36[%c128] : memref<192xf32, 1>, index -> <f32>
              %170 = ascvec.mul %168, %169, %164 : <f32>
              %171 = ascvec.mul_scalar %170, %arg37, %164 : <f32>, f32
              %172 = arith.muli %arg40, %c64_i32 : i32
              %173 = arith.index_cast %172 : i32 to index
              ascvec.storealign %arg38[%173], %171, %164 : memref<512xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %90, 0 {cannir.buffer.generated_sync}
          %151 = ascscalar.memref_address %alloc_106 : memref<512xbf16, 1>
          %152 = ascscalar.memref_address %alloc_107 : memref<512xbf16, 1>
          %153 = arith.select %86, %151, %152 : i64
          %154 = ascscalar.bind_local_ptr %153 : memref<512xbf16, 1>
          %155 = arith.select %86, %c9_i64, %c10_i64 : i64
          scf.for %arg35 = %c0_i64 to %85 step %c1_i64  : i64 {
            %164 = arith.muli %arg35, %c64_i64 : i64
            %165 = arith.trunci %164 : i64 to i32
            %166 = arith.index_cast %165 : i32 to index
            ascvec.get_buf v, %c11_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c12_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %155, 0 {cannir.buffer.generated_sync}
            %167 = arith.index_cast %164 : i64 to index
            ascvec.vf(%alloc_110, %166, %alloc_108, %167, %alloc_109, %167, %154, %167 : memref<512xf32, 1>, index, memref<512xbf16, 1>, index, memref<512xbf16, 1>, index, memref<512xbf16, 1>, index) {
            ^bb0(%arg36: memref<512xf32, 1>, %arg37: index, %arg38: memref<512xbf16, 1>, %arg39: index, %arg40: memref<512xbf16, 1>, %arg41: index, %arg42: memref<512xbf16, 1>, %arg43: index):
              %c32 = arith.constant 32 : index
              %c1_i32_159 = arith.constant 1 : i32
              %c0_i32_160 = arith.constant 0 : i32
              %c32_i32_161 = arith.constant 32 : i32
              %168 = ascvec.create_mask : !ascvec.mask
              %mask, %remaining_out = ascvec.update_mask %c32_i32_161 : !ascvec.mask
              %169 = ascvec.loadalign %arg36[%arg37] : memref<512xf32, 1>, index -> <f32>
              %170 = ascvec.arange %c0_i32_160 : i32 -> <ui32>
              %171 = ascvec.mul_scalar %170, %c1_i32_159, %168 : <ui32>, i32
              %172 = ascvec.add_scalar %171, %c32_i32_161, %168 : <ui32>, i32
              %173 = ascvec.gather_reg %169, %171 : <f32>, <ui32> -> <f32>
              %174 = ascvec.gather_reg %169, %172 : <f32>, <ui32> -> <f32>
              %175 = ascvec.loadalign_unpack %arg38[%arg39] {mode = "b16_to_b32"} : memref<512xbf16, 1>, index -> <bf16>
              %176 = ascvec.cast %175, %168 {rounding = 6 : i32} : <bf16> -> <f32>
              %177 = ascvec.loadalign_unpack %arg40[%arg41] {mode = "b16_to_b32"} : memref<512xbf16, 1>, index -> <bf16>
              %178 = ascvec.cast %177, %168 {rounding = 6 : i32} : <bf16> -> <f32>
              %179 = ascvec.add_scalar %170, %c32_i32_161, %168 : <ui32>, i32
              %180 = ascvec.gather_reg %176, %179 : <f32>, <ui32> -> <f32>
              %181 = ascvec.gather_reg %178, %179 : <f32>, <ui32> -> <f32>
              %182 = ascvec.mul %173, %178, %mask : <f32>
              %183 = ascvec.mul %174, %176, %mask : <f32>
              %184 = ascvec.sub %182, %183, %mask : <f32>
              %185 = ascvec.mul %173, %180, %mask : <f32>
              %186 = ascvec.mul %174, %181, %mask : <f32>
              %187 = ascvec.add %185, %186, %mask : <f32>
              %188 = ascvec.cast %184, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
              %189 = ascvec.cast %187, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
              ascvec.storealign_pack %arg42[%arg43], %188, %mask {mode = "b32_to_b16"} : memref<512xbf16, 1>, index, <bf16>
              %190 = arith.addi %arg43, %c32 : index
              %c0_162 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg44 = %c0_162 to %c1 step %c1 {
                ascvec.storealign_pack %arg42[%190], %189, %mask {mode = "b32_to_b16"} : memref<512xbf16, 1>, index, <bf16>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %155, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %c12_i64, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %c11_i64, 0 {cannir.buffer.generated_sync}
          }
          %156 = arith.muli %62, %c64_i64 overflow<nsw> : i64
          %157 = arith.addi %130, %156 : i64
          %158 = arith.muli %157, %c16_i64 : i64
          %159 = ascscalar.memref_address %alloc_18 : memref<1xbf16>
          %160 = arith.divui %158, %c8_i64 : i64
          %161 = arith.addi %159, %160 : i64
          %162 = ascscalar.bind_gm_ptr %161 : memref<1xbf16>
          %163 = ascscalar.bind_local_ptr %153 : memref<1xbf16, 1>
          scf.if %119 {
            ascvec.get_buf mte3, %155, 0 {cannir.buffer.generated_sync}
            %164 = arith.trunci %99 : i64 to i32
            ascvec.copy_ub2gm %162[%c0], %163[%c0], %164, %c128_i32, %c256_i64, %c128_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
            ascvec.release_buf mte3, %155, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %92 : i64
        }
      }
      ascvec.sync_block_arrive mte3, %c5_i64
    }
    %47 = scf.for %arg33 = %0 to %45 step %1 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
      %48 = arith.addi %arg33, %1 : i64
      %49 = arith.cmpi slt, %48, %45 : i64
      scf.if %49 {
        %64 = arith.remsi %48, %c2_i64 : i64
        %65 = arith.cmpi ne, %64, %c0_i64 : i64
        %66 = arith.cmpi slt, %64, %c0_i64 : i64
        %67 = arith.andi %65, %66 : i1
        %68 = arith.divsi %48, %c2_i64 : i64
        %69 = arith.extui %67 : i1 to i64
        %70 = arith.subi %68, %69 : i64
        %71 = arith.muli %70, %c16_i64 : i64
        %72 = arith.addi %71, %c16_i64 : i64
        %73 = cannir.extract_layout %arg22 : !cannir.tensor<<gm, f32>, <<(-1, 384)>, <(384, 1)>>, <<(-1, 384)>, <(384, 1)>>, identity>
        %74 = cannir.extract_shape %73 : <<(-1, 384)>, <(384, 1)>> -> <(-1, 384)>
        %75 = cannir.extract_dim %74[0] : !cannir.shape<(-1, 384)>
        %76 = arith.minsi %72, %75 : i64
        %77 = arith.select %67, %c2_i64, %c0_i64 : i64
        %78 = arith.addi %64, %77 overflow<nsw> : i64
        %79 = arith.remsi %71, %c8_i64 : i64
        %80 = arith.cmpi ne, %79, %c0_i64 : i64
        %81 = arith.cmpi slt, %79, %c0_i64 : i64
        %82 = arith.andi %80, %81 : i1
        %83 = arith.divsi %71, %c8_i64 : i64
        %84 = arith.extui %82 : i1 to i64
        %85 = arith.subi %83, %84 : i64
        %86 = arith.addi %85, %18 : i64
        %87 = arith.addi %76, %c7_i64 : i64
        %88 = arith.remsi %87, %c8_i64 : i64
        %89 = arith.cmpi ne, %88, %c0_i64 : i64
        %90 = arith.cmpi slt, %88, %c0_i64 : i64
        %91 = arith.andi %89, %90 : i1
        %92 = arith.divsi %87, %c8_i64 : i64
        %93 = arith.extui %91 : i1 to i64
        %94 = arith.subi %92, %93 : i64
        %95 = arith.cmpi slt, %86, %94 : i64
        scf.if %95 {
          %96 = arith.muli %78, %c768_i64 overflow<nsw> : i64
          ascvec.get_buf mte2, %c17_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%alloc_115, %alloc_1, %c0_i32, %c0_i32, %c0_i64, %96, %c1_i64, %c768_i64, %c0_i64, %c768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 1536 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 47680 : i64, ub_capacity = 768 : i64}> : (memref<192xf32, 1>, memref<384xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c17_i64, 0 {cannir.buffer.generated_sync}
          %97 = memref.load %alloc_15[%c0] : memref<1xf32>
          %98 = scf.for %arg35 = %86 to %94 step %c2_i64 iter_args(%arg36 = %c0_i64) -> (i64)  : i64 {
            %99 = arith.muli %arg35, %c8_i64 : i64
            %100 = arith.subi %76, %99 : i64
            %101 = arith.minsi %100, %c8_i64 : i64
            %102 = arith.cmpi eq, %arg36, %c0_i64 : i64
            %103 = ascscalar.memref_address %alloc_111 : memref<1536xf32, 1>
            %104 = ascscalar.memref_address %alloc_112 : memref<1536xf32, 1>
            %105 = arith.select %102, %103, %104 : i64
            %106 = arith.select %102, %c13_i64, %c14_i64 : i64
            %107 = arith.addi %arg36, %c1_i64 overflow<nsw> : i64
            %108 = arith.remui %107, %c2_i64 : i64
            %109 = arith.divsi %75, %c8_i64 : i64
            %110 = arith.minsi %arg35, %109 : i64
            %111 = arith.muli %110, %c8_i64 : i64
            %112 = arith.subi %75, %111 : i64
            %113 = arith.minsi %112, %c8_i64 : i64
            %114 = arith.cmpi sle, %arg35, %109 : i64
            %115 = arith.select %114, %113, %c0_i64 : i64
            %116 = arith.muli %arg35, %c12288_i64 : i64
            %117 = arith.addi %116, %96 : i64
            %118 = ascscalar.bind_local_ptr %105 : memref<1xf32, 1>
            ascvec.get_buf mte2, %106, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%118, %alloc_16, %c0_i32, %c0_i32, %c0_i64, %117, %115, %c768_i64, %c1536_i64, %c768_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = -1 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %106, 0 {cannir.buffer.generated_sync}
            %119 = ascscalar.bind_local_ptr %105 : memref<1536xf32, 1>
            %120 = ascscalar.memref_address %alloc_113 : memref<1024xbf16, 1>
            %121 = ascscalar.memref_address %alloc_114 : memref<1024xbf16, 1>
            %122 = arith.select %102, %120, %121 : i64
            %123 = ascscalar.bind_local_ptr %122 : memref<1024xbf16, 1>
            %124 = arith.select %102, %c15_i64, %c16_i64 : i64
            %125 = arith.trunci %101 : i64 to i32
            ascvec.get_buf v, %106, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c17_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %124, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%119, %alloc_115, %97, %123, %125 : memref<1536xf32, 1>, memref<192xf32, 1>, f32, memref<1024xbf16, 1>, i32) {
            ^bb0(%arg37: memref<1536xf32, 1>, %arg38: memref<192xf32, 1>, %arg39: f32, %arg40: memref<1024xbf16, 1>, %arg41: i32):
              %c1_i32_159 = arith.constant 1 : i32
              %c64_i32 = arith.constant 64 : i32
              %c0_i32_160 = arith.constant 0 : i32
              %c128_i32_161 = arith.constant 128 : i32
              %c192_i32 = arith.constant 192 : i32
              %180 = ascvec.create_mask : !ascvec.mask
              scf.for %arg42 = %c0_i32_160 to %arg41 step %c1_i32_159  : i32 {
                scf.for %arg43 = %c0_i32_160 to %c128_i32_161 step %c64_i32  : i32 {
                  %181 = arith.muli %arg42, %c192_i32 : i32
                  %182 = arith.addi %181, %arg43 : i32
                  %183 = arith.index_cast %182 : i32 to index
                  %184 = ascvec.loadalign %arg37[%183] : memref<1536xf32, 1>, index -> <f32>
                  %185 = arith.index_cast %arg43 : i32 to index
                  %186 = ascvec.loadalign %arg38[%185] : memref<192xf32, 1>, index -> <f32>
                  %187 = ascvec.mul %184, %186, %180 : <f32>
                  %188 = ascvec.mul_scalar %187, %arg39, %180 : <f32>, f32
                  %189 = ascvec.cast %188, %180 {reg_layout = 0 : i32} : <f32> -> <bf16>
                  %190 = arith.muli %arg42, %c128_i32_161 : i32
                  %191 = arith.addi %190, %arg43 : i32
                  %192 = arith.index_cast %191 : i32 to index
                  ascvec.storealign_pack %arg40[%192], %189, %180 {mode = "b32_to_b16"} : memref<1024xbf16, 1>, index, <bf16>
                }
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %124, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %c17_i64, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %106, 0 {cannir.buffer.generated_sync}
            %126 = arith.muli %arg35, %c2048_i64 : i64
            %127 = arith.muli %78, %c128_i64 overflow<nsw> : i64
            %128 = arith.addi %126, %127 : i64
            %129 = arith.muli %128, %c16_i64 : i64
            %130 = ascscalar.memref_address %alloc_17 : memref<1xbf16>
            %131 = arith.divui %129, %c8_i64 : i64
            %132 = arith.addi %130, %131 : i64
            %133 = ascscalar.bind_gm_ptr %132 : memref<1xbf16>
            %134 = ascscalar.bind_local_ptr %122 : memref<1xbf16, 1>
            %135 = arith.cmpi ugt, %115, %c0_i64 : i64
            scf.if %135 {
              ascvec.get_buf mte3, %124, 0 {cannir.buffer.generated_sync}
              %180 = arith.trunci %115 : i64 to i32
              ascvec.copy_ub2gm %133[%c0], %134[%c0], %180, %c256_i32, %c512_i64, %c256_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
              ascvec.release_buf mte3, %124, 0 {cannir.buffer.generated_sync}
            }
            %136 = cannir.extract_layout %arg14 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
            %137 = cannir.extract_shape %136 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
            %138 = cannir.extract_dim %137[0] : !cannir.shape<(-1, 64)>
            %139 = arith.divsi %138, %c8_i64 : i64
            %140 = arith.minsi %arg35, %139 : i64
            %141 = arith.muli %140, %c8_i64 : i64
            %142 = arith.subi %138, %141 : i64
            %143 = arith.minsi %142, %c8_i64 : i64
            %144 = arith.cmpi sle, %arg35, %139 : i64
            %145 = arith.select %144, %143, %c0_i64 : i64
            %146 = arith.muli %arg35, %c1024_i64 : i64
            %147 = arith.muli %145, %c128_i64 : i64
            %148 = arith.cmpi eq, %147, %c0_i64 : i64
            %149 = arith.cmpi ne, %147, %c0_i64 : i64
            %150 = arith.extui %149 : i1 to i64
            %151 = arith.select %148, %c2_i64, %147 : i64
            ascvec.get_buf mte2, %c20_i64, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%alloc_118, %alloc_8, %c0_i32, %c0_i32, %c0_i64, %146, %150, %151, %c0_i64, %151, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 50496 : i64, ub_capacity = 1024 : i64}> : (memref<512xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %c20_i64, 0 {cannir.buffer.generated_sync}
            %152 = cannir.extract_layout %arg15 : !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>
            %153 = cannir.extract_shape %152 : <<(-1, 64)>, <(64, 1)>> -> <(-1, 64)>
            %154 = cannir.extract_dim %153[0] : !cannir.shape<(-1, 64)>
            %155 = arith.divsi %154, %c8_i64 : i64
            %156 = arith.minsi %arg35, %155 : i64
            %157 = arith.muli %156, %c8_i64 : i64
            %158 = arith.subi %154, %157 : i64
            %159 = arith.minsi %158, %c8_i64 : i64
            %160 = arith.cmpi sle, %arg35, %155 : i64
            %161 = arith.select %160, %159, %c0_i64 : i64
            %162 = arith.muli %161, %c128_i64 : i64
            %163 = arith.cmpi eq, %162, %c0_i64 : i64
            %164 = arith.cmpi ne, %162, %c0_i64 : i64
            %165 = arith.extui %164 : i1 to i64
            %166 = arith.select %163, %c2_i64, %162 : i64
            ascvec.get_buf mte2, %c21_i64, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%alloc_119, %alloc_9, %c0_i32, %c0_i32, %c0_i64, %146, %165, %166, %c0_i64, %166, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = 51520 : i64, ub_capacity = 1024 : i64}> : (memref<512xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %c21_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %106, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %c17_i64, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%119, %alloc_115, %97, %alloc_120, %125 : memref<1536xf32, 1>, memref<192xf32, 1>, f32, memref<512xf32, 1>, i32) {
            ^bb0(%arg37: memref<1536xf32, 1>, %arg38: memref<192xf32, 1>, %arg39: f32, %arg40: memref<512xf32, 1>, %arg41: i32):
              %c128 = arith.constant 128 : index
              %c1_i32_159 = arith.constant 1 : i32
              %c0_i32_160 = arith.constant 0 : i32
              %c64_i32 = arith.constant 64 : i32
              %c128_i32_161 = arith.constant 128 : i32
              %c192_i32 = arith.constant 192 : i32
              %180 = ascvec.create_mask : !ascvec.mask
              scf.for %arg42 = %c0_i32_160 to %arg41 step %c1_i32_159  : i32 {
                %181 = arith.muli %arg42, %c192_i32 : i32
                %182 = arith.addi %181, %c128_i32_161 : i32
                %183 = arith.index_cast %182 : i32 to index
                %184 = ascvec.loadalign %arg37[%183] : memref<1536xf32, 1>, index -> <f32>
                %185 = ascvec.loadalign %arg38[%c128] : memref<192xf32, 1>, index -> <f32>
                %186 = ascvec.mul %184, %185, %180 : <f32>
                %187 = ascvec.mul_scalar %186, %arg39, %180 : <f32>, f32
                %188 = arith.muli %arg42, %c64_i32 : i32
                %189 = arith.index_cast %188 : i32 to index
                ascvec.storealign %arg40[%189], %187, %180 : memref<512xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %c17_i64, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %106, 0 {cannir.buffer.generated_sync}
            %167 = ascscalar.memref_address %alloc_116 : memref<512xbf16, 1>
            %168 = ascscalar.memref_address %alloc_117 : memref<512xbf16, 1>
            %169 = arith.select %102, %167, %168 : i64
            %170 = ascscalar.bind_local_ptr %169 : memref<512xbf16, 1>
            %171 = arith.select %102, %c18_i64, %c19_i64 : i64
            scf.for %arg37 = %c0_i64 to %101 step %c1_i64  : i64 {
              %180 = arith.muli %arg37, %c64_i64 : i64
              %181 = arith.trunci %180 : i64 to i32
              %182 = arith.index_cast %181 : i32 to index
              ascvec.get_buf v, %c20_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %c21_i64, 0 {cannir.buffer.generated_sync}
              ascvec.get_buf v, %171, 0 {cannir.buffer.generated_sync}
              %183 = arith.index_cast %180 : i64 to index
              ascvec.vf(%alloc_120, %182, %alloc_118, %183, %alloc_119, %183, %170, %183 : memref<512xf32, 1>, index, memref<512xbf16, 1>, index, memref<512xbf16, 1>, index, memref<512xbf16, 1>, index) {
              ^bb0(%arg38: memref<512xf32, 1>, %arg39: index, %arg40: memref<512xbf16, 1>, %arg41: index, %arg42: memref<512xbf16, 1>, %arg43: index, %arg44: memref<512xbf16, 1>, %arg45: index):
                %c32 = arith.constant 32 : index
                %c1_i32_159 = arith.constant 1 : i32
                %c0_i32_160 = arith.constant 0 : i32
                %c32_i32_161 = arith.constant 32 : i32
                %184 = ascvec.create_mask : !ascvec.mask
                %mask, %remaining_out = ascvec.update_mask %c32_i32_161 : !ascvec.mask
                %185 = ascvec.loadalign %arg38[%arg39] : memref<512xf32, 1>, index -> <f32>
                %186 = ascvec.arange %c0_i32_160 : i32 -> <ui32>
                %187 = ascvec.mul_scalar %186, %c1_i32_159, %184 : <ui32>, i32
                %188 = ascvec.add_scalar %187, %c32_i32_161, %184 : <ui32>, i32
                %189 = ascvec.gather_reg %185, %187 : <f32>, <ui32> -> <f32>
                %190 = ascvec.gather_reg %185, %188 : <f32>, <ui32> -> <f32>
                %191 = ascvec.loadalign_unpack %arg40[%arg41] {mode = "b16_to_b32"} : memref<512xbf16, 1>, index -> <bf16>
                %192 = ascvec.cast %191, %184 {rounding = 6 : i32} : <bf16> -> <f32>
                %193 = ascvec.loadalign_unpack %arg42[%arg43] {mode = "b16_to_b32"} : memref<512xbf16, 1>, index -> <bf16>
                %194 = ascvec.cast %193, %184 {rounding = 6 : i32} : <bf16> -> <f32>
                %195 = ascvec.add_scalar %186, %c32_i32_161, %184 : <ui32>, i32
                %196 = ascvec.gather_reg %192, %195 : <f32>, <ui32> -> <f32>
                %197 = ascvec.gather_reg %194, %195 : <f32>, <ui32> -> <f32>
                %198 = ascvec.mul %189, %194, %mask : <f32>
                %199 = ascvec.mul %190, %192, %mask : <f32>
                %200 = ascvec.sub %198, %199, %mask : <f32>
                %201 = ascvec.mul %189, %196, %mask : <f32>
                %202 = ascvec.mul %190, %197, %mask : <f32>
                %203 = ascvec.add %201, %202, %mask : <f32>
                %204 = ascvec.cast %200, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                %205 = ascvec.cast %203, %mask {reg_layout = 0 : i32} : <f32> -> <bf16>
                ascvec.storealign_pack %arg44[%arg45], %204, %mask {mode = "b32_to_b16"} : memref<512xbf16, 1>, index, <bf16>
                %206 = arith.addi %arg45, %c32 : index
                %c0_162 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                scf.for %arg46 = %c0_162 to %c1 step %c1 {
                  ascvec.storealign_pack %arg44[%206], %205, %mask {mode = "b32_to_b16"} : memref<512xbf16, 1>, index, <bf16>
                }
              } {cannir.vf_mode = "simd"}
              ascvec.release_buf v, %171, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c21_i64, 0 {cannir.buffer.generated_sync}
              ascvec.release_buf v, %c20_i64, 0 {cannir.buffer.generated_sync}
            }
            %172 = arith.muli %78, %c64_i64 overflow<nsw> : i64
            %173 = arith.addi %146, %172 : i64
            %174 = arith.muli %173, %c16_i64 : i64
            %175 = ascscalar.memref_address %alloc_18 : memref<1xbf16>
            %176 = arith.divui %174, %c8_i64 : i64
            %177 = arith.addi %175, %176 : i64
            %178 = ascscalar.bind_gm_ptr %177 : memref<1xbf16>
            %179 = ascscalar.bind_local_ptr %169 : memref<1xbf16, 1>
            scf.if %135 {
              ascvec.get_buf mte3, %171, 0 {cannir.buffer.generated_sync}
              %180 = arith.trunci %115 : i64 to i32
              ascvec.copy_ub2gm %178[%c0], %179[%c0], %180, %c128_i32, %c256_i64, %c128_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
              ascvec.release_buf mte3, %171, 0 {cannir.buffer.generated_sync}
            }
            scf.yield %108 : i64
          }
        }
        ascvec.sync_block_arrive mte3, %c5_i64
      }
      %50 = arith.remsi %arg33, %c2_i64 : i64
      %51 = arith.cmpi ne, %50, %c0_i64 : i64
      %52 = arith.cmpi slt, %50, %c0_i64 : i64
      %53 = arith.andi %51, %52 : i1
      %54 = arith.select %53, %c2_i64, %c0_i64 : i64
      %55 = arith.addi %50, %54 overflow<nsw> : i64
      %56 = arith.divsi %arg33, %c2_i64 : i64
      %57 = arith.extui %53 : i1 to i64
      %58 = arith.subi %56, %57 : i64
      %59 = arith.muli %58, %c16_i64 : i64
      %60 = arith.addi %59, %18 : i64
      %61 = arith.addi %59, %c16_i64 : i64
      %62 = arith.minsi %61, %36 : i64
      ascvec.sync_block_wait s, %c3_i64
      %63 = scf.for %arg35 = %60 to %62 step %c2_i64 iter_args(%arg36 = %arg34) -> (i64)  : i64 {
        %64 = arith.cmpi eq, %arg36, %c0_i64 : i64
        %65 = ascscalar.memref_address %alloc_94 : memref<512xbf16, 1>
        %66 = ascscalar.memref_address %alloc_95 : memref<512xbf16, 1>
        %67 = arith.select %64, %65, %66 : i64
        %68 = arith.cmpi ne, %arg36, %c0_i64 : i64
        %69 = arith.extui %68 : i1 to i64
        %70 = arith.addi %arg36, %c1_i64 overflow<nsw> : i64
        %71 = arith.remui %70, %c2_i64 : i64
        %72 = arith.minsi %arg35, %36 : i64
        %73 = arith.subi %36, %72 : i64
        %74 = arith.minsi %73, %c1_i64 : i64
        %75 = arith.cmpi sle, %arg35, %36 : i64
        %76 = arith.select %75, %74, %c0_i64 : i64
        %77 = arith.muli %arg35, %c2304_i64 : i64
        %78 = arith.muli %55, %c1152_i64 overflow<nsw> : i64
        %79 = arith.addi %77, %78 : i64
        %80 = ascscalar.bind_local_ptr %67 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %69, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%80, %alloc_19, %c0_i32, %c0_i32, %c0_i64, %79, %76, %c1024_i64, %c2304_i64, %c1024_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %69, 0 {cannir.buffer.generated_sync}
        %81 = ascscalar.bind_local_ptr %67 : memref<512xbf16, 1>
        %82 = ascscalar.memref_address %alloc_96 : memref<576xbf16, 1>
        %83 = ascscalar.memref_address %alloc_97 : memref<576xbf16, 1>
        %84 = arith.select %64, %82, %83 : i64
        %85 = ascscalar.bind_local_ptr %84 : memref<576xbf16, 1>
        ascvec.get_buf v, %69, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%81, %85 : memref<512xbf16, 1>, memref<576xbf16, 1>) {
        ^bb0(%arg37: memref<512xbf16, 1>, %arg38: memref<576xbf16, 1>):
          %c64_i32 = arith.constant 64 : i32
          %c512_i32 = arith.constant 512 : i32
          %c0_i32_159 = arith.constant 0 : i32
          %105 = ascvec.create_mask : !ascvec.mask
          scf.for %arg39 = %c0_i32_159 to %c512_i32 step %c64_i32  : i32 {
            %106 = arith.index_cast %arg39 : i32 to index
            %107 = ascvec.loadalign_unpack %arg37[%106] {mode = "b16_to_b32"} : memref<512xbf16, 1>, index -> <bf16>
            ascvec.storealign_pack %arg38[%106], %107, %105 {mode = "b32_to_b16"} : memref<576xbf16, 1>, index, <bf16>
          }
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %69, 0 {cannir.buffer.generated_sync}
        %86 = ascscalar.memref_address %alloc_98 : memref<64xbf16, 1>
        %87 = ascscalar.memref_address %alloc_99 : memref<64xbf16, 1>
        %88 = arith.select %64, %86, %87 : i64
        %89 = arith.select %64, %c2_i64, %c3_i64 : i64
        %90 = arith.muli %arg35, %c256_i64 : i64
        %91 = arith.muli %55, %c128_i64 overflow<nsw> : i64
        %92 = arith.addi %90, %91 : i64
        %93 = ascscalar.bind_local_ptr %88 : memref<1xbf16, 1>
        ascvec.get_buf mte2, %89, 0 {cannir.buffer.generated_sync}
        "ascvec.copy_gm2ub_plan"(%93, %alloc_18, %c0_i32, %c0_i32, %c0_i64, %92, %76, %c128_i64, %c256_i64, %c128_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
        ascvec.release_buf mte2, %89, 0 {cannir.buffer.generated_sync}
        %94 = ascscalar.bind_local_ptr %88 : memref<64xbf16, 1>
        ascvec.get_buf v, %89, 0 {cannir.buffer.generated_sync}
        ascvec.vf(%94, %85 : memref<64xbf16, 1>, memref<576xbf16, 1>) {
        ^bb0(%arg37: memref<64xbf16, 1>, %arg38: memref<576xbf16, 1>):
          %c512 = arith.constant 512 : index
          %c0_159 = arith.constant 0 : index
          %c0_160 = arith.constant 0 : index
          %c1 = arith.constant 1 : index
          scf.for %arg39 = %c0_160 to %c1 step %c1 {
            %105 = ascvec.create_mask : !ascvec.mask
            %106 = ascvec.loadalign_unpack %arg37[%c0_159] {mode = "b16_to_b32"} : memref<64xbf16, 1>, index -> <bf16>
            ascvec.storealign_pack %arg38[%c512], %106, %105 {mode = "b32_to_b16"} : memref<576xbf16, 1>, index, <bf16>
          }
        } {cannir.vf_mode = "simd"}
        ascvec.release_buf v, %89, 0 {cannir.buffer.generated_sync}
        %95 = arith.muli %arg35, %c1152_i64 : i64
        %96 = arith.muli %55, %c576_i64 overflow<nsw> : i64
        %97 = arith.addi %96, %c512_i64 overflow<nsw> : i64
        %98 = arith.addi %95, %97 : i64
        %99 = arith.muli %98, %c16_i64 : i64
        %100 = ascscalar.memref_address %alloc_19 : memref<1xbf16>
        %101 = arith.divui %99, %c8_i64 : i64
        %102 = arith.addi %100, %101 : i64
        %103 = ascscalar.bind_gm_ptr %102 : memref<1xbf16>
        %104 = arith.cmpi ugt, %76, %c0_i64 : i64
        scf.if %104 {
          ascvec.get_buf mte3, %89, 0 {cannir.buffer.generated_sync}
          %105 = arith.trunci %76 : i64 to i32
          ascvec.copy_ub2gm %103[%c0], %93[%c0], %105, %c128_i32, %c2304_i64, %c128_i32 l2_cache_ctl %c0_i32 : memref<1xbf16>, memref<1xbf16, 1>
          ascvec.release_buf mte3, %89, 0 {cannir.buffer.generated_sync}
        }
        scf.if %6 {
          ascvec.vf(%alloc_100, %85 : memref<128xf32, 1>, memref<576xbf16, 1>) {
          ^bb0(%arg37: memref<128xf32, 1>, %arg38: memref<576xbf16, 1>):
            %c64 = arith.constant 64 : index
            %c64_i32 = arith.constant 64 : i32
            %c576_i32_159 = arith.constant 576 : i32
            %c0_i32_160 = arith.constant 0 : i32
            %cst = arith.constant 0.000000e+00 : f32
            %cst_161 = arith.constant 1.000000e+00 : f32
            %c0_162 = arith.constant 0 : index
            %105 = ascvec.create_mask : !ascvec.mask
            ascvec.mem_bar vst_vld
            %106 = ascvec.loadalign %arg37[%c0_162] : memref<128xf32, 1>, index -> <f32>
            %107 = ascvec.loadalign %arg37[%c64] : memref<128xf32, 1>, index -> <f32>
            %108 = ascvec.duplicate_scalar %cst_161, %105 : f32 -> <f32>
            %109 = ascvec.duplicate_scalar %cst, %105 : f32 -> <f32>
            %110:2 = scf.for %arg39 = %c0_i32_160 to %c576_i32_159 step %c64_i32 iter_args(%arg40 = %106, %arg41 = %107) -> (!ascvec.vreg<f32>, !ascvec.vreg<f32>)  : i32 {
              %111 = arith.index_cast %arg39 : i32 to index
              %112 = ascvec.loadalign_unpack %arg38[%111] {mode = "b16_to_b32"} : memref<576xbf16, 1>, index -> <bf16>
              %113 = ascvec.cast %112, %105 {rounding = 6 : i32} : <bf16> -> <f32>
              %114 = ascvec.abs %113, %105 : <f32>
              %115 = ascvec.max %arg40, %114, %105 : <f32>
              %116 = ascvec.cmp_ne %113, %113, %105 : <f32>, <f32> -> !ascvec.mask
              %117 = ascvec.select %108, %109, %116 : <f32>
              %118 = ascvec.max %arg41, %117, %105 : <f32>
              scf.yield %115, %118 : !ascvec.vreg<f32>, !ascvec.vreg<f32>
            }
            %c0_163 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg39 = %c0_163 to %c1 step %c1 {
              ascvec.storealign %arg37[%c0_162], %110#0, %105 : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg37[%c64], %110#1, %105 : memref<128xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
        }
        scf.yield %71 : i64
      }
      ascvec.sync_block_arrive mte3, %c4_i64
      scf.yield %63 : i64
    }
    scf.if %6 {
      %48 = cannir.extract_layout %arg16 : !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
      %49 = cannir.extract_shape %48 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
      %50 = cannir.extract_dim %49[0] : !cannir.shape<(-1, 1024)>
      %51 = arith.muli %50, %c1024_i64 : i64
      %52 = arith.remsi %51, %c8_i64 : i64
      %53 = arith.cmpi ne, %52, %c0_i64 : i64
      %54 = arith.cmpi slt, %52, %c0_i64 : i64
      %55 = arith.andi %53, %54 : i1
      %56 = arith.divsi %51, %c8_i64 : i64
      %57 = arith.extui %55 : i1 to i64
      %58 = arith.subi %56, %57 : i64
      ascvec.get_buf v, %c22_i64, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%alloc_121, %alloc_100 : memref<8xf32, 1>, memref<128xf32, 1>) {
      ^bb0(%arg33: memref<8xf32, 1>, %arg34: memref<128xf32, 1>):
        %c1 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %c0_159 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c8_i32 = arith.constant 8 : i32
        %71 = ascvec.create_mask : !ascvec.mask
        %mask, %remaining_out = ascvec.update_mask %c8_i32 : !ascvec.mask
        %72 = ascvec.duplicate_scalar %cst, %mask : f32 -> <f32>
        ascvec.storealign %arg33[%c0_159], %72, %mask : memref<8xf32, 1>, index, <f32>
        ascvec.mem_bar vst_vld
        %c0_160 = arith.constant 0 : index
        %c1_161 = arith.constant 1 : index
        scf.for %arg35 = %c0_160 to %c1_161 step %c1_161 {
          %73 = ascvec.loadalign %arg34[%c0_159] : memref<128xf32, 1>, index -> <f32>
          %74 = ascvec.reduce_max %73, %71 : <f32>
          %75 = ascvec.duplicate %74, %71 : <f32> -> <f32>
          %76 = ascvec.loadalign %arg34[%c64] : memref<128xf32, 1>, index -> <f32>
          %77 = ascvec.reduce_max %76, %71 : <f32>
          %78 = ascvec.duplicate %77, %71 : <f32> -> <f32>
          ascvec.storealign_1st %arg33[%c0_159], %75 : memref<8xf32, 1>, index, <f32>
          ascvec.storealign_1st %arg33[%c1], %78 : memref<8xf32, 1>, index, <f32>
        }
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c22_i64, 0 {cannir.buffer.generated_sync}
      %59 = arith.minsi %19, %58 : i64
      %60 = arith.subi %58, %59 : i64
      %61 = arith.minsi %60, %c1_i64 : i64
      %62 = arith.cmpi sle, %19, %58 : i64
      %63 = arith.select %62, %61, %c0_i64 : i64
      %64 = arith.muli %19, %c256_i64 : i64
      %65 = ascscalar.memref_address %alloc_10 : memref<1xf32>
      %66 = arith.divui %64, %c8_i64 : i64
      %67 = arith.addi %65, %66 : i64
      %68 = ascscalar.bind_gm_ptr %67 : memref<1xf32>
      %69 = ascscalar.bitcast_memref %alloc_121 byte_offset = %c0_i64 : memref<8xf32, 1> -> memref<?xf32, 1>
      %70 = arith.cmpi ugt, %63, %c0_i64 : i64
      scf.if %70 {
        ascvec.get_buf mte3, %c22_i64, 0 {cannir.buffer.generated_sync}
        %71 = arith.trunci %63 : i64 to i32
        ascvec.copy_ub2gm %68[%c0], %69[%c0], %71, %c32_i32, %c32_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
        ascvec.release_buf mte3, %c22_i64, 0 {cannir.buffer.generated_sync}
      }
    }
    ascvec.sync_all
    ascvec.sync_block_arrive mte3, %c0_i64
    ascvec.sync_block_wait s, %c2_i64
    scf.if %6 {
      %48 = cannir.extract_layout %arg16 : !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
      %49 = cannir.extract_shape %48 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
      %50 = cannir.extract_dim %49[0] : !cannir.shape<(-1, 1024)>
      %51 = arith.muli %50, %c1024_i64 : i64
      %52 = arith.remsi %51, %c8_i64 : i64
      %53 = arith.cmpi ne, %52, %c0_i64 : i64
      %54 = arith.cmpi slt, %52, %c0_i64 : i64
      %55 = arith.andi %53, %54 : i1
      %56 = arith.divsi %51, %c8_i64 : i64
      %57 = arith.extui %55 : i1 to i64
      %58 = arith.subi %56, %57 : i64
      %59 = arith.maxsi %5, %c1_i64 : i64
      %60 = arith.divsi %58, %59 : i64
      %61 = arith.minsi %60, %c0_i64 : i64
      %62 = arith.muli %61, %5 : i64
      %63 = arith.subi %58, %62 : i64
      %64 = arith.minsi %5, %63 : i64
      %65 = arith.cmpi sge, %60, %c0_i64 : i64
      %66 = arith.select %65, %64, %c0_i64 : i64
      %67 = arith.muli %66, %c32_i64 : i64
      %68 = arith.cmpi eq, %67, %c0_i64 : i64
      %69 = arith.cmpi ne, %67, %c0_i64 : i64
      %70 = arith.extui %69 : i1 to i64
      %71 = arith.select %68, %c4_i64, %67 : i64
      ascvec.get_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%alloc_123, %alloc_10, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %70, %71, %c0_i64, %71, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 32 : i64, ub_capacity = 4096 : i64}> : (memref<1024xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
      %72 = arith.trunci %5 : i64 to i32
      ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
      ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
      ascvec.vf(%alloc_123, %72, %alloc_122 : memref<1024xf32, 1>, i32, memref<1xf32, 1>) {
      ^bb0(%arg33: memref<1024xf32, 1>, %arg34: i32, %arg35: memref<1xf32, 1>):
        %c2143289344_i32 = arith.constant 2143289344 : i32
        %cst = arith.constant 3.276800e+04 : f32
        %c0_i32_159 = arith.constant 0 : i32
        %c1_i32_160 = arith.constant 1 : i32
        %c0_161 = arith.constant 0 : index
        %c8_i32 = arith.constant 8 : i32
        %cst_162 = arith.constant 0.000000e+00 : f32
        %89 = ascvec.create_mask : !ascvec.mask
        %90 = ascvec.duplicate_scalar %cst_162, %89 : f32 -> <f32>
        %91:2 = scf.for %arg36 = %c0_i32_159 to %arg34 step %c1_i32_160 iter_args(%arg37 = %90, %arg38 = %90) -> (!ascvec.vreg<f32>, !ascvec.vreg<f32>)  : i32 {
          %92 = arith.muli %arg36, %c8_i32 : i32
          %93 = arith.index_cast %92 : i32 to index
          %94 = ascvec.loadalign_brc %arg33[%93] : memref<1024xf32, 1>, index -> <f32>
          %95 = ascvec.max %arg37, %94, %89 : <f32>
          %96 = arith.addi %92, %c1_i32_160 : i32
          %97 = arith.index_cast %96 : i32 to index
          %98 = ascvec.loadalign_brc %arg33[%97] : memref<1024xf32, 1>, index -> <f32>
          %99 = ascvec.max %arg38, %98, %89 : <f32>
          scf.yield %95, %99 : !ascvec.vreg<f32>, !ascvec.vreg<f32>
        }
        %c0_163 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        scf.for %arg36 = %c0_163 to %c1 step %c1 {
          %92 = ascvec.duplicate_scalar %cst, %89 : f32 -> <f32>
          %93 = ascvec.div %91#0, %92, %89 {precision_mode = true} : <f32>
          %94 = ascvec.duplicate_scalar %c2143289344_i32, %89 : i32 -> <ui32>
          %95 = ascvec.reinterpret %94 : <ui32> -> <f32>
          %96 = ascvec.cmp_gt_scalar %91#1, %cst_162, %89 : <f32>, f32 -> !ascvec.mask
          %97 = ascvec.select %95, %93, %96 : <f32>
          ascvec.storealign_1st %arg35[%c0_161], %97 : memref<1xf32, 1>, index, <f32>
        }
      } {cannir.vf_mode = "simd"}
      ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
      ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
      %73 = arith.cmpi eq, %0, %c0_i64 : i64
      %74 = arith.cmpi eq, %18, %c0_i64 : i64
      %75 = arith.select %73, %74, %false : i1
      scf.if %75 {
        %89 = ascscalar.bitcast_memref %alloc_122 byte_offset = %c0_i64 : memref<1xf32, 1> -> memref<?xf32, 1>
        ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
        ascvec.copy_ub2gm %alloc_21[%c0], %89[%c0], %c1_i32, %c4_i32, %c0_i64, %c4_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
        ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
      }
      %76 = cannir.extract_layout %arg26 : !cannir.tensor<<gm, bf16>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>
      %77 = cannir.extract_shape %76 : <<(-1, 576)>, <(576, 1)>> -> <(-1, 576)>
      %78 = cannir.extract_dim %77[0] : !cannir.shape<(-1, 576)>
      %79 = arith.remsi %78, %c16_i64 : i64
      %80 = arith.cmpi ne, %79, %c0_i64 : i64
      %81 = arith.cmpi slt, %79, %c0_i64 : i64
      %82 = arith.andi %80, %81 : i1
      %83 = arith.select %82, %c16_i64, %c0_i64 : i64
      %84 = arith.addi %79, %83 overflow<nsw> : i64
      %85 = arith.cmpi eq, %84, %c0_i64 : i64
      %86 = arith.muli %1, %c32_i64 : i64
      %87 = arith.cmpi sge, %78, %86 : i64
      %88 = arith.select %85, %87, %false : i1
      scf.if %88 {
        %89 = arith.divsi %78, %c16_i64 : i64
        %90 = arith.extui %82 : i1 to i64
        %91 = arith.subi %89, %90 : i64
        %92 = scf.for %arg33 = %19 to %91 step %5 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
          %93 = arith.cmpi eq, %arg34, %c0_i64 : i64
          %94 = ascscalar.memref_address %alloc_124 : memref<9216xbf16, 1>
          %95 = ascscalar.memref_address %alloc_125 : memref<9216xbf16, 1>
          %96 = arith.select %93, %94, %95 : i64
          %97 = arith.select %93, %c2_i64, %c3_i64 : i64
          %98 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
          %99 = arith.remui %98, %c2_i64 : i64
          %100 = arith.minsi %arg33, %91 : i64
          %101 = arith.subi %91, %100 : i64
          %102 = arith.minsi %101, %c1_i64 : i64
          %103 = arith.cmpi sle, %arg33, %91 : i64
          %104 = arith.select %103, %102, %c0_i64 : i64
          %105 = arith.muli %arg33, %c18432_i64 : i64
          %106 = arith.muli %104, %c18432_i64 : i64
          %107 = arith.cmpi eq, %106, %c0_i64 : i64
          %108 = arith.cmpi ne, %106, %c0_i64 : i64
          %109 = arith.extui %108 : i1 to i64
          %110 = arith.select %107, %c2_i64, %106 : i64
          %111 = ascscalar.bind_local_ptr %96 : memref<1xbf16, 1>
          ascvec.get_buf mte2, %97, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%111, %alloc_19, %c0_i32, %c0_i32, %c0_i64, %105, %109, %110, %c0_i64, %110, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %97, 0 {cannir.buffer.generated_sync}
          %112 = ascscalar.bind_local_ptr %96 : memref<9216xbf16, 1>
          %113 = ascscalar.memref_address %alloc_126 : memref<9216xui8, 1>
          %114 = ascscalar.memref_address %alloc_127 : memref<9216xui8, 1>
          %115 = arith.select %93, %113, %114 : i64
          %116 = ascscalar.bind_local_ptr %115 : memref<9216xui8, 1>
          %117 = arith.select %93, %c4_i64, %c5_i64 : i64
          ascvec.get_buf v, %97, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %117, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_122, %112, %116 : memref<1xf32, 1>, memref<9216xbf16, 1>, memref<9216xui8, 1>) {
          ^bb0(%arg35: memref<1xf32, 1>, %arg36: memref<9216xbf16, 1>, %arg37: memref<9216xui8, 1>):
            %c64_i32 = arith.constant 64 : i32
            %c9216_i32_159 = arith.constant 9216 : i32
            %c0_i32_160 = arith.constant 0 : i32
            %cst = arith.constant 1.1920929E-7 : f32
            %cst_161 = arith.constant 3.276800e+04 : f32
            %cst_162 = arith.constant -3.276800e+04 : f32
            %cst_163 = arith.constant 1.000000e+00 : f32
            %cst_164 = arith.constant 0.000000e+00 : f32
            %c0_165 = arith.constant 0 : index
            %125 = ascvec.create_mask : !ascvec.mask
            ascvec.mem_bar vst_vld
            %126 = ascvec.loadalign_brc %arg35[%c0_165] : memref<1xf32, 1>, index -> <f32>
            %127 = ascvec.cmp_eq_scalar %126, %cst_164, %125 : <f32>, f32 -> !ascvec.mask
            %128 = ascvec.duplicate_scalar %cst_163, %125 : f32 -> <f32>
            %129 = ascvec.select %128, %126, %127 : <f32>
            scf.for %arg38 = %c0_i32_160 to %c9216_i32_159 step %c64_i32  : i32 {
              %130 = arith.index_cast %arg38 : i32 to index
              %131 = ascvec.loadalign_unpack %arg36[%130] {mode = "b16_to_b32"} : memref<9216xbf16, 1>, index -> <bf16>
              %132 = ascvec.cast %131, %125 {rounding = 6 : i32} : <bf16> -> <f32>
              %133 = ascvec.div %132, %129, %125 : <f32>
              %134 = ascvec.duplicate_scalar %cst_164, %125 : f32 -> <f32>
              %135 = ascvec.select %134, %133, %127 : <f32>
              %136 = ascvec.cmp_eq %135, %135, %125 : <f32>, <f32> -> !ascvec.mask
              %137 = ascvec.select %135, %134, %136 : <f32>
              %138 = ascvec.max_scalar %137, %cst_162, %125 : <f32>, f32
              %139 = ascvec.min_scalar %138, %cst_161, %125 : <f32>, f32
              %140 = ascvec.abs %139, %125 : <f32>
              %141 = ascvec.cmp_lt_scalar %140, %cst, %125 : <f32>, f32 -> !ascvec.mask
              %142 = ascvec.select %134, %139, %141 : <f32>
              %143 = ascvec.cast %142, %125 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
              %144 = ascvec.reinterpret %143 : <!ascdtype.hifloat8> -> <f8E4M3FN>
              ascvec.storealign_pack %arg37[%130], %144, %125 {mode = "b32_to_b8"} : memref<9216xui8, 1>, index, <f8E4M3FN>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %117, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %97, 0 {cannir.buffer.generated_sync}
          %118 = arith.muli %arg33, %c73728_i64 : i64
          %119 = ascscalar.memref_address %alloc_20 : memref<1xui8>
          %120 = arith.divui %118, %c8_i64 : i64
          %121 = arith.addi %119, %120 : i64
          %122 = ascscalar.bind_gm_ptr %121 : memref<1xui8>
          %123 = ascscalar.bind_local_ptr %115 : memref<1xui8, 1>
          %124 = arith.cmpi ugt, %104, %c0_i64 : i64
          scf.if %124 {
            ascvec.get_buf mte3, %117, 0 {cannir.buffer.generated_sync}
            %125 = arith.trunci %104 : i64 to i32
            ascvec.copy_ub2gm %122[%c0], %123[%c0], %125, %c9216_i32, %c9216_i64, %c9216_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
            ascvec.release_buf mte3, %117, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %99 : i64
        }
      } else {
        %89 = arith.remsi %78, %c8_i64 : i64
        %90 = arith.cmpi ne, %89, %c0_i64 : i64
        %91 = arith.cmpi slt, %89, %c0_i64 : i64
        %92 = arith.andi %90, %91 : i1
        %93 = arith.select %92, %c8_i64, %c0_i64 : i64
        %94 = arith.addi %89, %93 overflow<nsw> : i64
        %95 = arith.cmpi eq, %94, %c0_i64 : i64
        %96 = arith.cmpi sge, %78, %8 : i64
        %97 = arith.select %95, %96, %false : i1
        scf.if %97 {
          %98 = arith.divsi %78, %c8_i64 : i64
          %99 = arith.extui %92 : i1 to i64
          %100 = arith.subi %98, %99 : i64
          %101 = scf.for %arg33 = %19 to %100 step %5 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
            %102 = arith.cmpi eq, %arg34, %c0_i64 : i64
            %103 = ascscalar.memref_address %alloc_128 : memref<4608xbf16, 1>
            %104 = ascscalar.memref_address %alloc_129 : memref<4608xbf16, 1>
            %105 = arith.select %102, %103, %104 : i64
            %106 = arith.select %102, %c6_i64, %c7_i64 : i64
            %107 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
            %108 = arith.remui %107, %c2_i64 : i64
            %109 = arith.minsi %arg33, %100 : i64
            %110 = arith.subi %100, %109 : i64
            %111 = arith.minsi %110, %c1_i64 : i64
            %112 = arith.cmpi sle, %arg33, %100 : i64
            %113 = arith.select %112, %111, %c0_i64 : i64
            %114 = arith.muli %arg33, %c9216_i64 : i64
            %115 = arith.muli %113, %c9216_i64 : i64
            %116 = arith.cmpi eq, %115, %c0_i64 : i64
            %117 = arith.cmpi ne, %115, %c0_i64 : i64
            %118 = arith.extui %117 : i1 to i64
            %119 = arith.select %116, %c2_i64, %115 : i64
            %120 = ascscalar.bind_local_ptr %105 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %106, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%120, %alloc_19, %c0_i32, %c0_i32, %c0_i64, %114, %118, %119, %c0_i64, %119, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %106, 0 {cannir.buffer.generated_sync}
            %121 = ascscalar.bind_local_ptr %105 : memref<4608xbf16, 1>
            %122 = ascscalar.memref_address %alloc_130 : memref<4608xui8, 1>
            %123 = ascscalar.memref_address %alloc_131 : memref<4608xui8, 1>
            %124 = arith.select %102, %122, %123 : i64
            %125 = ascscalar.bind_local_ptr %124 : memref<4608xui8, 1>
            %126 = arith.select %102, %c8_i64, %c9_i64 : i64
            ascvec.get_buf v, %106, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %126, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_122, %121, %125 : memref<1xf32, 1>, memref<4608xbf16, 1>, memref<4608xui8, 1>) {
            ^bb0(%arg35: memref<1xf32, 1>, %arg36: memref<4608xbf16, 1>, %arg37: memref<4608xui8, 1>):
              %c64_i32 = arith.constant 64 : i32
              %c4608_i32_159 = arith.constant 4608 : i32
              %c0_i32_160 = arith.constant 0 : i32
              %cst = arith.constant 1.1920929E-7 : f32
              %cst_161 = arith.constant 3.276800e+04 : f32
              %cst_162 = arith.constant -3.276800e+04 : f32
              %cst_163 = arith.constant 1.000000e+00 : f32
              %cst_164 = arith.constant 0.000000e+00 : f32
              %c0_165 = arith.constant 0 : index
              %134 = ascvec.create_mask : !ascvec.mask
              ascvec.mem_bar vst_vld
              %135 = ascvec.loadalign_brc %arg35[%c0_165] : memref<1xf32, 1>, index -> <f32>
              %136 = ascvec.cmp_eq_scalar %135, %cst_164, %134 : <f32>, f32 -> !ascvec.mask
              %137 = ascvec.duplicate_scalar %cst_163, %134 : f32 -> <f32>
              %138 = ascvec.select %137, %135, %136 : <f32>
              scf.for %arg38 = %c0_i32_160 to %c4608_i32_159 step %c64_i32  : i32 {
                %139 = arith.index_cast %arg38 : i32 to index
                %140 = ascvec.loadalign_unpack %arg36[%139] {mode = "b16_to_b32"} : memref<4608xbf16, 1>, index -> <bf16>
                %141 = ascvec.cast %140, %134 {rounding = 6 : i32} : <bf16> -> <f32>
                %142 = ascvec.div %141, %138, %134 : <f32>
                %143 = ascvec.duplicate_scalar %cst_164, %134 : f32 -> <f32>
                %144 = ascvec.select %143, %142, %136 : <f32>
                %145 = ascvec.cmp_eq %144, %144, %134 : <f32>, <f32> -> !ascvec.mask
                %146 = ascvec.select %144, %143, %145 : <f32>
                %147 = ascvec.max_scalar %146, %cst_162, %134 : <f32>, f32
                %148 = ascvec.min_scalar %147, %cst_161, %134 : <f32>, f32
                %149 = ascvec.abs %148, %134 : <f32>
                %150 = ascvec.cmp_lt_scalar %149, %cst, %134 : <f32>, f32 -> !ascvec.mask
                %151 = ascvec.select %143, %148, %150 : <f32>
                %152 = ascvec.cast %151, %134 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
                %153 = ascvec.reinterpret %152 : <!ascdtype.hifloat8> -> <f8E4M3FN>
                ascvec.storealign_pack %arg37[%139], %153, %134 {mode = "b32_to_b8"} : memref<4608xui8, 1>, index, <f8E4M3FN>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %126, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %106, 0 {cannir.buffer.generated_sync}
            %127 = arith.muli %arg33, %c36864_i64 : i64
            %128 = ascscalar.memref_address %alloc_20 : memref<1xui8>
            %129 = arith.divui %127, %c8_i64 : i64
            %130 = arith.addi %128, %129 : i64
            %131 = ascscalar.bind_gm_ptr %130 : memref<1xui8>
            %132 = ascscalar.bind_local_ptr %124 : memref<1xui8, 1>
            %133 = arith.cmpi ugt, %113, %c0_i64 : i64
            scf.if %133 {
              ascvec.get_buf mte3, %126, 0 {cannir.buffer.generated_sync}
              %134 = arith.trunci %113 : i64 to i32
              ascvec.copy_ub2gm %131[%c0], %132[%c0], %134, %c4608_i32, %c4608_i64, %c4608_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
              ascvec.release_buf mte3, %126, 0 {cannir.buffer.generated_sync}
            }
            scf.yield %108 : i64
          }
        } else {
          %98 = scf.for %arg33 = %19 to %78 step %5 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
            %99 = arith.cmpi eq, %arg34, %c0_i64 : i64
            %100 = ascscalar.memref_address %alloc_132 : memref<576xbf16, 1>
            %101 = ascscalar.memref_address %alloc_133 : memref<576xbf16, 1>
            %102 = arith.select %99, %100, %101 : i64
            %103 = arith.select %99, %c10_i64, %c11_i64 : i64
            %104 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
            %105 = arith.remui %104, %c2_i64 : i64
            %106 = arith.minsi %arg33, %78 : i64
            %107 = arith.subi %78, %106 : i64
            %108 = arith.minsi %107, %c1_i64 : i64
            %109 = arith.cmpi sle, %arg33, %78 : i64
            %110 = arith.select %109, %108, %c0_i64 : i64
            %111 = arith.muli %arg33, %c1152_i64 : i64
            %112 = arith.muli %110, %c1152_i64 : i64
            %113 = arith.cmpi eq, %112, %c0_i64 : i64
            %114 = arith.cmpi ne, %112, %c0_i64 : i64
            %115 = arith.extui %114 : i1 to i64
            %116 = arith.select %113, %c2_i64, %112 : i64
            %117 = ascscalar.bind_local_ptr %102 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %103, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%117, %alloc_19, %c0_i32, %c0_i32, %c0_i64, %111, %115, %116, %c0_i64, %116, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %103, 0 {cannir.buffer.generated_sync}
            %118 = ascscalar.bind_local_ptr %102 : memref<576xbf16, 1>
            %119 = ascscalar.memref_address %alloc_134 : memref<576xui8, 1>
            %120 = ascscalar.memref_address %alloc_135 : memref<576xui8, 1>
            %121 = arith.select %99, %119, %120 : i64
            %122 = ascscalar.bind_local_ptr %121 : memref<576xui8, 1>
            %123 = arith.select %99, %c12_i64, %c13_i64 : i64
            ascvec.get_buf v, %103, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %123, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_122, %118, %122 : memref<1xf32, 1>, memref<576xbf16, 1>, memref<576xui8, 1>) {
            ^bb0(%arg35: memref<1xf32, 1>, %arg36: memref<576xbf16, 1>, %arg37: memref<576xui8, 1>):
              %c64_i32 = arith.constant 64 : i32
              %c576_i32_159 = arith.constant 576 : i32
              %c0_i32_160 = arith.constant 0 : i32
              %cst = arith.constant 1.1920929E-7 : f32
              %cst_161 = arith.constant 3.276800e+04 : f32
              %cst_162 = arith.constant -3.276800e+04 : f32
              %cst_163 = arith.constant 1.000000e+00 : f32
              %cst_164 = arith.constant 0.000000e+00 : f32
              %c0_165 = arith.constant 0 : index
              %139 = ascvec.create_mask : !ascvec.mask
              ascvec.mem_bar vst_vld
              %140 = ascvec.loadalign_brc %arg35[%c0_165] : memref<1xf32, 1>, index -> <f32>
              %141 = ascvec.cmp_eq_scalar %140, %cst_164, %139 : <f32>, f32 -> !ascvec.mask
              %142 = ascvec.duplicate_scalar %cst_163, %139 : f32 -> <f32>
              %143 = ascvec.select %142, %140, %141 : <f32>
              scf.for %arg38 = %c0_i32_160 to %c576_i32_159 step %c64_i32  : i32 {
                %144 = arith.index_cast %arg38 : i32 to index
                %145 = ascvec.loadalign_unpack %arg36[%144] {mode = "b16_to_b32"} : memref<576xbf16, 1>, index -> <bf16>
                %146 = ascvec.cast %145, %139 {rounding = 6 : i32} : <bf16> -> <f32>
                %147 = ascvec.div %146, %143, %139 : <f32>
                %148 = ascvec.duplicate_scalar %cst_164, %139 : f32 -> <f32>
                %149 = ascvec.select %148, %147, %141 : <f32>
                %150 = ascvec.cmp_eq %149, %149, %139 : <f32>, <f32> -> !ascvec.mask
                %151 = ascvec.select %149, %148, %150 : <f32>
                %152 = ascvec.max_scalar %151, %cst_162, %139 : <f32>, f32
                %153 = ascvec.min_scalar %152, %cst_161, %139 : <f32>, f32
                %154 = ascvec.abs %153, %139 : <f32>
                %155 = ascvec.cmp_lt_scalar %154, %cst, %139 : <f32>, f32 -> !ascvec.mask
                %156 = ascvec.select %148, %153, %155 : <f32>
                %157 = ascvec.cast %156, %139 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
                %158 = ascvec.reinterpret %157 : <!ascdtype.hifloat8> -> <f8E4M3FN>
                ascvec.storealign_pack %arg37[%144], %158, %139 {mode = "b32_to_b8"} : memref<576xui8, 1>, index, <f8E4M3FN>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %123, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %103, 0 {cannir.buffer.generated_sync}
            %124 = cannir.extract_layout %arg27 : !cannir.tensor<<gm, ui8>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>
            %125 = cannir.extract_shape %124 : <<(-1, 576)>, <(576, 1)>> -> <(-1, 576)>
            %126 = cannir.extract_dim %125[0] : !cannir.shape<(-1, 576)>
            %127 = arith.minsi %arg33, %126 : i64
            %128 = arith.subi %126, %127 : i64
            %129 = arith.minsi %128, %c1_i64 : i64
            %130 = arith.cmpi sle, %arg33, %126 : i64
            %131 = arith.select %130, %129, %c0_i64 : i64
            %132 = arith.muli %arg33, %c4608_i64 : i64
            %133 = ascscalar.memref_address %alloc_20 : memref<1xui8>
            %134 = arith.divui %132, %c8_i64 : i64
            %135 = arith.addi %133, %134 : i64
            %136 = ascscalar.bind_gm_ptr %135 : memref<1xui8>
            %137 = ascscalar.bind_local_ptr %121 : memref<1xui8, 1>
            %138 = arith.cmpi ugt, %131, %c0_i64 : i64
            scf.if %138 {
              ascvec.get_buf mte3, %123, 0 {cannir.buffer.generated_sync}
              %139 = arith.trunci %131 : i64 to i32
              ascvec.copy_ub2gm %136[%c0], %137[%c0], %139, %c576_i32, %c576_i64, %c576_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
              ascvec.release_buf mte3, %123, 0 {cannir.buffer.generated_sync}
            }
            scf.yield %105 : i64
          }
        }
      }
    } else {
      %48 = cannir.extract_layout %arg26 : !cannir.tensor<<gm, bf16>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>
      %49 = cannir.extract_shape %48 : <<(-1, 576)>, <(576, 1)>> -> <(-1, 576)>
      %50 = cannir.extract_dim %49[0] : !cannir.shape<(-1, 576)>
      %51 = arith.cmpi sge, %50, %c16_i64 : i64
      scf.if %51 {
        %62 = arith.cmpi sgt, %5, %50 : i64
        %63 = arith.select %62, %50, %5 : i64
        %64 = arith.cmpi sgt, %63, %c128_i64 : i64
        %65 = arith.select %64, %c128_i64, %63 : i64
        %66 = cannir.extract_layout %arg16 : !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
        %67 = cannir.extract_shape %66 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
        %68 = cannir.extract_dim %67[0] : !cannir.shape<(-1, 1024)>
        %69 = arith.muli %68, %c1024_i64 : i64
        %70 = arith.remsi %69, %c8_i64 : i64
        %71 = arith.cmpi ne, %70, %c0_i64 : i64
        %72 = arith.cmpi slt, %70, %c0_i64 : i64
        %73 = arith.andi %71, %72 : i1
        %74 = arith.divsi %69, %c8_i64 : i64
        %75 = arith.extui %73 : i1 to i64
        %76 = arith.subi %74, %75 : i64
        %77 = arith.cmpi slt, %19, %65 : i64
        scf.if %77 {
          ascvec.vf(%alloc_138 : memref<128xf32, 1>) {
          ^bb0(%arg33: memref<128xf32, 1>):
            %c64 = arith.constant 64 : index
            %c0_159 = arith.constant 0 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c0_160 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg34 = %c0_160 to %c1 step %c1 {
              %92 = ascvec.create_mask : !ascvec.mask
              %93 = ascvec.duplicate_scalar %cst, %92 : f32 -> <f32>
              ascvec.storealign %arg33[%c0_159], %93, %92 : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg33[%c64], %93, %92 : memref<128xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          %79 = scf.for %arg33 = %19 to %50 step %65 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
            %92 = arith.cmpi eq, %arg34, %c0_i64 : i64
            %93 = ascscalar.memref_address %alloc_136 : memref<576xbf16, 1>
            %94 = ascscalar.memref_address %alloc_137 : memref<576xbf16, 1>
            %95 = arith.select %92, %93, %94 : i64
            %96 = arith.select %92, %c14_i64, %c15_i64 : i64
            %97 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
            %98 = arith.remui %97, %c2_i64 : i64
            %99 = arith.minsi %arg33, %50 : i64
            %100 = arith.subi %50, %99 : i64
            %101 = arith.minsi %100, %c1_i64 : i64
            %102 = arith.cmpi sle, %arg33, %50 : i64
            %103 = arith.select %102, %101, %c0_i64 : i64
            %104 = arith.muli %arg33, %c1152_i64 : i64
            %105 = arith.muli %103, %c1152_i64 : i64
            %106 = arith.cmpi eq, %105, %c0_i64 : i64
            %107 = arith.cmpi ne, %105, %c0_i64 : i64
            %108 = arith.extui %107 : i1 to i64
            %109 = arith.select %106, %c2_i64, %105 : i64
            %110 = ascscalar.bind_local_ptr %95 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %96, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%110, %alloc_19, %c0_i32, %c0_i32, %c0_i64, %104, %108, %109, %c0_i64, %109, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %96, 0 {cannir.buffer.generated_sync}
            %111 = ascscalar.bind_local_ptr %95 : memref<576xbf16, 1>
            ascvec.get_buf v, %96, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_138, %111 : memref<128xf32, 1>, memref<576xbf16, 1>) {
            ^bb0(%arg35: memref<128xf32, 1>, %arg36: memref<576xbf16, 1>):
              %c64 = arith.constant 64 : index
              %c64_i32 = arith.constant 64 : i32
              %c576_i32_159 = arith.constant 576 : i32
              %c0_i32_160 = arith.constant 0 : i32
              %cst = arith.constant 0.000000e+00 : f32
              %cst_161 = arith.constant 1.000000e+00 : f32
              %c0_162 = arith.constant 0 : index
              %112 = ascvec.create_mask : !ascvec.mask
              ascvec.mem_bar vst_vld
              %113 = ascvec.loadalign %arg35[%c0_162] : memref<128xf32, 1>, index -> <f32>
              %114 = ascvec.loadalign %arg35[%c64] : memref<128xf32, 1>, index -> <f32>
              %115 = ascvec.duplicate_scalar %cst_161, %112 : f32 -> <f32>
              %116 = ascvec.duplicate_scalar %cst, %112 : f32 -> <f32>
              %117:2 = scf.for %arg37 = %c0_i32_160 to %c576_i32_159 step %c64_i32 iter_args(%arg38 = %113, %arg39 = %114) -> (!ascvec.vreg<f32>, !ascvec.vreg<f32>)  : i32 {
                %118 = arith.index_cast %arg37 : i32 to index
                %119 = ascvec.loadalign_unpack %arg36[%118] {mode = "b16_to_b32"} : memref<576xbf16, 1>, index -> <bf16>
                %120 = ascvec.cast %119, %112 {rounding = 6 : i32} : <bf16> -> <f32>
                %121 = ascvec.abs %120, %112 : <f32>
                %122 = ascvec.max %arg38, %121, %112 : <f32>
                %123 = ascvec.cmp_ne %120, %120, %112 : <f32>, <f32> -> !ascvec.mask
                %124 = ascvec.select %115, %116, %123 : <f32>
                %125 = ascvec.max %arg39, %124, %112 : <f32>
                scf.yield %122, %125 : !ascvec.vreg<f32>, !ascvec.vreg<f32>
              }
              %c0_163 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg37 = %c0_163 to %c1 step %c1 {
                ascvec.storealign %arg35[%c0_162], %117#0, %112 : memref<128xf32, 1>, index, <f32>
                ascvec.storealign %arg35[%c64], %117#1, %112 : memref<128xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %96, 0 {cannir.buffer.generated_sync}
            scf.yield %98 : i64
          }
          ascvec.get_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_139, %alloc_138 : memref<8xf32, 1>, memref<128xf32, 1>) {
          ^bb0(%arg33: memref<8xf32, 1>, %arg34: memref<128xf32, 1>):
            %c1 = arith.constant 1 : index
            %c64 = arith.constant 64 : index
            %c0_159 = arith.constant 0 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c8_i32 = arith.constant 8 : i32
            %92 = ascvec.create_mask : !ascvec.mask
            %mask, %remaining_out = ascvec.update_mask %c8_i32 : !ascvec.mask
            %93 = ascvec.duplicate_scalar %cst, %mask : f32 -> <f32>
            ascvec.storealign %arg33[%c0_159], %93, %mask : memref<8xf32, 1>, index, <f32>
            ascvec.mem_bar vst_vld
            %c0_160 = arith.constant 0 : index
            %c1_161 = arith.constant 1 : index
            scf.for %arg35 = %c0_160 to %c1_161 step %c1_161 {
              %94 = ascvec.loadalign %arg34[%c0_159] : memref<128xf32, 1>, index -> <f32>
              %95 = ascvec.reduce_max %94, %92 : <f32>
              %96 = ascvec.duplicate %95, %92 : <f32> -> <f32>
              %97 = ascvec.loadalign %arg34[%c64] : memref<128xf32, 1>, index -> <f32>
              %98 = ascvec.reduce_max %97, %92 : <f32>
              %99 = ascvec.duplicate %98, %92 : <f32> -> <f32>
              ascvec.storealign_1st %arg33[%c0_159], %96 : memref<8xf32, 1>, index, <f32>
              ascvec.storealign_1st %arg33[%c1], %99 : memref<8xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c16_i64, 0 {cannir.buffer.generated_sync}
          %80 = arith.minsi %19, %76 : i64
          %81 = arith.subi %76, %80 : i64
          %82 = arith.minsi %81, %c1_i64 : i64
          %83 = arith.cmpi sle, %19, %76 : i64
          %84 = arith.select %83, %82, %c0_i64 : i64
          %85 = arith.muli %19, %c256_i64 : i64
          %86 = ascscalar.memref_address %alloc_10 : memref<1xf32>
          %87 = arith.divui %85, %c8_i64 : i64
          %88 = arith.addi %86, %87 : i64
          %89 = ascscalar.bind_gm_ptr %88 : memref<1xf32>
          %90 = ascscalar.bitcast_memref %alloc_139 byte_offset = %c0_i64 : memref<8xf32, 1> -> memref<?xf32, 1>
          %91 = arith.cmpi ugt, %84, %c0_i64 : i64
          scf.if %91 {
            ascvec.get_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
            %92 = arith.trunci %84 : i64 to i32
            ascvec.copy_ub2gm %89[%c0], %90[%c0], %92, %c32_i32, %c32_i64, %c32_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
            ascvec.release_buf mte3, %c16_i64, 0 {cannir.buffer.generated_sync}
          }
        }
        ascvec.sync_all
        ascvec.sync_block_arrive mte3, %c0_i64
        ascvec.sync_block_wait s, %c2_i64
        %78 = arith.cmpi eq, %19, %c0_i64 : i64
        scf.if %78 {
          %79 = arith.maxsi %65, %c1_i64 : i64
          %80 = arith.divsi %76, %79 : i64
          %81 = arith.minsi %80, %c0_i64 : i64
          %82 = arith.muli %81, %65 : i64
          %83 = arith.subi %76, %82 : i64
          %84 = arith.minsi %65, %83 : i64
          %85 = arith.cmpi sge, %80, %c0_i64 : i64
          %86 = arith.select %85, %84, %c0_i64 : i64
          %87 = arith.muli %86, %c32_i64 : i64
          %88 = arith.cmpi eq, %87, %c0_i64 : i64
          %89 = arith.cmpi ne, %87, %c0_i64 : i64
          %90 = arith.extui %89 : i1 to i64
          %91 = arith.select %88, %c4_i64, %87 : i64
          ascvec.get_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%alloc_141, %alloc_10, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %90, %91, %c0_i64, %91, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 32 : i64, ub_capacity = 4096 : i64}> : (memref<1024xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %c1_i64, 0 {cannir.buffer.generated_sync}
          %92 = arith.trunci %65 : i64 to i32
          ascvec.get_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_141, %92, %alloc_140 : memref<1024xf32, 1>, i32, memref<1xf32, 1>) {
          ^bb0(%arg33: memref<1024xf32, 1>, %arg34: i32, %arg35: memref<1xf32, 1>):
            %c2143289344_i32 = arith.constant 2143289344 : i32
            %cst = arith.constant 3.276800e+04 : f32
            %c0_i32_159 = arith.constant 0 : i32
            %c1_i32_160 = arith.constant 1 : i32
            %c0_161 = arith.constant 0 : index
            %c8_i32 = arith.constant 8 : i32
            %cst_162 = arith.constant 0.000000e+00 : f32
            %94 = ascvec.create_mask : !ascvec.mask
            %95 = ascvec.duplicate_scalar %cst_162, %94 : f32 -> <f32>
            %96:2 = scf.for %arg36 = %c0_i32_159 to %arg34 step %c1_i32_160 iter_args(%arg37 = %95, %arg38 = %95) -> (!ascvec.vreg<f32>, !ascvec.vreg<f32>)  : i32 {
              %97 = arith.muli %arg36, %c8_i32 : i32
              %98 = arith.index_cast %97 : i32 to index
              %99 = ascvec.loadalign_brc %arg33[%98] : memref<1024xf32, 1>, index -> <f32>
              %100 = ascvec.max %arg37, %99, %94 : <f32>
              %101 = arith.addi %97, %c1_i32_160 : i32
              %102 = arith.index_cast %101 : i32 to index
              %103 = ascvec.loadalign_brc %arg33[%102] : memref<1024xf32, 1>, index -> <f32>
              %104 = ascvec.max %arg38, %103, %94 : <f32>
              scf.yield %100, %104 : !ascvec.vreg<f32>, !ascvec.vreg<f32>
            }
            %c0_163 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg36 = %c0_163 to %c1 step %c1 {
              %97 = ascvec.duplicate_scalar %cst, %94 : f32 -> <f32>
              %98 = ascvec.div %96#0, %97, %94 {precision_mode = true} : <f32>
              %99 = ascvec.duplicate_scalar %c2143289344_i32, %94 : i32 -> <ui32>
              %100 = ascvec.reinterpret %99 : <ui32> -> <f32>
              %101 = ascvec.cmp_gt_scalar %96#1, %cst_162, %94 : <f32>, f32 -> !ascvec.mask
              %102 = ascvec.select %100, %98, %101 : <f32>
              ascvec.storealign_1st %arg35[%c0_161], %102 : memref<1xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %c1_i64, 0 {cannir.buffer.generated_sync}
          %93 = ascscalar.bitcast_memref %alloc_140 byte_offset = %c0_i64 : memref<1xf32, 1> -> memref<?xf32, 1>
          ascvec.get_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %alloc_21[%c0], %93[%c0], %c1_i32, %c4_i32, %c0_i64, %c4_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
          ascvec.release_buf mte3, %c0_i64, 0 {cannir.buffer.generated_sync}
        }
      } else {
        %62 = arith.cmpi eq, %0, %c0_i64 : i64
        %63 = arith.cmpi eq, %18, %c0_i64 : i64
        %64 = arith.select %62, %63, %false : i1
        scf.if %64 {
          ascvec.vf(%alloc_144 : memref<128xf32, 1>) {
          ^bb0(%arg33: memref<128xf32, 1>):
            %c64 = arith.constant 64 : index
            %c0_159 = arith.constant 0 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c0_160 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg34 = %c0_160 to %c1 step %c1 {
              %67 = ascvec.create_mask : !ascvec.mask
              %68 = ascvec.duplicate_scalar %cst, %67 : f32 -> <f32>
              ascvec.storealign %arg33[%c0_159], %68, %67 : memref<128xf32, 1>, index, <f32>
              ascvec.storealign %arg33[%c64], %68, %67 : memref<128xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          %65 = scf.for %arg33 = %c0_i64 to %50 step %c1_i64 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
            %67 = arith.cmpi eq, %arg34, %c0_i64 : i64
            %68 = ascscalar.memref_address %alloc_142 : memref<576xbf16, 1>
            %69 = ascscalar.memref_address %alloc_143 : memref<576xbf16, 1>
            %70 = arith.select %67, %68, %69 : i64
            %71 = arith.select %67, %c2_i64, %c3_i64 : i64
            %72 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
            %73 = arith.remui %72, %c2_i64 : i64
            %74 = arith.minsi %arg33, %50 : i64
            %75 = arith.subi %50, %74 : i64
            %76 = arith.minsi %75, %c1_i64 : i64
            %77 = arith.cmpi sle, %arg33, %50 : i64
            %78 = arith.select %77, %76, %c0_i64 : i64
            %79 = arith.muli %arg33, %c1152_i64 : i64
            %80 = arith.muli %78, %c1152_i64 : i64
            %81 = arith.cmpi eq, %80, %c0_i64 : i64
            %82 = arith.cmpi ne, %80, %c0_i64 : i64
            %83 = arith.extui %82 : i1 to i64
            %84 = arith.select %81, %c2_i64, %80 : i64
            %85 = ascscalar.bind_local_ptr %70 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %71, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%85, %alloc_19, %c0_i32, %c0_i32, %c0_i64, %79, %83, %84, %c0_i64, %84, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %71, 0 {cannir.buffer.generated_sync}
            %86 = ascscalar.bind_local_ptr %70 : memref<576xbf16, 1>
            ascvec.get_buf v, %71, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_144, %86 : memref<128xf32, 1>, memref<576xbf16, 1>) {
            ^bb0(%arg35: memref<128xf32, 1>, %arg36: memref<576xbf16, 1>):
              %c64 = arith.constant 64 : index
              %c64_i32 = arith.constant 64 : i32
              %c576_i32_159 = arith.constant 576 : i32
              %c0_i32_160 = arith.constant 0 : i32
              %cst = arith.constant 0.000000e+00 : f32
              %cst_161 = arith.constant 1.000000e+00 : f32
              %c0_162 = arith.constant 0 : index
              %87 = ascvec.create_mask : !ascvec.mask
              ascvec.mem_bar vst_vld
              %88 = ascvec.loadalign %arg35[%c0_162] : memref<128xf32, 1>, index -> <f32>
              %89 = ascvec.loadalign %arg35[%c64] : memref<128xf32, 1>, index -> <f32>
              %90 = ascvec.duplicate_scalar %cst_161, %87 : f32 -> <f32>
              %91 = ascvec.duplicate_scalar %cst, %87 : f32 -> <f32>
              %92:2 = scf.for %arg37 = %c0_i32_160 to %c576_i32_159 step %c64_i32 iter_args(%arg38 = %88, %arg39 = %89) -> (!ascvec.vreg<f32>, !ascvec.vreg<f32>)  : i32 {
                %93 = arith.index_cast %arg37 : i32 to index
                %94 = ascvec.loadalign_unpack %arg36[%93] {mode = "b16_to_b32"} : memref<576xbf16, 1>, index -> <bf16>
                %95 = ascvec.cast %94, %87 {rounding = 6 : i32} : <bf16> -> <f32>
                %96 = ascvec.abs %95, %87 : <f32>
                %97 = ascvec.max %arg38, %96, %87 : <f32>
                %98 = ascvec.cmp_ne %95, %95, %87 : <f32>, <f32> -> !ascvec.mask
                %99 = ascvec.select %90, %91, %98 : <f32>
                %100 = ascvec.max %arg39, %99, %87 : <f32>
                scf.yield %97, %100 : !ascvec.vreg<f32>, !ascvec.vreg<f32>
              }
              %c0_163 = arith.constant 0 : index
              %c1 = arith.constant 1 : index
              scf.for %arg37 = %c0_163 to %c1 step %c1 {
                ascvec.storealign %arg35[%c0_162], %92#0, %87 : memref<128xf32, 1>, index, <f32>
                ascvec.storealign %arg35[%c64], %92#1, %87 : memref<128xf32, 1>, index, <f32>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %71, 0 {cannir.buffer.generated_sync}
            scf.yield %73 : i64
          }
          ascvec.get_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_144, %alloc_145 : memref<128xf32, 1>, memref<1xf32, 1>) {
          ^bb0(%arg33: memref<128xf32, 1>, %arg34: memref<1xf32, 1>):
            %c64 = arith.constant 64 : index
            %cst = arith.constant 0.000000e+00 : f32
            %c2143289344_i32 = arith.constant 2143289344 : i32
            %cst_159 = arith.constant 3.276800e+04 : f32
            %c0_160 = arith.constant 0 : index
            %67 = ascvec.create_mask : !ascvec.mask
            ascvec.mem_bar vst_vld
            %c0_161 = arith.constant 0 : index
            %c1 = arith.constant 1 : index
            scf.for %arg35 = %c0_161 to %c1 step %c1 {
              %68 = ascvec.loadalign %arg33[%c0_160] : memref<128xf32, 1>, index -> <f32>
              %69 = ascvec.reduce_max %68, %67 : <f32>
              %70 = ascvec.duplicate %69, %67 : <f32> -> <f32>
              %71 = ascvec.loadalign %arg33[%c64] : memref<128xf32, 1>, index -> <f32>
              %72 = ascvec.reduce_max %71, %67 : <f32>
              %73 = ascvec.duplicate %72, %67 : <f32> -> <f32>
              %74 = ascvec.duplicate_scalar %cst_159, %67 : f32 -> <f32>
              %75 = ascvec.div %70, %74, %67 {precision_mode = true} : <f32>
              %76 = ascvec.duplicate_scalar %c2143289344_i32, %67 : i32 -> <ui32>
              %77 = ascvec.reinterpret %76 : <ui32> -> <f32>
              %78 = ascvec.cmp_gt_scalar %73, %cst, %67 : <f32>, f32 -> !ascvec.mask
              %79 = ascvec.select %77, %75, %78 : <f32>
              ascvec.storealign_1st %arg34[%c0_160], %79 : memref<1xf32, 1>, index, <f32>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %c4_i64, 0 {cannir.buffer.generated_sync}
          %66 = ascscalar.bitcast_memref %alloc_145 byte_offset = %c0_i64 : memref<1xf32, 1> -> memref<?xf32, 1>
          ascvec.get_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
          ascvec.copy_ub2gm %alloc_21[%c0], %66[%c0], %c1_i32, %c4_i32, %c0_i64, %c4_i32 l2_cache_ctl %c0_i32 : memref<1xf32>, memref<?xf32, 1>
          ascvec.release_buf mte3, %c4_i64, 0 {cannir.buffer.generated_sync}
        }
      }
      ascvec.sync_all
      ascvec.sync_block_arrive mte3, %c0_i64
      ascvec.sync_block_wait s, %c2_i64
      ascvec.get_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
      "ascvec.copy_gm2ub_plan"(%alloc_146, %alloc_21, %c0_i32, %c0_i32, %c0_i64, %c0_i64, %c1_i64, %c4_i64, %c0_i64, %c4_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = 4 : i64, pad_storage_bits = false, storage_dtype = f32, ub_base = 0 : i64, ub_capacity = 4 : i64}> : (memref<1xf32, 1>, memref<1xf32>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
      ascvec.release_buf mte2, %c0_i64, 0 {cannir.buffer.generated_sync}
      %52 = arith.remsi %50, %c16_i64 : i64
      %53 = arith.cmpi ne, %52, %c0_i64 : i64
      %54 = arith.cmpi slt, %52, %c0_i64 : i64
      %55 = arith.andi %53, %54 : i1
      %56 = arith.select %55, %c16_i64, %c0_i64 : i64
      %57 = arith.addi %52, %56 overflow<nsw> : i64
      %58 = arith.cmpi eq, %57, %c0_i64 : i64
      %59 = arith.muli %1, %c32_i64 : i64
      %60 = arith.cmpi sge, %50, %59 : i64
      %61 = arith.select %58, %60, %false : i1
      scf.if %61 {
        %62 = arith.divsi %50, %c16_i64 : i64
        %63 = arith.extui %55 : i1 to i64
        %64 = arith.subi %62, %63 : i64
        %65 = scf.for %arg33 = %19 to %64 step %5 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
          %66 = arith.cmpi eq, %arg34, %c0_i64 : i64
          %67 = ascscalar.memref_address %alloc_147 : memref<9216xbf16, 1>
          %68 = ascscalar.memref_address %alloc_148 : memref<9216xbf16, 1>
          %69 = arith.select %66, %67, %68 : i64
          %70 = arith.select %66, %c1_i64, %c2_i64 : i64
          %71 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
          %72 = arith.remui %71, %c2_i64 : i64
          %73 = arith.minsi %arg33, %64 : i64
          %74 = arith.subi %64, %73 : i64
          %75 = arith.minsi %74, %c1_i64 : i64
          %76 = arith.cmpi sle, %arg33, %64 : i64
          %77 = arith.select %76, %75, %c0_i64 : i64
          %78 = arith.muli %arg33, %c18432_i64 : i64
          %79 = arith.muli %77, %c18432_i64 : i64
          %80 = arith.cmpi eq, %79, %c0_i64 : i64
          %81 = arith.cmpi ne, %79, %c0_i64 : i64
          %82 = arith.extui %81 : i1 to i64
          %83 = arith.select %80, %c2_i64, %79 : i64
          %84 = ascscalar.bind_local_ptr %69 : memref<1xbf16, 1>
          ascvec.get_buf mte2, %70, 0 {cannir.buffer.generated_sync}
          "ascvec.copy_gm2ub_plan"(%84, %alloc_19, %c0_i32, %c0_i32, %c0_i64, %78, %82, %83, %c0_i64, %83, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
          ascvec.release_buf mte2, %70, 0 {cannir.buffer.generated_sync}
          %85 = ascscalar.bind_local_ptr %69 : memref<9216xbf16, 1>
          %86 = ascscalar.memref_address %alloc_149 : memref<9216xui8, 1>
          %87 = ascscalar.memref_address %alloc_150 : memref<9216xui8, 1>
          %88 = arith.select %66, %86, %87 : i64
          %89 = ascscalar.bind_local_ptr %88 : memref<9216xui8, 1>
          %90 = arith.select %66, %c3_i64, %c4_i64 : i64
          ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %70, 0 {cannir.buffer.generated_sync}
          ascvec.get_buf v, %90, 0 {cannir.buffer.generated_sync}
          ascvec.vf(%alloc_146, %85, %89 : memref<1xf32, 1>, memref<9216xbf16, 1>, memref<9216xui8, 1>) {
          ^bb0(%arg35: memref<1xf32, 1>, %arg36: memref<9216xbf16, 1>, %arg37: memref<9216xui8, 1>):
            %c64_i32 = arith.constant 64 : i32
            %c9216_i32_159 = arith.constant 9216 : i32
            %c0_i32_160 = arith.constant 0 : i32
            %cst = arith.constant 1.1920929E-7 : f32
            %cst_161 = arith.constant 3.276800e+04 : f32
            %cst_162 = arith.constant -3.276800e+04 : f32
            %cst_163 = arith.constant 1.000000e+00 : f32
            %cst_164 = arith.constant 0.000000e+00 : f32
            %c0_165 = arith.constant 0 : index
            %98 = ascvec.create_mask : !ascvec.mask
            ascvec.mem_bar vst_vld
            %99 = ascvec.loadalign_brc %arg35[%c0_165] : memref<1xf32, 1>, index -> <f32>
            %100 = ascvec.cmp_eq_scalar %99, %cst_164, %98 : <f32>, f32 -> !ascvec.mask
            %101 = ascvec.duplicate_scalar %cst_163, %98 : f32 -> <f32>
            %102 = ascvec.select %101, %99, %100 : <f32>
            scf.for %arg38 = %c0_i32_160 to %c9216_i32_159 step %c64_i32  : i32 {
              %103 = arith.index_cast %arg38 : i32 to index
              %104 = ascvec.loadalign_unpack %arg36[%103] {mode = "b16_to_b32"} : memref<9216xbf16, 1>, index -> <bf16>
              %105 = ascvec.cast %104, %98 {rounding = 6 : i32} : <bf16> -> <f32>
              %106 = ascvec.div %105, %102, %98 : <f32>
              %107 = ascvec.duplicate_scalar %cst_164, %98 : f32 -> <f32>
              %108 = ascvec.select %107, %106, %100 : <f32>
              %109 = ascvec.cmp_eq %108, %108, %98 : <f32>, <f32> -> !ascvec.mask
              %110 = ascvec.select %108, %107, %109 : <f32>
              %111 = ascvec.max_scalar %110, %cst_162, %98 : <f32>, f32
              %112 = ascvec.min_scalar %111, %cst_161, %98 : <f32>, f32
              %113 = ascvec.abs %112, %98 : <f32>
              %114 = ascvec.cmp_lt_scalar %113, %cst, %98 : <f32>, f32 -> !ascvec.mask
              %115 = ascvec.select %107, %112, %114 : <f32>
              %116 = ascvec.cast %115, %98 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
              %117 = ascvec.reinterpret %116 : <!ascdtype.hifloat8> -> <f8E4M3FN>
              ascvec.storealign_pack %arg37[%103], %117, %98 {mode = "b32_to_b8"} : memref<9216xui8, 1>, index, <f8E4M3FN>
            }
          } {cannir.vf_mode = "simd"}
          ascvec.release_buf v, %90, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %70, 0 {cannir.buffer.generated_sync}
          ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
          %91 = arith.muli %arg33, %c73728_i64 : i64
          %92 = ascscalar.memref_address %alloc_20 : memref<1xui8>
          %93 = arith.divui %91, %c8_i64 : i64
          %94 = arith.addi %92, %93 : i64
          %95 = ascscalar.bind_gm_ptr %94 : memref<1xui8>
          %96 = ascscalar.bind_local_ptr %88 : memref<1xui8, 1>
          %97 = arith.cmpi ugt, %77, %c0_i64 : i64
          scf.if %97 {
            ascvec.get_buf mte3, %90, 0 {cannir.buffer.generated_sync}
            %98 = arith.trunci %77 : i64 to i32
            ascvec.copy_ub2gm %95[%c0], %96[%c0], %98, %c9216_i32, %c9216_i64, %c9216_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
            ascvec.release_buf mte3, %90, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %72 : i64
        }
      } else {
        %62 = arith.remsi %50, %c8_i64 : i64
        %63 = arith.cmpi ne, %62, %c0_i64 : i64
        %64 = arith.cmpi slt, %62, %c0_i64 : i64
        %65 = arith.andi %63, %64 : i1
        %66 = arith.select %65, %c8_i64, %c0_i64 : i64
        %67 = arith.addi %62, %66 overflow<nsw> : i64
        %68 = arith.cmpi eq, %67, %c0_i64 : i64
        %69 = arith.cmpi sge, %50, %8 : i64
        %70 = arith.select %68, %69, %false : i1
        scf.if %70 {
          %71 = arith.divsi %50, %c8_i64 : i64
          %72 = arith.extui %65 : i1 to i64
          %73 = arith.subi %71, %72 : i64
          %74 = scf.for %arg33 = %19 to %73 step %5 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
            %75 = arith.cmpi eq, %arg34, %c0_i64 : i64
            %76 = ascscalar.memref_address %alloc_151 : memref<4608xbf16, 1>
            %77 = ascscalar.memref_address %alloc_152 : memref<4608xbf16, 1>
            %78 = arith.select %75, %76, %77 : i64
            %79 = arith.select %75, %c5_i64, %c6_i64 : i64
            %80 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
            %81 = arith.remui %80, %c2_i64 : i64
            %82 = arith.minsi %arg33, %73 : i64
            %83 = arith.subi %73, %82 : i64
            %84 = arith.minsi %83, %c1_i64 : i64
            %85 = arith.cmpi sle, %arg33, %73 : i64
            %86 = arith.select %85, %84, %c0_i64 : i64
            %87 = arith.muli %arg33, %c9216_i64 : i64
            %88 = arith.muli %86, %c9216_i64 : i64
            %89 = arith.cmpi eq, %88, %c0_i64 : i64
            %90 = arith.cmpi ne, %88, %c0_i64 : i64
            %91 = arith.extui %90 : i1 to i64
            %92 = arith.select %89, %c2_i64, %88 : i64
            %93 = ascscalar.bind_local_ptr %78 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %79, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%93, %alloc_19, %c0_i32, %c0_i32, %c0_i64, %87, %91, %92, %c0_i64, %92, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %79, 0 {cannir.buffer.generated_sync}
            %94 = ascscalar.bind_local_ptr %78 : memref<4608xbf16, 1>
            %95 = ascscalar.memref_address %alloc_153 : memref<4608xui8, 1>
            %96 = ascscalar.memref_address %alloc_154 : memref<4608xui8, 1>
            %97 = arith.select %75, %95, %96 : i64
            %98 = ascscalar.bind_local_ptr %97 : memref<4608xui8, 1>
            %99 = arith.select %75, %c7_i64, %c8_i64 : i64
            ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %79, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %99, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_146, %94, %98 : memref<1xf32, 1>, memref<4608xbf16, 1>, memref<4608xui8, 1>) {
            ^bb0(%arg35: memref<1xf32, 1>, %arg36: memref<4608xbf16, 1>, %arg37: memref<4608xui8, 1>):
              %c64_i32 = arith.constant 64 : i32
              %c4608_i32_159 = arith.constant 4608 : i32
              %c0_i32_160 = arith.constant 0 : i32
              %cst = arith.constant 1.1920929E-7 : f32
              %cst_161 = arith.constant 3.276800e+04 : f32
              %cst_162 = arith.constant -3.276800e+04 : f32
              %cst_163 = arith.constant 1.000000e+00 : f32
              %cst_164 = arith.constant 0.000000e+00 : f32
              %c0_165 = arith.constant 0 : index
              %107 = ascvec.create_mask : !ascvec.mask
              ascvec.mem_bar vst_vld
              %108 = ascvec.loadalign_brc %arg35[%c0_165] : memref<1xf32, 1>, index -> <f32>
              %109 = ascvec.cmp_eq_scalar %108, %cst_164, %107 : <f32>, f32 -> !ascvec.mask
              %110 = ascvec.duplicate_scalar %cst_163, %107 : f32 -> <f32>
              %111 = ascvec.select %110, %108, %109 : <f32>
              scf.for %arg38 = %c0_i32_160 to %c4608_i32_159 step %c64_i32  : i32 {
                %112 = arith.index_cast %arg38 : i32 to index
                %113 = ascvec.loadalign_unpack %arg36[%112] {mode = "b16_to_b32"} : memref<4608xbf16, 1>, index -> <bf16>
                %114 = ascvec.cast %113, %107 {rounding = 6 : i32} : <bf16> -> <f32>
                %115 = ascvec.div %114, %111, %107 : <f32>
                %116 = ascvec.duplicate_scalar %cst_164, %107 : f32 -> <f32>
                %117 = ascvec.select %116, %115, %109 : <f32>
                %118 = ascvec.cmp_eq %117, %117, %107 : <f32>, <f32> -> !ascvec.mask
                %119 = ascvec.select %117, %116, %118 : <f32>
                %120 = ascvec.max_scalar %119, %cst_162, %107 : <f32>, f32
                %121 = ascvec.min_scalar %120, %cst_161, %107 : <f32>, f32
                %122 = ascvec.abs %121, %107 : <f32>
                %123 = ascvec.cmp_lt_scalar %122, %cst, %107 : <f32>, f32 -> !ascvec.mask
                %124 = ascvec.select %116, %121, %123 : <f32>
                %125 = ascvec.cast %124, %107 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
                %126 = ascvec.reinterpret %125 : <!ascdtype.hifloat8> -> <f8E4M3FN>
                ascvec.storealign_pack %arg37[%112], %126, %107 {mode = "b32_to_b8"} : memref<4608xui8, 1>, index, <f8E4M3FN>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %99, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %79, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
            %100 = arith.muli %arg33, %c36864_i64 : i64
            %101 = ascscalar.memref_address %alloc_20 : memref<1xui8>
            %102 = arith.divui %100, %c8_i64 : i64
            %103 = arith.addi %101, %102 : i64
            %104 = ascscalar.bind_gm_ptr %103 : memref<1xui8>
            %105 = ascscalar.bind_local_ptr %97 : memref<1xui8, 1>
            %106 = arith.cmpi ugt, %86, %c0_i64 : i64
            scf.if %106 {
              ascvec.get_buf mte3, %99, 0 {cannir.buffer.generated_sync}
              %107 = arith.trunci %86 : i64 to i32
              ascvec.copy_ub2gm %104[%c0], %105[%c0], %107, %c4608_i32, %c4608_i64, %c4608_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
              ascvec.release_buf mte3, %99, 0 {cannir.buffer.generated_sync}
            }
            scf.yield %81 : i64
          }
        } else {
          %71 = scf.for %arg33 = %19 to %50 step %5 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
            %72 = arith.cmpi eq, %arg34, %c0_i64 : i64
            %73 = ascscalar.memref_address %alloc_155 : memref<576xbf16, 1>
            %74 = ascscalar.memref_address %alloc_156 : memref<576xbf16, 1>
            %75 = arith.select %72, %73, %74 : i64
            %76 = arith.select %72, %c9_i64, %c10_i64 : i64
            %77 = arith.addi %arg34, %c1_i64 overflow<nsw> : i64
            %78 = arith.remui %77, %c2_i64 : i64
            %79 = arith.minsi %arg33, %50 : i64
            %80 = arith.subi %50, %79 : i64
            %81 = arith.minsi %80, %c1_i64 : i64
            %82 = arith.cmpi sle, %arg33, %50 : i64
            %83 = arith.select %82, %81, %c0_i64 : i64
            %84 = arith.muli %arg33, %c1152_i64 : i64
            %85 = arith.muli %83, %c1152_i64 : i64
            %86 = arith.cmpi eq, %85, %c0_i64 : i64
            %87 = arith.cmpi ne, %85, %c0_i64 : i64
            %88 = arith.extui %87 : i1 to i64
            %89 = arith.select %86, %c2_i64, %85 : i64
            %90 = ascscalar.bind_local_ptr %75 : memref<1xbf16, 1>
            ascvec.get_buf mte2, %76, 0 {cannir.buffer.generated_sync}
            "ascvec.copy_gm2ub_plan"(%90, %alloc_19, %c0_i32, %c0_i32, %c0_i64, %84, %88, %89, %c0_i64, %89, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64, %c1_i64, %c0_i64, %c0_i64) <{constant_pad = true, gm_capacity = -1 : i64, pad_storage_bits = false, storage_dtype = bf16, ub_base = -1 : i64, ub_capacity = 2 : i64}> : (memref<1xbf16, 1>, memref<1xbf16>, i32, i32, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> ()
            ascvec.release_buf mte2, %76, 0 {cannir.buffer.generated_sync}
            %91 = ascscalar.bind_local_ptr %75 : memref<576xbf16, 1>
            %92 = ascscalar.memref_address %alloc_157 : memref<576xui8, 1>
            %93 = ascscalar.memref_address %alloc_158 : memref<576xui8, 1>
            %94 = arith.select %72, %92, %93 : i64
            %95 = ascscalar.bind_local_ptr %94 : memref<576xui8, 1>
            %96 = arith.select %72, %c11_i64, %c12_i64 : i64
            ascvec.get_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %76, 0 {cannir.buffer.generated_sync}
            ascvec.get_buf v, %96, 0 {cannir.buffer.generated_sync}
            ascvec.vf(%alloc_146, %91, %95 : memref<1xf32, 1>, memref<576xbf16, 1>, memref<576xui8, 1>) {
            ^bb0(%arg35: memref<1xf32, 1>, %arg36: memref<576xbf16, 1>, %arg37: memref<576xui8, 1>):
              %c64_i32 = arith.constant 64 : i32
              %c576_i32_159 = arith.constant 576 : i32
              %c0_i32_160 = arith.constant 0 : i32
              %cst = arith.constant 1.1920929E-7 : f32
              %cst_161 = arith.constant 3.276800e+04 : f32
              %cst_162 = arith.constant -3.276800e+04 : f32
              %cst_163 = arith.constant 1.000000e+00 : f32
              %cst_164 = arith.constant 0.000000e+00 : f32
              %c0_165 = arith.constant 0 : index
              %112 = ascvec.create_mask : !ascvec.mask
              ascvec.mem_bar vst_vld
              %113 = ascvec.loadalign_brc %arg35[%c0_165] : memref<1xf32, 1>, index -> <f32>
              %114 = ascvec.cmp_eq_scalar %113, %cst_164, %112 : <f32>, f32 -> !ascvec.mask
              %115 = ascvec.duplicate_scalar %cst_163, %112 : f32 -> <f32>
              %116 = ascvec.select %115, %113, %114 : <f32>
              scf.for %arg38 = %c0_i32_160 to %c576_i32_159 step %c64_i32  : i32 {
                %117 = arith.index_cast %arg38 : i32 to index
                %118 = ascvec.loadalign_unpack %arg36[%117] {mode = "b16_to_b32"} : memref<576xbf16, 1>, index -> <bf16>
                %119 = ascvec.cast %118, %112 {rounding = 6 : i32} : <bf16> -> <f32>
                %120 = ascvec.div %119, %116, %112 : <f32>
                %121 = ascvec.duplicate_scalar %cst_164, %112 : f32 -> <f32>
                %122 = ascvec.select %121, %120, %114 : <f32>
                %123 = ascvec.cmp_eq %122, %122, %112 : <f32>, <f32> -> !ascvec.mask
                %124 = ascvec.select %122, %121, %123 : <f32>
                %125 = ascvec.max_scalar %124, %cst_162, %112 : <f32>, f32
                %126 = ascvec.min_scalar %125, %cst_161, %112 : <f32>, f32
                %127 = ascvec.abs %126, %112 : <f32>
                %128 = ascvec.cmp_lt_scalar %127, %cst, %112 : <f32>, f32 -> !ascvec.mask
                %129 = ascvec.select %121, %126, %128 : <f32>
                %130 = ascvec.cast %129, %112 {reg_layout = 0 : i32, rounding = 1 : i32} : <f32> -> <!ascdtype.hifloat8>
                %131 = ascvec.reinterpret %130 : <!ascdtype.hifloat8> -> <f8E4M3FN>
                ascvec.storealign_pack %arg37[%117], %131, %112 {mode = "b32_to_b8"} : memref<576xui8, 1>, index, <f8E4M3FN>
              }
            } {cannir.vf_mode = "simd"}
            ascvec.release_buf v, %96, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %76, 0 {cannir.buffer.generated_sync}
            ascvec.release_buf v, %c0_i64, 0 {cannir.buffer.generated_sync}
            %97 = cannir.extract_layout %arg27 : !cannir.tensor<<gm, ui8>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>
            %98 = cannir.extract_shape %97 : <<(-1, 576)>, <(576, 1)>> -> <(-1, 576)>
            %99 = cannir.extract_dim %98[0] : !cannir.shape<(-1, 576)>
            %100 = arith.minsi %arg33, %99 : i64
            %101 = arith.subi %99, %100 : i64
            %102 = arith.minsi %101, %c1_i64 : i64
            %103 = arith.cmpi sle, %arg33, %99 : i64
            %104 = arith.select %103, %102, %c0_i64 : i64
            %105 = arith.muli %arg33, %c4608_i64 : i64
            %106 = ascscalar.memref_address %alloc_20 : memref<1xui8>
            %107 = arith.divui %105, %c8_i64 : i64
            %108 = arith.addi %106, %107 : i64
            %109 = ascscalar.bind_gm_ptr %108 : memref<1xui8>
            %110 = ascscalar.bind_local_ptr %94 : memref<1xui8, 1>
            %111 = arith.cmpi ugt, %104, %c0_i64 : i64
            scf.if %111 {
              ascvec.get_buf mte3, %96, 0 {cannir.buffer.generated_sync}
              %112 = arith.trunci %104 : i64 to i32
              ascvec.copy_ub2gm %109[%c0], %110[%c0], %112, %c576_i32, %c576_i64, %c576_i32 l2_cache_ctl %c0_i32 : memref<1xui8>, memref<1xui8, 1>
              ascvec.release_buf mte3, %96, 0 {cannir.buffer.generated_sync}
            }
            scf.yield %78 : i64
          }
        }
      }
    }
    ascvec.sync_all
    ascvec.sync_block_arrive mte3, %c0_i64
    ascvec.sync_block_wait s, %c2_i64
    return
  }
  func.func @kernel__QuantMlaPrologKernel___call___0_aic(%arg0: !cannir.tensor<<gm, ui8>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg1: !cannir.tensor<<gm, ui8>, <<(1024, 1024)>, <(1024, 1)>>, <<(1024, 1024)>, <(1024, 1)>>, identity>, %arg2: !cannir.tensor<<gm, ui8>, <<(384, 1024)>, <(1024, 1)>>, <<(384, 1024)>, <(1024, 1)>>, identity>, %arg3: !cannir.tensor<<gm, ui8>, <<(576, 1024)>, <(1024, 1)>>, <<(576, 1024)>, <(1024, 1)>>, identity>, %arg4: !cannir.tensor<<gm, bf16>, <<(2, 128, 512)>, <(65536, 512, 1)>>, <<(2, 128, 512)>, <(65536, 512, 1)>>, identity>, %arg5: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg6: !cannir.tensor<<gm, f32>, <<(1, 1024)>, <(1024, 1)>>, <<(1, 1024)>, <(1024, 1)>>, identity>, %arg7: !cannir.tensor<<gm, f32>, <<(1, 384)>, <(384, 1)>>, <<(1, 384)>, <(384, 1)>>, identity>, %arg8: !cannir.tensor<<gm, f32>, <<(1, 576)>, <(576, 1)>>, <<(1, 576)>, <(576, 1)>>, identity>, %arg9: !cannir.tensor<<gm, f32>, <<(1, 1024)>, <(1024, 1)>>, <<(1, 1024)>, <(1024, 1)>>, identity>, %arg10: !cannir.tensor<<gm, f32>, <<(1, 512)>, <(512, 1)>>, <<(1, 512)>, <(512, 1)>>, identity>, %arg11: !cannir.tensor<<gm, ui8>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, <<(-1, -1, 1, 576)>, <(-1, 576, 576, 1)>>, identity>, %arg12: !cannir.tensor<<gm, i64>, <<(-1)>, <(1)>>, <<(-1)>, <(1)>>, identity>, %arg13: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg14: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg15: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg16: !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg17: !cannir.tensor<<gm, f32>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>, %arg18: !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg19: !cannir.tensor<<gm, f32>, <<(-1, 512)>, <(512, 1)>>, <<(-1, 512)>, <(512, 1)>>, identity>, %arg20: !cannir.tensor<<gm, ui8>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>, %arg21: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg22: !cannir.tensor<<gm, f32>, <<(-1, 384)>, <(384, 1)>>, <<(-1, 384)>, <(384, 1)>>, identity>, %arg23: !cannir.tensor<<gm, bf16>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>, %arg24: !cannir.tensor<<gm, bf16>, <<(-1, 64)>, <(64, 1)>>, <<(-1, 64)>, <(64, 1)>>, identity>, %arg25: !cannir.tensor<<gm, bf16>, <<(-1, 512)>, <(512, 1)>>, <<(-1, 512)>, <(512, 1)>>, identity>, %arg26: !cannir.tensor<<gm, bf16>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>, %arg27: !cannir.tensor<<gm, ui8>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>, %arg28: !cannir.tensor<<gm, f32>, <<(1)>, <(1)>>, <<(1)>, <(1)>>, identity>, %arg29: f32, %arg30: f32, %arg31: f32, %arg32: !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>) attributes {cannir.buffer.auto_sync, cannir.buffer.sync_analyzed, cannir.buffer.sync_verified, cannir.buffer.tensor_state, cannir.core = "aic", cannir.cube_state_mode = 0 : i32, cannir.cube_state_planned, cannir.device, cannir.explicit_ring_buffer_cursors, cannir.kernel_group = "kernel__QuantMlaPrologKernel___call___0", cannir.resource.ids_prepared} {
    %c1152_i32 = arith.constant 1152 : i32
    %c384_i32 = arith.constant 384 : i32
    %c16_i32 = arith.constant 16 : i32
    %c256_i32 = arith.constant 256 : i32
    %c256_i64 = arith.constant 256 : i64
    %c0_i32 = arith.constant 0 : i32
    %c512_i64 = arith.constant 512 : i64
    %c384_i64 = arith.constant 384 : i64
    %false = arith.constant false
    %c0 = arith.constant 0 : index
    %c1024_i32 = arith.constant 1024 : i32
    %c64_i32 = arith.constant 64 : i32
    %c8_i32 = arith.constant 8 : i32
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c1024_i64 = arith.constant 1024 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c512_i32 = arith.constant 512 : i32
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c4_i32 = arith.constant 4 : i32
    %c128_i32 = arith.constant 128 : i32
    %c1152_i64 = arith.constant 1152 : i64
    %c576_i32 = arith.constant 576 : i32
    %c256 = arith.constant 256 : index
    %c15 = arith.constant 15 : index
    %c16 = arith.constant 16 : index
    %c16384_i64 = arith.constant 16384 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %c288_i64 = arith.constant 288 : i64
    %c131072_i64 = arith.constant 131072 : i64
    %c192_i64 = arith.constant 192 : i64
    %c65536_i64 = arith.constant 65536 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %c36_i64 = arith.constant 36 : i64
    %c32_i64 = arith.constant 32 : i64
    %c17_i64 = arith.constant 17 : i64
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
    %alloc = memref.alloc() {cannir.arg_index = 0 : i64} : memref<1xui8>
    %alloc_0 = memref.alloc() {cannir.arg_index = 1 : i64} : memref<1048576xui8>
    %alloc_1 = memref.alloc() {cannir.arg_index = 2 : i64} : memref<393216xui8>
    %alloc_2 = memref.alloc() {cannir.arg_index = 3 : i64} : memref<589824xui8>
    %alloc_3 = memref.alloc() {cannir.arg_index = 4 : i64} : memref<131072xbf16>
    %alloc_4 = memref.alloc() {cannir.arg_index = 17 : i64} : memref<1xf32>
    %alloc_5 = memref.alloc() {cannir.arg_index = 20 : i64} : memref<1xui8>
    %alloc_6 = memref.alloc() {cannir.arg_index = 22 : i64} : memref<1xf32>
    %alloc_7 = memref.alloc() {cannir.arg_index = 23 : i64} : memref<1xbf16>
    %alloc_8 = memref.alloc() {cannir.arg_index = 26 : i64} : memref<1xbf16>
    %alloc_9 = memref.alloc() {cannir.arg_index = 32 : i64} : memref<1xf32>
    %alloc_10 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xf32, 5>
    %alloc_11 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xf32, 5>
    %alloc_12 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048xf32, 5>
    %alloc_13 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<16384xbf16, 4>
    %alloc_14 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384xbf16, 4>
    %alloc_15 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<1024xbf16, 3>
    %alloc_16 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xbf16, 3>
    %alloc_17 = memref.alloc() {cannir.byte_offset = 36864 : i64} : memref<16384xbf16, 2>
    %alloc_18 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<16384xbf16, 2>
    %alloc_19 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<1024xbf16, 2>
    %alloc_20 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<1024xbf16, 2>
    %alloc_21 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<4096xf32, 5>
    %alloc_22 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xf32, 5>
    %alloc_23 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<16384x!ascdtype.hifloat8, 4>
    %alloc_24 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384x!ascdtype.hifloat8, 4>
    %alloc_25 = memref.alloc() {cannir.byte_offset = 2048 : i64} : memref<2048x!ascdtype.hifloat8, 3>
    %alloc_26 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<2048x!ascdtype.hifloat8, 3>
    %alloc_27 = memref.alloc() {cannir.byte_offset = 81920 : i64} : memref<65536xui8, 2>
    %alloc_28 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<65536xui8, 2>
    %alloc_29 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<8192xui8, 2>
    %alloc_30 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<8192xui8, 2>
    %alloc_31 = memref.alloc() {cannir.byte_offset = 16384 : i64} : memref<16384x!ascdtype.hifloat8, 4>
    %alloc_32 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<16384x!ascdtype.hifloat8, 4>
    %alloc_33 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<4096x!ascdtype.hifloat8, 3>
    %alloc_34 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096x!ascdtype.hifloat8, 3>
    %alloc_35 = memref.alloc() {cannir.byte_offset = 24576 : i64} : memref<16384xui8, 2>
    %alloc_36 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<16384xui8, 2>
    %alloc_37 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<4096xui8, 2>
    %alloc_38 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xui8, 2>
    %alloc_39 = memref.alloc() {cannir.byte_offset = 32768 : i64} : memref<32768x!ascdtype.hifloat8, 4>
    %alloc_40 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<32768x!ascdtype.hifloat8, 4>
    %alloc_41 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<4096x!ascdtype.hifloat8, 3>
    %alloc_42 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096x!ascdtype.hifloat8, 3>
    %alloc_43 = memref.alloc() {cannir.byte_offset = 40960 : i64} : memref<32768xui8, 2>
    %alloc_44 = memref.alloc() {cannir.byte_offset = 8192 : i64} : memref<32768xui8, 2>
    %alloc_45 = memref.alloc() {cannir.byte_offset = 4096 : i64} : memref<4096xui8, 2>
    %alloc_46 = memref.alloc() {cannir.byte_offset = 0 : i64} : memref<4096xui8, 2>
    %0 = ascscalar.get_block_idx
    %1 = ascscalar.get_block_num
    %2 = scf.for %arg33 = %0 to %c8_i64 step %1 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
      %62 = arith.remsi %arg33, %c8_i64 : i64
      %63 = arith.cmpi ne, %62, %c0_i64 : i64
      %64 = arith.cmpi slt, %62, %c0_i64 : i64
      %65 = arith.andi %63, %64 : i1
      %66 = arith.divsi %arg33, %c8_i64 : i64
      %67 = arith.extui %65 : i1 to i64
      %68 = arith.subi %66, %67 : i64
      %69 = arith.select %65, %c8_i64, %c0_i64 : i64
      %70 = arith.addi %62, %69 overflow<nsw> : i64
      %71 = arith.muli %68, %c16_i64 : i64
      %72 = arith.subi %c3_i64, %71 : i64
      %73 = arith.minsi %72, %c16_i64 : i64
      %74 = arith.addi %73, %c15_i64 : i64
      %75 = arith.remsi %74, %c16_i64 : i64
      %76 = arith.cmpi ne, %75, %c0_i64 : i64
      %77 = arith.cmpi slt, %75, %c0_i64 : i64
      %78 = arith.andi %76, %77 : i1
      %79 = arith.divsi %74, %c16_i64 : i64
      %80 = arith.extui %78 : i1 to i64
      %81 = arith.subi %79, %80 : i64
      %82 = arith.muli %81, %c256_i64 : i64
      %83:2 = scf.for %arg35 = %c0_i64 to %c5_i64 step %c1_i64 iter_args(%arg36 = %c0_i64, %arg37 = %arg34) -> (i64, i64)  : i64 {
        %93 = arith.cmpi slt, %arg35, %c4_i64 : i64
        %94 = scf.if %93 -> (i64) {
          %97 = arith.cmpi eq, %arg36, %c0_i64 : i64
          %98 = ascscalar.memref_address %alloc_46 : memref<4096xui8, 2>
          %99 = ascscalar.memref_address %alloc_45 : memref<4096xui8, 2>
          %100 = arith.select %97, %98, %99 : i64
          %101 = arith.cmpi ne, %arg36, %c0_i64 : i64
          %102 = arith.extui %101 : i1 to i64
          %103 = arith.addi %arg36, %c1_i64 overflow<nsw> : i64
          %104 = arith.remui %103, %c2_i64 : i64
          %105 = arith.divsi %73, %c16_i64 : i64
          %106 = arith.remsi %73, %c16_i64 : i64
          %107 = arith.addi %106, %c15_i64 overflow<nsw> : i64
          %108 = arith.divsi %107, %c16_i64 : i64
          %109 = arith.addi %105, %108 : i64
          %110 = arith.muli %109, %c512_i64 : i64
          %111 = arith.index_cast %110 : i64 to index
          %112 = arith.divui %111, %c32 : index
          %113 = ascscalar.bind_local_ptr %100 : memref<1xui8, 2>
          %114 = arith.muli %68, %c2048_i64 : i64
          %115 = arith.muli %arg35, %c32_i64 overflow<nsw> : i64
          %116 = arith.addi %114, %115 : i64
          %117 = arith.muli %116, %c8_i64 : i64
          %118 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, ui8>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
          %119 = cannir.extract_shape %118 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
          %120 = cannir.extract_dim %119[0] : !cannir.shape<(-1, 1024)>
          %121 = arith.divsi %120, %c16_i64 : i64
          %122 = arith.minsi %68, %121 : i64
          %123 = arith.muli %122, %c16_i64 : i64
          %124 = arith.subi %120, %123 : i64
          %125 = arith.minsi %124, %c16_i64 : i64
          %126 = arith.cmpi sle, %68, %121 : i64
          %127 = arith.select %126, %125, %c0_i64 : i64
          %128 = arith.index_cast %112 : index to i64
          asccube.get_buf mte2, %102, 0 {cannir.buffer.generated_sync}
          %129 = arith.cmpi ugt, %127, %c0_i64 : i64
          scf.if %129 {
            %139 = arith.index_cast %117 : i64 to index
            %140 = arith.trunci %127 : i64 to i32
            %141 = arith.trunci %128 : i64 to i32
            asccube.copy_gm2l1 %113[%c0], %alloc[%139], %140, %c256_i32, %c1024_i64, %c1_i32, %c0_i64, %c0_i64, %141 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xui8, 2>, memref<1xui8>
          }
          asccube.release_buf mte2, %102, 0 {cannir.buffer.generated_sync}
          %130 = ascscalar.memref_address %alloc_44 : memref<32768xui8, 2>
          %131 = ascscalar.memref_address %alloc_43 : memref<32768xui8, 2>
          %132 = arith.select %97, %130, %131 : i64
          %133 = arith.select %97, %c2_i64, %c3_i64 : i64
          %134 = ascscalar.bind_local_ptr %132 : memref<1xui8, 2>
          %135 = arith.muli %70, %c16384_i64 overflow<nsw> : i64
          %136 = arith.addi %135, %115 overflow<nsw> : i64
          %137 = arith.muli %136, %c8_i64 overflow<nsw> : i64
          asccube.get_buf mte2, %133, 0 {cannir.buffer.generated_sync}
          %138 = arith.index_cast %137 : i64 to index
          asccube.copy_gm2l1 %134[%c0], %alloc_0[%138], %c128_i32, %c256_i32, %c1024_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xui8, 2>, memref<1048576xui8>
          asccube.release_buf mte2, %133, 0 {cannir.buffer.generated_sync}
          scf.yield %104 : i64
        } else {
          scf.yield %arg36 : i64
        }
        %95 = arith.cmpi sgt, %arg35, %c0_i64 : i64
        %96 = scf.if %95 -> (i64) {
          %97 = arith.cmpi eq, %arg37, %c0_i64 : i64
          %98 = ascscalar.memref_address %alloc_46 : memref<4096xui8, 2>
          %99 = ascscalar.memref_address %alloc_45 : memref<4096xui8, 2>
          %100 = arith.select %97, %98, %99 : i64
          %101 = arith.cmpi ne, %arg37, %c0_i64 : i64
          %102 = arith.extui %101 : i1 to i64
          %103 = arith.addi %arg37, %c1_i64 : i64
          %104 = arith.remui %103, %c2_i64 : i64
          %105 = ascscalar.memref_address %alloc_44 : memref<32768xui8, 2>
          %106 = ascscalar.memref_address %alloc_43 : memref<32768xui8, 2>
          %107 = arith.select %97, %105, %106 : i64
          %108 = arith.select %97, %c2_i64, %c3_i64 : i64
          %109 = ascscalar.memref_address %alloc_42 : memref<4096x!ascdtype.hifloat8, 3>
          %110 = ascscalar.memref_address %alloc_41 : memref<4096x!ascdtype.hifloat8, 3>
          %111 = arith.select %97, %109, %110 : i64
          %112 = arith.select %97, %c4_i64, %c5_i64 : i64
          %113 = arith.divsi %73, %c16_i64 : i64
          %114 = arith.remsi %73, %c16_i64 : i64
          %115 = arith.addi %114, %c15_i64 overflow<nsw> : i64
          %116 = arith.divsi %115, %c16_i64 : i64
          %117 = arith.addi %113, %116 : i64
          %118 = arith.muli %117, %c512_i64 : i64
          %119 = ascscalar.bind_local_ptr %111 : memref<1x!ascdtype.hifloat8, 3>
          %120 = ascscalar.bind_local_ptr %100 : memref<1x!ascdtype.hifloat8, 2>
          %121 = arith.index_cast %73 : i64 to index
          %122 = arith.addi %121, %c15 : index
          %123 = arith.divui %122, %c16 : index
          %124 = arith.index_cast %123 : index to i64
          %125 = arith.index_cast %118 : i64 to index
          %126 = arith.divui %125, %c512 : index
          %127 = arith.index_cast %126 : index to i64
          %128 = arith.cmpi ugt, %124, %c0_i64 : i64
          scf.if %128 {
            asccube.get_buf mte1, %112, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %102, 0 {cannir.buffer.generated_sync}
            %138 = arith.trunci %124 : i64 to i32
            %139 = arith.trunci %127 : i64 to i32
            asccube.copy_l12l0a %119[%c0], %120[%c0], %c0_i32, %c0_i32, %138, %c8_i32, %139, %139 : memref<1x!ascdtype.hifloat8, 3>, memref<1x!ascdtype.hifloat8, 2>
            asccube.release_buf mte1, %102, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %112, 0 {cannir.buffer.generated_sync}
          }
          %129 = ascscalar.memref_address %alloc_40 : memref<32768x!ascdtype.hifloat8, 4>
          %130 = ascscalar.memref_address %alloc_39 : memref<32768x!ascdtype.hifloat8, 4>
          %131 = arith.select %97, %129, %130 : i64
          %132 = arith.select %97, %c6_i64, %c7_i64 : i64
          %133 = ascscalar.bind_local_ptr %131 : memref<1x!ascdtype.hifloat8, 4>
          %134 = ascscalar.bind_local_ptr %107 : memref<1x!ascdtype.hifloat8, 2>
          asccube.get_buf mte1, %132, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %108, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %133[%c0], %134[%c0], %c0_i32, %c0_i32, %c8_i32, %c8_i32, %c8_i32, %c8_i32 : memref<1x!ascdtype.hifloat8, 4>, memref<1x!ascdtype.hifloat8, 2>
          asccube.release_buf mte1, %108, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %132, 0 {cannir.buffer.generated_sync}
          %135 = arith.cmpi eq, %arg35, %c1_i64 : i64
          %136 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<2048xf32, 5> -> memref<?xf32, 5>
          %137 = arith.cmpi ugt, %73, %c0_i64 : i64
          scf.if %137 {
            asccube.get_buf m, %c8_i64, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %112, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %132, 0 {cannir.buffer.generated_sync}
            scf.if %135 {
              %138 = arith.trunci %73 : i64 to i32
              asccube.mmad %136[%c0], %119, %133, %138, %c256_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1x!ascdtype.hifloat8, 3>, memref<1x!ascdtype.hifloat8, 4>
            } else {
              %138 = arith.trunci %73 : i64 to i32
              asccube.mmad %136[%c0], %119, %133, %138, %c256_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<1x!ascdtype.hifloat8, 3>, memref<1x!ascdtype.hifloat8, 4>
            }
            asccube.release_buf m, %132, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %112, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %c8_i64, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %104 : i64
        } else {
          scf.yield %arg37 : i64
        }
        scf.yield %94, %96 : i64, i64
      }
      %84 = ascscalar.bitcast_memref %alloc_10 byte_offset = %c0_i64 : memref<2048xf32, 5> -> memref<?xf32, 5>
      %85 = arith.muli %68, %c512_i64 : i64
      %86 = arith.muli %70, %c4_i64 overflow<nsw> : i64
      %87 = arith.addi %85, %86 : i64
      %88 = arith.muli %87, %c32_i64 : i64
      %89 = arith.index_cast %82 : i64 to index
      %90 = arith.divui %89, %c16 : index
      %91 = arith.index_cast %90 : index to i64
      %92 = arith.cmpi ugt, %73, %c0_i64 : i64
      scf.if %92 {
        asccube.get_buf fixpipe, %c8_i64, 0 {cannir.buffer.generated_sync}
        %93 = arith.trunci %73 : i64 to i32
        %94 = arith.trunci %91 : i64 to i32
        %95 = arith.index_cast %88 : i64 to index
        asccube.copy_l0c2gm %alloc_9[%95], %84[%c0], %c128_i32, %93, %c1024_i32, %94, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xf32>, memref<?xf32, 5>
        asccube.release_buf fixpipe, %c8_i64, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %83#1 : i64
    }
    asccube.sync_all
    asccube.sync_block_arrive fixpipe, %c1_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c1_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c2_i64
    %3 = cannir.extract_layout %arg0 : !cannir.tensor<<gm, ui8>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
    %4 = cannir.extract_shape %3 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
    %5 = cannir.extract_dim %4[0] : !cannir.shape<(-1, 1024)>
    %6 = arith.addi %5, %c15_i64 : i64
    %7 = arith.remsi %6, %c16_i64 : i64
    %8 = arith.cmpi ne, %7, %c0_i64 : i64
    %9 = arith.cmpi slt, %7, %c0_i64 : i64
    %10 = arith.andi %8, %9 : i1
    %11 = arith.divsi %6, %c16_i64 : i64
    %12 = arith.extui %10 : i1 to i64
    %13 = arith.subi %11, %12 : i64
    %14 = arith.muli %13, %c9_i64 : i64
    %15 = scf.for %arg33 = %0 to %14 step %1 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
      %62 = arith.remsi %arg33, %c9_i64 : i64
      %63 = arith.cmpi ne, %62, %c0_i64 : i64
      %64 = arith.cmpi slt, %62, %c0_i64 : i64
      %65 = arith.andi %63, %64 : i1
      %66 = arith.divsi %arg33, %c9_i64 : i64
      %67 = arith.extui %65 : i1 to i64
      %68 = arith.subi %66, %67 : i64
      %69 = arith.select %65, %c9_i64, %c0_i64 : i64
      %70 = arith.addi %62, %69 overflow<nsw> : i64
      %71:2 = scf.for %arg35 = %c0_i64 to %c5_i64 step %c1_i64 iter_args(%arg36 = %c0_i64, %arg37 = %arg34) -> (i64, i64)  : i64 {
        %88 = arith.cmpi slt, %arg35, %c4_i64 : i64
        %89 = scf.if %88 -> (i64) {
          %92 = arith.cmpi eq, %arg36, %c0_i64 : i64
          %93 = ascscalar.memref_address %alloc_38 : memref<4096xui8, 2>
          %94 = ascscalar.memref_address %alloc_37 : memref<4096xui8, 2>
          %95 = arith.select %92, %93, %94 : i64
          %96 = arith.select %92, %c9_i64, %c10_i64 : i64
          %97 = arith.addi %arg36, %c1_i64 overflow<nsw> : i64
          %98 = arith.remui %97, %c2_i64 : i64
          %99 = ascscalar.bind_local_ptr %95 : memref<1xui8, 2>
          %100 = arith.muli %68, %c2048_i64 : i64
          %101 = arith.muli %arg35, %c32_i64 overflow<nsw> : i64
          %102 = arith.addi %100, %101 : i64
          %103 = arith.muli %102, %c8_i64 : i64
          %104 = arith.divsi %5, %c16_i64 : i64
          %105 = arith.minsi %68, %104 : i64
          %106 = arith.muli %105, %c16_i64 : i64
          %107 = arith.subi %5, %106 : i64
          %108 = arith.minsi %107, %c16_i64 : i64
          %109 = arith.cmpi sle, %68, %104 : i64
          %110 = arith.select %109, %108, %c0_i64 : i64
          asccube.get_buf mte2, %96, 0 {cannir.buffer.generated_sync}
          %111 = arith.cmpi ugt, %110, %c0_i64 : i64
          scf.if %111 {
            %121 = arith.index_cast %103 : i64 to index
            %122 = arith.trunci %110 : i64 to i32
            asccube.copy_gm2l1 %99[%c0], %alloc[%121], %122, %c256_i32, %c1024_i64, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xui8, 2>, memref<1xui8>
          }
          asccube.release_buf mte2, %96, 0 {cannir.buffer.generated_sync}
          %112 = ascscalar.memref_address %alloc_36 : memref<16384xui8, 2>
          %113 = ascscalar.memref_address %alloc_35 : memref<16384xui8, 2>
          %114 = arith.select %92, %112, %113 : i64
          %115 = arith.select %92, %c11_i64, %c12_i64 : i64
          %116 = ascscalar.bind_local_ptr %114 : memref<1xui8, 2>
          %117 = arith.muli %70, %c8192_i64 overflow<nsw> : i64
          %118 = arith.addi %117, %101 overflow<nsw> : i64
          %119 = arith.muli %118, %c8_i64 overflow<nsw> : i64
          asccube.get_buf mte2, %115, 0 {cannir.buffer.generated_sync}
          %120 = arith.index_cast %119 : i64 to index
          asccube.copy_gm2l1 %116[%c0], %alloc_2[%120], %c64_i32, %c256_i32, %c1024_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xui8, 2>, memref<589824xui8>
          asccube.release_buf mte2, %115, 0 {cannir.buffer.generated_sync}
          scf.yield %98 : i64
        } else {
          scf.yield %arg36 : i64
        }
        %90 = arith.cmpi sgt, %arg35, %c0_i64 : i64
        %91 = scf.if %90 -> (i64) {
          %92 = arith.cmpi eq, %arg37, %c0_i64 : i64
          %93 = ascscalar.memref_address %alloc_38 : memref<4096xui8, 2>
          %94 = ascscalar.memref_address %alloc_37 : memref<4096xui8, 2>
          %95 = arith.select %92, %93, %94 : i64
          %96 = arith.select %92, %c9_i64, %c10_i64 : i64
          %97 = arith.addi %arg37, %c1_i64 : i64
          %98 = arith.remui %97, %c2_i64 : i64
          %99 = ascscalar.memref_address %alloc_36 : memref<16384xui8, 2>
          %100 = ascscalar.memref_address %alloc_35 : memref<16384xui8, 2>
          %101 = arith.select %92, %99, %100 : i64
          %102 = arith.select %92, %c11_i64, %c12_i64 : i64
          %103 = ascscalar.memref_address %alloc_34 : memref<4096x!ascdtype.hifloat8, 3>
          %104 = ascscalar.memref_address %alloc_33 : memref<4096x!ascdtype.hifloat8, 3>
          %105 = arith.select %92, %103, %104 : i64
          %106 = arith.select %92, %c13_i64, %c14_i64 : i64
          %107 = ascscalar.bind_local_ptr %105 : memref<1x!ascdtype.hifloat8, 3>
          %108 = ascscalar.bind_local_ptr %95 : memref<1x!ascdtype.hifloat8, 2>
          asccube.get_buf mte1, %106, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %96, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0a %107[%c0], %108[%c0], %c0_i32, %c0_i32, %c1_i32, %c8_i32, %c1_i32, %c1_i32 : memref<1x!ascdtype.hifloat8, 3>, memref<1x!ascdtype.hifloat8, 2>
          asccube.release_buf mte1, %96, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %106, 0 {cannir.buffer.generated_sync}
          %109 = ascscalar.memref_address %alloc_32 : memref<16384x!ascdtype.hifloat8, 4>
          %110 = ascscalar.memref_address %alloc_31 : memref<16384x!ascdtype.hifloat8, 4>
          %111 = arith.select %92, %109, %110 : i64
          %112 = arith.select %92, %c15_i64, %c16_i64 : i64
          %113 = ascscalar.bind_local_ptr %111 : memref<1x!ascdtype.hifloat8, 4>
          %114 = ascscalar.bind_local_ptr %101 : memref<1x!ascdtype.hifloat8, 2>
          asccube.get_buf mte1, %112, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %102, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %113[%c0], %114[%c0], %c0_i32, %c0_i32, %c4_i32, %c8_i32, %c4_i32, %c4_i32 : memref<1x!ascdtype.hifloat8, 4>, memref<1x!ascdtype.hifloat8, 2>
          asccube.release_buf mte1, %102, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %112, 0 {cannir.buffer.generated_sync}
          %115 = arith.cmpi eq, %arg35, %c1_i64 : i64
          %116 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<1024xf32, 5> -> memref<?xf32, 5>
          asccube.get_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %106, 0 {cannir.buffer.generated_sync}
          asccube.get_buf m, %112, 0 {cannir.buffer.generated_sync}
          scf.if %115 {
            asccube.mmad %116[%c0], %107, %113, %c16_i32, %c256_i32, %c64_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1x!ascdtype.hifloat8, 3>, memref<1x!ascdtype.hifloat8, 4>
          } else {
            asccube.mmad %116[%c0], %107, %113, %c16_i32, %c256_i32, %c64_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<1x!ascdtype.hifloat8, 3>, memref<1x!ascdtype.hifloat8, 4>
          }
          asccube.release_buf m, %112, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %106, 0 {cannir.buffer.generated_sync}
          asccube.release_buf m, %c17_i64, 0 {cannir.buffer.generated_sync}
          scf.yield %98 : i64
        } else {
          scf.yield %arg37 : i64
        }
        scf.yield %89, %91 : i64, i64
      }
      %72 = cannir.extract_layout %arg17 : !cannir.tensor<<gm, f32>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>
      %73 = cannir.extract_shape %72 : <<(-1, 576)>, <(576, 1)>> -> <(-1, 576)>
      %74 = cannir.extract_dim %73[0] : !cannir.shape<(-1, 576)>
      %75 = arith.divsi %74, %c16_i64 : i64
      %76 = arith.minsi %68, %75 : i64
      %77 = arith.muli %76, %c16_i64 : i64
      %78 = arith.subi %74, %77 : i64
      %79 = arith.minsi %78, %c16_i64 : i64
      %80 = arith.cmpi sle, %68, %75 : i64
      %81 = arith.select %80, %79, %c0_i64 : i64
      %82 = ascscalar.bitcast_memref %alloc_11 byte_offset = %c0_i64 : memref<1024xf32, 5> -> memref<?xf32, 5>
      %83 = arith.muli %68, %c288_i64 : i64
      %84 = arith.muli %70, %c2_i64 overflow<nsw> : i64
      %85 = arith.addi %83, %84 : i64
      %86 = arith.muli %85, %c32_i64 : i64
      %87 = arith.cmpi ugt, %81, %c0_i64 : i64
      scf.if %87 {
        asccube.get_buf fixpipe, %c17_i64, 0 {cannir.buffer.generated_sync}
        %88 = arith.trunci %81 : i64 to i32
        %89 = arith.index_cast %86 : i64 to index
        asccube.copy_l0c2gm %alloc_4[%89], %82[%c0], %c64_i32, %88, %c576_i32, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xf32>, memref<?xf32, 5>
        asccube.release_buf fixpipe, %c17_i64, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %71#1 : i64
    }
    %16 = arith.muli %1, %c2_i64 : i64
    %17 = arith.cmpi sle, %16, %c128_i64 : i64
    %18 = scf.if %17 -> (i1) {
      %62 = cannir.extract_layout %arg22 : !cannir.tensor<<gm, f32>, <<(-1, 384)>, <(384, 1)>>, <<(-1, 384)>, <(384, 1)>>, identity>
      %63 = cannir.extract_shape %62 : <<(-1, 384)>, <(384, 1)>> -> <(-1, 384)>
      %64 = cannir.extract_dim %63[0] : !cannir.shape<(-1, 384)>
      %65 = arith.muli %64, %c384_i64 : i64
      %66 = arith.remsi %65, %c8_i64 : i64
      %67 = arith.cmpi ne, %66, %c0_i64 : i64
      %68 = arith.cmpi slt, %66, %c0_i64 : i64
      %69 = arith.andi %67, %68 : i1
      %70 = arith.divsi %65, %c8_i64 : i64
      %71 = arith.extui %69 : i1 to i64
      %72 = arith.subi %70, %71 : i64
      %73 = arith.cmpi sle, %16, %72 : i64
      scf.yield %73 : i1
    } else {
      scf.yield %false : i1
    }
    %19 = arith.muli %1, %c16_i64 : i64
    %20 = arith.cmpi sle, %5, %19 : i64
    %21 = arith.select %18, %20, %false : i1
    %22 = arith.cmpi sle, %5, %c4_i64 : i64
    %23 = arith.select %22, %true, %21 : i1
    scf.if %22 {
    } else {
      scf.if %21 {
        asccube.sync_all
        asccube.sync_block_wait s, %c0_i64
        asccube.sync_block_arrive fixpipe, %c1_i64 {mode = 0 : i32}
        asccube.sync_block_wait s, %c1_i64 {mode = 0 : i32}
        asccube.sync_block_arrive mte3, %c2_i64
      }
    }
    scf.if %23 {
      asccube.sync_all
      asccube.sync_block_arrive fixpipe, %c6_i64 {mode = 0 : i32}
      asccube.sync_block_wait s, %c6_i64 {mode = 0 : i32}
      asccube.sync_block_arrive mte3, %c7_i64
    } else {
      asccube.sync_all
      asccube.sync_block_wait s, %c0_i64
      asccube.sync_block_arrive fixpipe, %c1_i64 {mode = 0 : i32}
      asccube.sync_block_wait s, %c1_i64 {mode = 0 : i32}
      asccube.sync_block_arrive mte3, %c2_i64
    }
    asccube.sync_all
    %24 = arith.cmpi sgt, %5, %c4_i64 : i64
    %25 = arith.xori %21, %true : i1
    %26 = arith.select %24, %25, %false : i1
    scf.if %26 {
      scf.if %18 {
      } else {
        %62 = cannir.extract_layout %arg18 : !cannir.tensor<<gm, f32>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
        %63 = cannir.extract_shape %62 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
        %64 = cannir.extract_dim %63[0] : !cannir.shape<(-1, 1024)>
        %65 = arith.cmpi sge, %64, %c16_i64 : i64
        scf.if %65 {
          asccube.sync_all
          asccube.sync_block_wait s, %c0_i64
          asccube.sync_block_arrive fixpipe, %c1_i64 {mode = 0 : i32}
          asccube.sync_block_wait s, %c1_i64 {mode = 0 : i32}
          asccube.sync_block_arrive mte3, %c2_i64
        }
        asccube.sync_all
        asccube.sync_block_wait s, %c0_i64
        asccube.sync_block_arrive fixpipe, %c1_i64 {mode = 0 : i32}
        asccube.sync_block_wait s, %c1_i64 {mode = 0 : i32}
        asccube.sync_block_arrive mte3, %c2_i64
      }
      asccube.sync_all
      asccube.sync_block_wait s, %c0_i64
      asccube.sync_block_arrive fixpipe, %c1_i64 {mode = 0 : i32}
      asccube.sync_block_wait s, %c1_i64 {mode = 0 : i32}
      asccube.sync_block_arrive mte3, %c2_i64
    }
    asccube.sync_all
    %27 = cannir.extract_layout %arg20 : !cannir.tensor<<gm, ui8>, <<(-1, 1024)>, <(1024, 1)>>, <<(-1, 1024)>, <(1024, 1)>>, identity>
    %28 = cannir.extract_shape %27 : <<(-1, 1024)>, <(1024, 1)>> -> <(-1, 1024)>
    %29 = cannir.extract_dim %28[0] : !cannir.shape<(-1, 1024)>
    %30 = arith.addi %29, %c15_i64 : i64
    %31 = arith.remsi %30, %c16_i64 : i64
    %32 = arith.cmpi ne, %31, %c0_i64 : i64
    %33 = arith.cmpi slt, %31, %c0_i64 : i64
    %34 = arith.andi %32, %33 : i1
    %35 = arith.divsi %30, %c16_i64 : i64
    %36 = arith.extui %34 : i1 to i64
    %37 = arith.subi %35, %36 : i64
    %38 = arith.muli %37, %c3_i64 : i64
    %39 = scf.if %23 -> (i64) {
      %62 = arith.cmpi slt, %0, %38 : i64
      %63 = arith.extui %62 : i1 to i64
      scf.if %62 {
        %64 = arith.remsi %0, %c3_i64 : i64
        %65 = arith.cmpi ne, %64, %c0_i64 : i64
        %66 = arith.cmpi slt, %64, %c0_i64 : i64
        %67 = arith.andi %65, %66 : i1
        %68 = arith.select %67, %c3_i64, %c0_i64 : i64
        %69 = arith.addi %64, %68 overflow<nsw> : i64
        %70 = ascscalar.bitcast_memref %alloc_28 byte_offset = %c0_i64 : memref<65536xui8, 2> -> memref<?xui8, 2>
        %71 = arith.muli %69, %c131072_i64 overflow<nsw> : i64
        asccube.get_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
        %72 = arith.index_cast %71 : i64 to index
        asccube.copy_gm2l1 %70[%c0], %alloc_1[%72], %c128_i32, %c512_i32, %c1024_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<?xui8, 2>, memref<393216xui8>
        asccube.release_buf mte2, %c2_i64, 0 {cannir.buffer.generated_sync}
      }
      asccube.sync_block_wait s, %c8_i64
      asccube.sync_block_arrive fixpipe, %c1_i64 {mode = 0 : i32}
      asccube.sync_block_wait s, %c1_i64 {mode = 0 : i32}
      scf.yield %63 : i64
    } else {
      scf.yield %c0_i64 : i64
    }
    %40:3 = scf.for %arg33 = %0 to %38 step %1 iter_args(%arg34 = %c0_i64, %arg35 = %39, %arg36 = %c0_i64) -> (i64, i64, i64)  : i64 {
      %62 = arith.remsi %arg33, %c3_i64 : i64
      %63 = arith.cmpi ne, %62, %c0_i64 : i64
      %64 = arith.cmpi slt, %62, %c0_i64 : i64
      %65 = arith.andi %63, %64 : i1
      %66 = arith.divsi %arg33, %c3_i64 : i64
      %67 = arith.extui %65 : i1 to i64
      %68 = arith.subi %66, %67 : i64
      %69 = arith.select %65, %c3_i64, %c0_i64 : i64
      %70 = arith.addi %62, %69 overflow<nsw> : i64
      %71:4 = scf.for %arg37 = %c0_i64 to %c3_i64 step %c1_i64 iter_args(%arg38 = %c0_i64, %arg39 = %arg34, %arg40 = %arg35, %arg41 = %arg36) -> (i64, i64, i64, i64)  : i64 {
        %88 = arith.cmpi slt, %arg37, %c2_i64 : i64
        %89:2 = scf.if %88 -> (i64, i64) {
          %92 = arith.cmpi eq, %arg38, %c0_i64 : i64
          %93 = ascscalar.memref_address %alloc_30 : memref<8192xui8, 2>
          %94 = ascscalar.memref_address %alloc_29 : memref<8192xui8, 2>
          %95 = arith.select %92, %93, %94 : i64
          %96 = arith.cmpi ne, %arg38, %c0_i64 : i64
          %97 = arith.extui %96 : i1 to i64
          %98 = arith.addi %arg38, %c1_i64 overflow<nsw> : i64
          %99 = arith.remui %98, %c2_i64 : i64
          %100 = ascscalar.bind_local_ptr %95 : memref<1xui8, 2>
          %101 = arith.muli %68, %c2048_i64 : i64
          %102 = arith.muli %arg37, %c64_i64 overflow<nsw> : i64
          %103 = arith.addi %101, %102 : i64
          %104 = arith.muli %103, %c8_i64 : i64
          %105 = arith.divsi %29, %c16_i64 : i64
          %106 = arith.minsi %68, %105 : i64
          %107 = arith.muli %106, %c16_i64 : i64
          %108 = arith.subi %29, %107 : i64
          %109 = arith.minsi %108, %c16_i64 : i64
          %110 = arith.cmpi sle, %68, %105 : i64
          %111 = arith.select %110, %109, %c0_i64 : i64
          asccube.get_buf mte2, %97, 0 {cannir.buffer.generated_sync}
          %112 = arith.cmpi ugt, %111, %c0_i64 : i64
          scf.if %112 {
            %118 = arith.index_cast %104 : i64 to index
            %119 = arith.trunci %111 : i64 to i32
            asccube.copy_gm2l1 %100[%c0], %alloc_5[%118], %119, %c512_i32, %c1024_i64, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xui8, 2>, memref<1xui8>
          }
          asccube.release_buf mte2, %97, 0 {cannir.buffer.generated_sync}
          %113 = arith.cmpi ne, %arg33, %0 : i64
          %114 = arith.cmpi ne, %arg37, %c0_i64 : i64
          %115 = arith.select %113, %true, %114 : i1
          %116 = arith.select %23, %115, %true : i1
          %117 = scf.if %116 -> (i64) {
            %118 = arith.cmpi eq, %arg40, %c0_i64 : i64
            %119 = ascscalar.memref_address %alloc_28 : memref<65536xui8, 2>
            %120 = ascscalar.memref_address %alloc_27 : memref<65536xui8, 2>
            %121 = arith.select %118, %119, %120 : i64
            %122 = arith.select %118, %c2_i64, %c3_i64 : i64
            %123 = arith.addi %arg40, %c1_i64 : i64
            %124 = arith.remui %123, %c2_i64 : i64
            %125 = ascscalar.bind_local_ptr %121 : memref<1xui8, 2>
            %126 = arith.muli %70, %c16384_i64 overflow<nsw> : i64
            %127 = arith.addi %126, %102 overflow<nsw> : i64
            %128 = arith.muli %127, %c8_i64 overflow<nsw> : i64
            asccube.get_buf mte2, %122, 0 {cannir.buffer.generated_sync}
            %129 = arith.index_cast %128 : i64 to index
            asccube.copy_gm2l1 %125[%c0], %alloc_1[%129], %c128_i32, %c512_i32, %c1024_i64, %c1_i32, %c0_i64, %c0_i64, %c128_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xui8, 2>, memref<393216xui8>
            asccube.release_buf mte2, %122, 0 {cannir.buffer.generated_sync}
            scf.yield %124 : i64
          } else {
            scf.yield %arg40 : i64
          }
          scf.yield %99, %117 : i64, i64
        } else {
          scf.yield %arg38, %arg40 : i64, i64
        }
        %90 = arith.cmpi sgt, %arg37, %c0_i64 : i64
        %91:2 = scf.if %90 -> (i64, i64) {
          %92 = arith.cmpi eq, %arg39, %c0_i64 : i64
          %93 = ascscalar.memref_address %alloc_30 : memref<8192xui8, 2>
          %94 = ascscalar.memref_address %alloc_29 : memref<8192xui8, 2>
          %95 = arith.select %92, %93, %94 : i64
          %96 = arith.cmpi ne, %arg39, %c0_i64 : i64
          %97 = arith.extui %96 : i1 to i64
          %98 = arith.addi %arg39, %c1_i64 : i64
          %99 = arith.remui %98, %c2_i64 : i64
          %100 = ascscalar.memref_address %alloc_28 : memref<65536xui8, 2>
          %101 = ascscalar.memref_address %alloc_27 : memref<65536xui8, 2>
          %102 = arith.select %92, %100, %101 : i64
          %103 = arith.select %92, %c2_i64, %c3_i64 : i64
          %104 = scf.for %arg42 = %c0_i64 to %c4_i64 step %c1_i64 iter_args(%arg43 = %arg41) -> (i64)  : i64 {
            %105 = arith.subi %arg37, %c1_i64 overflow<nsw> : i64
            %106 = arith.muli %105, %c4_i64 overflow<nsw> : i64
            %107 = arith.addi %106, %arg42 overflow<nsw> : i64
            %108 = arith.cmpi eq, %arg43, %c0_i64 : i64
            %109 = ascscalar.memref_address %alloc_26 : memref<2048x!ascdtype.hifloat8, 3>
            %110 = ascscalar.memref_address %alloc_25 : memref<2048x!ascdtype.hifloat8, 3>
            %111 = arith.select %108, %109, %110 : i64
            %112 = arith.select %108, %c4_i64, %c5_i64 : i64
            %113 = arith.addi %arg43, %c1_i64 : i64
            %114 = arith.remui %113, %c2_i64 : i64
            %115 = ascscalar.bind_local_ptr %111 : memref<1x!ascdtype.hifloat8, 3>
            %116 = ascscalar.bind_local_ptr %95 : memref<1x!ascdtype.hifloat8, 2>
            %117 = arith.muli %arg42, %c4_i64 overflow<nsw> : i64
            asccube.get_buf mte1, %112, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %97, 0 {cannir.buffer.generated_sync}
            %118 = arith.trunci %117 : i64 to i32
            asccube.copy_l12l0a %115[%c0], %116[%c0], %c0_i32, %118, %c1_i32, %c4_i32, %c1_i32, %c1_i32 : memref<1x!ascdtype.hifloat8, 3>, memref<1x!ascdtype.hifloat8, 2>
            asccube.release_buf mte1, %97, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %112, 0 {cannir.buffer.generated_sync}
            %119 = ascscalar.memref_address %alloc_24 : memref<16384x!ascdtype.hifloat8, 4>
            %120 = ascscalar.memref_address %alloc_23 : memref<16384x!ascdtype.hifloat8, 4>
            %121 = arith.select %108, %119, %120 : i64
            %122 = arith.select %108, %c6_i64, %c7_i64 : i64
            %123 = ascscalar.bind_local_ptr %121 : memref<1x!ascdtype.hifloat8, 4>
            %124 = ascscalar.bind_local_ptr %102 : memref<1x!ascdtype.hifloat8, 2>
            asccube.get_buf mte1, %122, 0 {cannir.buffer.generated_sync}
            asccube.get_buf mte1, %103, 0 {cannir.buffer.generated_sync}
            asccube.copy_l12l0b %123[%c0], %124[%c0], %c0_i32, %118, %c8_i32, %c4_i32, %c8_i32, %c8_i32 : memref<1x!ascdtype.hifloat8, 4>, memref<1x!ascdtype.hifloat8, 2>
            asccube.release_buf mte1, %103, 0 {cannir.buffer.generated_sync}
            asccube.release_buf mte1, %122, 0 {cannir.buffer.generated_sync}
            %125 = arith.cmpi eq, %107, %c0_i64 : i64
            %126 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<2048xf32, 5> -> memref<?xf32, 5>
            asccube.get_buf m, %c8_i64, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %112, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %122, 0 {cannir.buffer.generated_sync}
            scf.if %125 {
              asccube.mmad %126[%c0], %115, %123, %c16_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = true} : memref<?xf32, 5>, memref<1x!ascdtype.hifloat8, 3>, memref<1x!ascdtype.hifloat8, 4>
            } else {
              asccube.mmad %126[%c0], %115, %123, %c16_i32, %c128_i32, %c128_i32, %c0_i32 {c_matrix_init_val = false} : memref<?xf32, 5>, memref<1x!ascdtype.hifloat8, 3>, memref<1x!ascdtype.hifloat8, 4>
            }
            asccube.release_buf m, %122, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %112, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %c8_i64, 0 {cannir.buffer.generated_sync}
            scf.yield %114 : i64
          }
          scf.yield %99, %104 : i64, i64
        } else {
          scf.yield %arg39, %arg41 : i64, i64
        }
        scf.yield %89#0, %91#0, %89#1, %91#1 : i64, i64, i64, i64
      }
      %72 = cannir.extract_layout %arg22 : !cannir.tensor<<gm, f32>, <<(-1, 384)>, <(384, 1)>>, <<(-1, 384)>, <(384, 1)>>, identity>
      %73 = cannir.extract_shape %72 : <<(-1, 384)>, <(384, 1)>> -> <(-1, 384)>
      %74 = cannir.extract_dim %73[0] : !cannir.shape<(-1, 384)>
      %75 = arith.divsi %74, %c16_i64 : i64
      %76 = arith.minsi %68, %75 : i64
      %77 = arith.muli %76, %c16_i64 : i64
      %78 = arith.subi %74, %77 : i64
      %79 = arith.minsi %78, %c16_i64 : i64
      %80 = arith.cmpi sle, %68, %75 : i64
      %81 = arith.select %80, %79, %c0_i64 : i64
      %82 = ascscalar.bitcast_memref %alloc_12 byte_offset = %c0_i64 : memref<2048xf32, 5> -> memref<?xf32, 5>
      %83 = arith.muli %68, %c192_i64 : i64
      %84 = arith.muli %70, %c4_i64 overflow<nsw> : i64
      %85 = arith.addi %83, %84 : i64
      %86 = arith.muli %85, %c32_i64 : i64
      %87 = arith.cmpi ugt, %81, %c0_i64 : i64
      scf.if %87 {
        asccube.get_buf fixpipe, %c8_i64, 0 {cannir.buffer.generated_sync}
        %88 = arith.trunci %81 : i64 to i32
        %89 = arith.index_cast %86 : i64 to index
        asccube.copy_l0c2gm %alloc_6[%89], %82[%c0], %c128_i32, %88, %c384_i32, %c16_i32, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xf32>, memref<?xf32, 5>
        asccube.release_buf fixpipe, %c8_i64, 0 {cannir.buffer.generated_sync}
      }
      scf.yield %71#1, %71#2, %71#3 : i64, i64, i64
    }
    asccube.sync_all
    asccube.sync_block_wait s, %c0_i64
    asccube.sync_block_arrive fixpipe, %c1_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c1_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c2_i64
    %41 = cannir.extract_layout %arg23 : !cannir.tensor<<gm, bf16>, <<(-1, 128)>, <(128, 1)>>, <<(-1, 128)>, <(128, 1)>>, identity>
    %42 = cannir.extract_shape %41 : <<(-1, 128)>, <(128, 1)>> -> <(-1, 128)>
    %43 = cannir.extract_dim %42[0] : !cannir.shape<(-1, 128)>
    %44 = arith.remsi %43, %c2_i64 : i64
    %45 = arith.cmpi ne, %44, %c0_i64 : i64
    %46 = arith.cmpi slt, %44, %c0_i64 : i64
    %47 = arith.andi %45, %46 : i1
    %48 = arith.divsi %43, %c2_i64 : i64
    %49 = arith.extui %47 : i1 to i64
    %50 = arith.subi %48, %49 : i64
    %51 = arith.addi %50, %c15_i64 : i64
    %52 = arith.remsi %51, %c16_i64 : i64
    %53 = arith.cmpi ne, %52, %c0_i64 : i64
    %54 = arith.cmpi slt, %52, %c0_i64 : i64
    %55 = arith.andi %53, %54 : i1
    %56 = arith.divsi %51, %c16_i64 : i64
    %57 = arith.extui %55 : i1 to i64
    %58 = arith.subi %56, %57 : i64
    %59 = arith.muli %58, %c2_i64 : i64
    %60 = scf.for %arg33 = %0 to %59 step %1 iter_args(%arg34 = %c0_i64) -> (i64)  : i64 {
      %62 = arith.remsi %arg33, %c2_i64 : i64
      %63 = arith.cmpi ne, %62, %c0_i64 : i64
      %64 = arith.cmpi slt, %62, %c0_i64 : i64
      %65 = arith.andi %63, %64 : i1
      %66 = arith.divsi %arg33, %c2_i64 : i64
      %67 = arith.extui %65 : i1 to i64
      %68 = arith.subi %66, %67 : i64
      %69 = arith.select %65, %c2_i64, %c0_i64 : i64
      %70 = arith.addi %62, %69 overflow<nsw> : i64
      %71 = arith.cmpi sge, %arg34, %c2_i64 : i64
      scf.if %71 {
        asccube.sync_block_wait s, %c4_i64
      }
      asccube.sync_block_wait s, %c5_i64
      %72 = scf.for %arg35 = %c0_i64 to %c2_i64 step %c1_i64 iter_args(%arg36 = %c0_i64) -> (i64)  : i64 {
        %74 = arith.divsi %50, %c16_i64 : i64
        %75 = arith.minsi %68, %74 : i64
        %76 = arith.muli %75, %c16_i64 : i64
        %77 = arith.subi %50, %76 : i64
        %78 = arith.minsi %77, %c16_i64 : i64
        %79 = arith.cmpi sle, %68, %74 : i64
        %80 = arith.select %79, %78, %c0_i64 : i64
        %81 = arith.cmpi eq, %arg36, %c0_i64 : i64
        %82 = ascscalar.memref_address %alloc_22 : memref<4096xf32, 5>
        %83 = ascscalar.memref_address %alloc_21 : memref<4096xf32, 5>
        %84 = arith.select %81, %82, %83 : i64
        %85 = arith.select %81, %c8_i64, %c9_i64 : i64
        %86 = arith.addi %arg36, %c1_i64 overflow<nsw> : i64
        %87 = arith.remui %86, %c2_i64 : i64
        %88 = arith.addi %80, %c15_i64 : i64
        %89 = arith.remsi %88, %c16_i64 : i64
        %90 = arith.cmpi ne, %89, %c0_i64 : i64
        %91 = arith.cmpi slt, %89, %c0_i64 : i64
        %92 = arith.andi %90, %91 : i1
        %93 = arith.divsi %88, %c16_i64 : i64
        %94 = arith.extui %92 : i1 to i64
        %95 = arith.subi %93, %94 : i64
        %96 = arith.muli %95, %c256_i64 : i64
        %97 = scf.for %arg37 = %c0_i64 to %c2_i64 step %c1_i64 iter_args(%arg38 = %c0_i64) -> (i64)  : i64 {
          %109 = arith.cmpi eq, %arg38, %c0_i64 : i64
          %110 = ascscalar.memref_address %alloc_20 : memref<1024xbf16, 2>
          %111 = ascscalar.memref_address %alloc_19 : memref<1024xbf16, 2>
          %112 = arith.select %109, %110, %111 : i64
          %113 = arith.cmpi ne, %arg38, %c0_i64 : i64
          %114 = arith.extui %113 : i1 to i64
          %115 = arith.addi %arg38, %c1_i64 overflow<nsw> : i64
          %116 = arith.remui %115, %c2_i64 : i64
          %117 = arith.muli %70, %c2_i64 overflow<nsw> : i64
          %118 = arith.addi %117, %arg37 overflow<nsw> : i64
          %119 = arith.divsi %80, %c16_i64 : i64
          %120 = arith.remsi %80, %c16_i64 : i64
          %121 = arith.addi %120, %c15_i64 overflow<nsw> : i64
          %122 = arith.divsi %121, %c16_i64 : i64
          %123 = arith.addi %119, %122 : i64
          %124 = ascscalar.bind_local_ptr %112 : memref<1xbf16, 2>
          %125 = arith.muli %68, %c256_i64 : i64
          %126 = arith.muli %118, %c4_i64 overflow<nsw> : i64
          %127 = arith.addi %125, %126 : i64
          %128 = arith.muli %127, %c16_i64 : i64
          asccube.get_buf mte2, %114, 0 {cannir.buffer.generated_sync}
          %129 = arith.cmpi ugt, %80, %c0_i64 : i64
          scf.if %129 {
            %160 = arith.index_cast %128 : i64 to index
            %161 = arith.trunci %80 : i64 to i32
            asccube.copy_gm2l1 %124[%c0], %alloc_7[%160], %161, %c64_i32, %c512_i64, %c1_i32, %c0_i64, %c0_i64, %c16_i32 l2_cache_ctl %c0_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<1xbf16>
          }
          asccube.release_buf mte2, %114, 0 {cannir.buffer.generated_sync}
          %130 = ascscalar.memref_address %alloc_18 : memref<16384xbf16, 2>
          %131 = ascscalar.memref_address %alloc_17 : memref<16384xbf16, 2>
          %132 = arith.select %109, %130, %131 : i64
          %133 = arith.select %109, %c2_i64, %c3_i64 : i64
          %134 = ascscalar.bind_local_ptr %132 : memref<1xbf16, 2>
          %135 = arith.muli %70, %c65536_i64 overflow<nsw> : i64
          %136 = arith.index_cast %135 : i64 to index
          %137 = arith.muli %arg37, %c2048_i64 overflow<nsw> : i64
          %138 = arith.muli %arg35, %c16_i64 overflow<nsw> : i64
          %139 = arith.addi %137, %138 overflow<nsw> : i64
          %140 = arith.muli %139, %c16_i64 overflow<nsw> : i64
          %141 = arith.index_cast %140 : i64 to index
          %142 = arith.addi %136, %141 : index
          asccube.get_buf mte2, %133, 0 {cannir.buffer.generated_sync}
          asccube.copy_gm2l1 %134[%c0], %alloc_3[%142], %c64_i32, %c256_i32, %c1024_i64, %c1_i32, %c0_i64, %c0_i64, %c64_i32 l2_cache_ctl %c4_i32 {nd2nz = true} : memref<1xbf16, 2>, memref<131072xbf16>
          asccube.release_buf mte2, %133, 0 {cannir.buffer.generated_sync}
          %143 = ascscalar.memref_address %alloc_16 : memref<1024xbf16, 3>
          %144 = ascscalar.memref_address %alloc_15 : memref<1024xbf16, 3>
          %145 = arith.select %109, %143, %144 : i64
          %146 = arith.select %109, %c4_i64, %c5_i64 : i64
          %147 = arith.muli %123, %c256_i64 : i64
          %148 = ascscalar.bind_local_ptr %145 : memref<1xbf16, 3>
          %149 = arith.index_cast %147 : i64 to index
          %150 = arith.divui %149, %c256 : index
          %151 = arith.index_cast %150 : index to i64
          asccube.get_buf mte1, %146, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %114, 0 {cannir.buffer.generated_sync}
          %152 = arith.trunci %151 : i64 to i32
          asccube.copy_l12l0a %148[%c0], %124[%c0], %c0_i32, %c0_i32, %c1_i32, %c4_i32, %c1_i32, %152 : memref<1xbf16, 3>, memref<1xbf16, 2>
          asccube.release_buf mte1, %114, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %146, 0 {cannir.buffer.generated_sync}
          %153 = ascscalar.memref_address %alloc_14 : memref<16384xbf16, 4>
          %154 = ascscalar.memref_address %alloc_13 : memref<16384xbf16, 4>
          %155 = arith.select %109, %153, %154 : i64
          %156 = arith.select %109, %c6_i64, %c7_i64 : i64
          %157 = ascscalar.bind_local_ptr %155 : memref<1xbf16, 4>
          asccube.get_buf mte1, %156, 0 {cannir.buffer.generated_sync}
          asccube.get_buf mte1, %133, 0 {cannir.buffer.generated_sync}
          asccube.copy_l12l0b %157[%c0], %134[%c0], %c0_i32, %c0_i32, %c4_i32, %c16_i32, %c4_i32, %c16_i32 {transpose = true} : memref<1xbf16, 4>, memref<1xbf16, 2>
          asccube.release_buf mte1, %133, 0 {cannir.buffer.generated_sync}
          asccube.release_buf mte1, %156, 0 {cannir.buffer.generated_sync}
          %158 = arith.cmpi eq, %arg37, %c0_i64 : i64
          %159 = ascscalar.bind_local_ptr %84 : memref<1xf32, 5>
          scf.if %129 {
            asccube.get_buf m, %85, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %146, 0 {cannir.buffer.generated_sync}
            asccube.get_buf m, %156, 0 {cannir.buffer.generated_sync}
            scf.if %158 {
              %160 = arith.trunci %80 : i64 to i32
              asccube.mmad %159[%c0], %148, %157, %160, %c64_i32, %c256_i32, %c0_i32 {c_matrix_init_val = true} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
            } else {
              %160 = arith.trunci %80 : i64 to i32
              asccube.mmad %159[%c0], %148, %157, %160, %c64_i32, %c256_i32, %c0_i32 {c_matrix_init_val = false} : memref<1xf32, 5>, memref<1xbf16, 3>, memref<1xbf16, 4>
            }
            asccube.release_buf m, %156, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %146, 0 {cannir.buffer.generated_sync}
            asccube.release_buf m, %85, 0 {cannir.buffer.generated_sync}
          }
          scf.yield %116 : i64
        }
        %98 = ascscalar.bind_local_ptr %84 : memref<1xf32, 5>
        %99 = arith.muli %68, %c1152_i64 : i64
        %100 = arith.muli %70, %c36_i64 overflow<nsw> : i64
        %101 = arith.addi %99, %100 : i64
        %102 = arith.muli %arg35, %c16_i64 overflow<nsw> : i64
        %103 = arith.addi %101, %102 : i64
        %104 = arith.muli %103, %c16_i64 : i64
        %105 = arith.index_cast %96 : i64 to index
        %106 = arith.divui %105, %c16 : index
        %107 = arith.index_cast %106 : index to i64
        %108 = arith.cmpi ugt, %80, %c0_i64 : i64
        scf.if %108 {
          asccube.get_buf fixpipe, %85, 0 {cannir.buffer.generated_sync}
          %109 = arith.trunci %80 : i64 to i32
          %110 = arith.trunci %107 : i64 to i32
          %111 = arith.index_cast %104 : i64 to index
          asccube.copy_l0c2gm %alloc_8[%111], %98[%c0], %c256_i32, %109, %c1152_i32, %110, %c0_i32 l2_cache_ctl %c0_i32 {nz2nd_en = true, relu_en = false} : memref<1xbf16>, memref<1xf32, 5>
          asccube.release_buf fixpipe, %85, 0 {cannir.buffer.generated_sync}
        }
        scf.yield %87 : i64
      }
      asccube.sync_block_arrive fixpipe, %c3_i64
      %73 = arith.addi %arg34, %c1_i64 : i64
      scf.yield %73 : i64
    }
    %61 = arith.minsi %60, %c2_i64 : i64
    scf.for %arg33 = %c0_i64 to %61 step %c1_i64  : i64 {
      asccube.sync_block_wait s, %c4_i64
    }
    asccube.sync_all
    asccube.sync_block_wait s, %c0_i64
    asccube.sync_block_arrive fixpipe, %c1_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c1_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c2_i64
    scf.if %17 {
    } else {
      %62 = cannir.extract_layout %arg26 : !cannir.tensor<<gm, bf16>, <<(-1, 576)>, <(576, 1)>>, <<(-1, 576)>, <(576, 1)>>, identity>
      %63 = cannir.extract_shape %62 : <<(-1, 576)>, <(576, 1)>> -> <(-1, 576)>
      %64 = cannir.extract_dim %63[0] : !cannir.shape<(-1, 576)>
      %65 = arith.cmpi sge, %64, %c16_i64 : i64
      scf.if %65 {
        asccube.sync_all
        asccube.sync_block_wait s, %c0_i64
        asccube.sync_block_arrive fixpipe, %c1_i64 {mode = 0 : i32}
        asccube.sync_block_wait s, %c1_i64 {mode = 0 : i32}
        asccube.sync_block_arrive mte3, %c2_i64
      }
      asccube.sync_all
      asccube.sync_block_wait s, %c0_i64
      asccube.sync_block_arrive fixpipe, %c1_i64 {mode = 0 : i32}
      asccube.sync_block_wait s, %c1_i64 {mode = 0 : i32}
      asccube.sync_block_arrive mte3, %c2_i64
    }
    asccube.sync_all
    asccube.sync_block_wait s, %c0_i64
    asccube.sync_block_arrive fixpipe, %c1_i64 {mode = 0 : i32}
    asccube.sync_block_wait s, %c1_i64 {mode = 0 : i32}
    asccube.sync_block_arrive mte3, %c2_i64
    return
  }
}
