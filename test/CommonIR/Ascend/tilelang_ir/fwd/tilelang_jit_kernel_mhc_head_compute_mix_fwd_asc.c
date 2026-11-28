#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(256) inline void mhc_head_compute_mix_fwd_asc_kernel_simt_vf_0(__ubuf__ float* input_ub, int32_t input_ub_version_counter_1, int32_t w, int32_t num_tokens, __gm__ float* output_mix, __gm__ float* mhc_scale, __gm__ float* mhc_base) {
  float input_frag[1];
  input_frag[0] = input_ub[(((input_ub_version_counter_1 & 1) * 128) + (((int32_t)threadIdx.x) & 127))];
  if ((((int32_t)threadIdx.x) >> 7) == 0) {
    if ((((((((w * 20) + ((int32_t)block_idx)) / max(1, min(8, ((num_tokens + 31) >> 5)))) * max(1, min(8, ((num_tokens + 31) >> 5)))) * 32) + ((((w * 20) + ((int32_t)block_idx)) % max(1, min(8, ((num_tokens + 31) >> 5)))) * 32)) + ((((int32_t)threadIdx.x) & 127) >> 2)) < num_tokens) {
      output_mix[(((((((((int64_t)w) * (int64_t)20) + ((int64_t)((int32_t)block_idx))) / max((int64_t)1, min((int64_t)8, ((((int64_t)num_tokens) + (int64_t)31) >> (int64_t)5)))) * max((int64_t)1, min((int64_t)8, ((((int64_t)num_tokens) + (int64_t)31) >> (int64_t)5)))) * (int64_t)128) + ((((((int64_t)w) * (int64_t)20) + ((int64_t)((int32_t)block_idx))) % max((int64_t)1, min((int64_t)8, ((((int64_t)num_tokens) + (int64_t)31) >> (int64_t)5)))) * (int64_t)128)) + (((int64_t)((int32_t)threadIdx.x)) & (int64_t)127))] = ((float(0x1p+0f/*1.000000e+00*/) / (float(0x1p+0f/*1.000000e+00*/) + expf((float(0x0p+0f/*0.000000e+00*/) - ((input_frag[0] * mhc_scale[0]) + mhc_base[(((int32_t)threadIdx.x) & 3)]))))) + float(0x1.0c6f7a0b5ed8dp-20f/*1.000000e-06*/));
    }
  }
}

extern "C" __global__ __vector__ void mhc_head_compute_mix_fwd_asc_kernel(__gm__ float* input_mix, __gm__ float* mhc_base, __gm__ float* mhc_scale, __gm__ float* output_mix, int32_t num_tokens) {
  asc_init();
  __ubuf__ float *input_ub = (__ubuf__ float *)0;
  int32_t input_ub_version_counter_1 = 0;
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
  input_ub_version_counter_1 = 0;
  for (int32_t w = 0; w < ((((((((((num_tokens + 31) >> 5) - 1) / max(1, min(8, ((num_tokens + 31) >> 5)))) + (((((num_tokens + 31) >> 5) - 1) % max(1, min(8, ((num_tokens + 31) >> 5)))) >> 31)) + 1) * max(1, min(8, ((num_tokens + 31) >> 5)))) + 19) / 20) + ((((((((((num_tokens + 31) >> 5) - 1) / max(1, min(8, ((num_tokens + 31) >> 5)))) + (((((num_tokens + 31) >> 5) - 1) % max(1, min(8, ((num_tokens + 31) >> 5)))) >> 31)) + 1) * max(1, min(8, ((num_tokens + 31) >> 5)))) + 19) % 20) >> 31)); ++w) {
    if ((((((w * 20) + ((int32_t)block_idx)) / max(1, min(8, ((num_tokens + 31) >> 5)))) * max(1, min(8, ((num_tokens + 31) >> 5)))) + (((w * 20) + ((int32_t)block_idx)) % max(1, min(8, ((num_tokens + 31) >> 5))))) < ((num_tokens + 31) >> 5)) {
      asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>((input_ub_version_counter_1 & 1)));
      asc_copy_gm2ub_align((__ubuf__ uint8_t*)((&(input_ub[((input_ub_version_counter_1 & 1) * 128)]))), (__gm__ uint8_t*)((&(input_mix[((((((((int64_t)w) * (int64_t)20) + ((int64_t)((int32_t)block_idx))) / max((int64_t)1, min((int64_t)8, ((((int64_t)num_tokens) + (int64_t)31) >> (int64_t)5)))) * max((int64_t)1, min((int64_t)8, ((((int64_t)num_tokens) + (int64_t)31) >> (int64_t)5)))) * (int64_t)128) + ((((((int64_t)w) * (int64_t)20) + ((int64_t)((int32_t)block_idx))) % max((int64_t)1, min((int64_t)8, ((((int64_t)num_tokens) + (int64_t)31) >> (int64_t)5)))) * (int64_t)128))]))), 1, (min(32, max(((num_tokens - (((((w * 20) + ((int32_t)block_idx)) / max(1, min(8, ((num_tokens + 31) >> 5)))) * max(1, min(8, ((num_tokens + 31) >> 5)))) * 32)) - ((((w * 20) + ((int32_t)block_idx)) % max(1, min(8, ((num_tokens + 31) >> 5)))) * 32)), 0)) * 16), 0, 0, 0, static_cast<asc_load_l2_cache_mode>(0), (min(32, max(((num_tokens - (((((w * 20) + ((int32_t)block_idx)) / max(1, min(8, ((num_tokens + 31) >> 5)))) * max(1, min(8, ((num_tokens + 31) >> 5)))) * 32)) - ((((w * 20) + ((int32_t)block_idx)) % max(1, min(8, ((num_tokens + 31) >> 5)))) * 32)), 0)) * 16), (min(32, max(((num_tokens - (((((w * 20) + ((int32_t)block_idx)) / max(1, min(8, ((num_tokens + 31) >> 5)))) * max(1, min(8, ((num_tokens + 31) >> 5)))) * 32)) - ((((w * 20) + ((int32_t)block_idx)) % max(1, min(8, ((num_tokens + 31) >> 5)))) * 32)), 0)) * 16));
      asc_sync_notify(PIPE_MTE2, PIPE_V, static_cast<event_t>((input_ub_version_counter_1 & 1)));
      asc_sync_wait(PIPE_MTE2, PIPE_V, static_cast<event_t>((input_ub_version_counter_1 & 1)));
      asc_vf_call<mhc_head_compute_mix_fwd_asc_kernel_simt_vf_0>(cce::dim3(256), input_ub, input_ub_version_counter_1, w, num_tokens, output_mix, mhc_scale, mhc_base);
      asc_sync_notify(PIPE_V, PIPE_MTE2, static_cast<event_t>((input_ub_version_counter_1 & 1)));
      input_ub_version_counter_1 = (input_ub_version_counter_1 + 1);
    }
  }
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(0));
  asc_sync_wait(PIPE_V, PIPE_MTE2, static_cast<event_t>(1));
}


