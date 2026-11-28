#include <tl_templates/ascend/common.h>
#include <tl_templates/ascend/debug.h>
__simt_vf__ __launch_bounds__(128) inline void mhc_head_compute_mix_bwd_asc_kernel_simt_vf_0(__ubuf__ uint8_t* buf_dyn_shmem, int32_t num_tokens, __gm__ float* input_mix, __gm__ float* mhc_scale, __gm__ float* mhc_base, __gm__ float* output_mix_grad, __gm__ float* input_mix_grad, __gm__ float* mhc_scale_grad_partial, __gm__ float* mhc_base_grad_partial) {
  float mhc_scale_grad_reducer[1];
  float mhc_base_grad_reducer[1];
  float mhc_scale_grad_frag[1];
  float mhc_base_grad_frag[1];
  float grad_frag[1];
  float input_recompute_frag[1];
  mhc_scale_grad_reducer[0] = float(0x0p+0f/*0.000000e+00*/);
  mhc_base_grad_reducer[0] = float(0x0p+0f/*0.000000e+00*/);
  for (int32_t w = 0; w < ((num_tokens + 639) / 640); ++w) {
    if (((w * 20) + ((int32_t)block_idx)) < ((num_tokens + 31) >> 5)) {
      if ((((w * 640) + (((int32_t)block_idx) * 32)) + (((int32_t)threadIdx.x) >> 2)) < num_tokens) {
        input_recompute_frag[0] = (float(0x1p+0f/*1.000000e+00*/) / (float(0x1p+0f/*1.000000e+00*/) + expf((float(0x0p+0f/*0.000000e+00*/) - ((input_mix[(((((int64_t)w) * (int64_t)2560) + (((int64_t)((int32_t)block_idx)) * (int64_t)128)) + ((int64_t)((int32_t)threadIdx.x)))] * mhc_scale[0]) + mhc_base[(((int32_t)threadIdx.x) & 3)])))));
        grad_frag[0] = ((input_recompute_frag[0] * (float(0x1p+0f/*1.000000e+00*/) - input_recompute_frag[0])) * output_mix_grad[(((((int64_t)w) * (int64_t)2560) + (((int64_t)((int32_t)block_idx)) * (int64_t)128)) + ((int64_t)((int32_t)threadIdx.x)))]);
        input_mix_grad[(((((int64_t)w) * (int64_t)2560) + (((int64_t)((int32_t)block_idx)) * (int64_t)128)) + ((int64_t)((int32_t)threadIdx.x)))] = (grad_frag[0] * mhc_scale[0]);
        mhc_scale_grad_reducer[0] = (mhc_scale_grad_reducer[0] + (grad_frag[0] * input_mix[(((((int64_t)w) * (int64_t)2560) + (((int64_t)((int32_t)block_idx)) * (int64_t)128)) + ((int64_t)((int32_t)threadIdx.x)))]));
        mhc_base_grad_reducer[0] = (mhc_base_grad_reducer[0] + grad_frag[0]);
      }
    }
  }
  mhc_scale_grad_reducer[0] = tl::AscendAllReduce<tl::SumOp, 128, 4, 0>::run(mhc_scale_grad_reducer[0], (&(((__ubuf__ float*)buf_dyn_shmem)[0])));
  mhc_scale_grad_reducer[0] = tl::AscendAllReduce<tl::SumOp, 4, 1, 0>::run(mhc_scale_grad_reducer[0]);
  mhc_scale_grad_frag[0] = mhc_scale_grad_reducer[0];
  mhc_base_grad_reducer[0] = tl::AscendAllReduce<tl::SumOp, 128, 4, 0>::run(mhc_base_grad_reducer[0], (&(((__ubuf__ float*)buf_dyn_shmem)[128])));
  mhc_base_grad_frag[0] = mhc_base_grad_reducer[0];
  if (((int32_t)threadIdx.x) == 0) {
    mhc_scale_grad_partial[((int32_t)block_idx)] = mhc_scale_grad_frag[0];
  }
  if ((((int32_t)threadIdx.x) >> 2) == 0) {
    mhc_base_grad_partial[((((int32_t)block_idx) * 4) + (((int32_t)threadIdx.x) & 3))] = mhc_base_grad_frag[0];
  }
}

extern "C" __global__ __vector__ void mhc_head_compute_mix_bwd_asc_kernel(__gm__ float* input_mix, __gm__ float* input_mix_grad, __gm__ float* mhc_base, __gm__ float* mhc_base_grad_partial, __gm__ float* mhc_scale, __gm__ float* mhc_scale_grad_partial, __gm__ float* output_mix_grad, int32_t num_tokens) {
  asc_init();
  __ubuf__ uint8_t *buf_dyn_shmem = (__ubuf__ uint8_t *)0;
  asc_vf_call<mhc_head_compute_mix_bwd_asc_kernel_simt_vf_0>(cce::dim3(128), buf_dyn_shmem, num_tokens, input_mix, mhc_scale, mhc_base, output_mix_grad, input_mix_grad, mhc_scale_grad_partial, mhc_base_grad_partial);
}


