#include "../measurement/runtime.cpp"

extern "C" int launch_layout_gemm(void *module, void *input, void *output,
                                  const void *host_input, void *host_output,
                                  size_t size, unsigned shared) {
  mcError_t err = mcMemcpy(input, host_input, size, mcMemcpyHostToDevice);
  if (err != mcSuccess)
    return err;
  if ((err = mcMemset(output, 0xa5, size)) != mcSuccess)
    return err;
  mcFunction_t function{};
  if ((err = mcModuleGetFunction(&function, static_cast<mcModule_t>(module),
                                 "layout_gemm")) != mcSuccess)
    return err;
  struct {
    void *in;
    void *out;
    void *scratch;
    void *profile;
  } args{input, output, nullptr, nullptr};
  size_t length = sizeof(args);
  void *extra[] = {reinterpret_cast<void *>(1), &args,
                   reinterpret_cast<void *>(2), &length, nullptr};
  if ((err = mcModuleLaunchKernel(function, 1, 1, 1, 64, 1, 1, shared, nullptr,
                                  nullptr, extra)) != mcSuccess)
    return err;
  if ((err = mcDeviceSynchronize()) != mcSuccess)
    return err;
  return mcMemcpy(host_output, output, size, mcMemcpyDeviceToHost);
}
