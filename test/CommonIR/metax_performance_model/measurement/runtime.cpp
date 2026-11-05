#include <cstdio>
#include <cstring>
#include <mc_runtime_api.h>

extern "C" {
const char *error_text(int code) {
  return mcGetErrorString(static_cast<mcError_t>(code));
}

int device_info(int device, char *output, unsigned capacity) {
  mcError_t err = mcSetDevice(device);
  if (err != mcSuccess)
    return err;
  mcDeviceProp_t p{};
  err = mcGetDeviceProperties(&p, device);
  if (err != mcSuccess)
    return err;
  int runtime = 0, driver = 0;
  if ((err = mcRuntimeGetVersion(&runtime)) != mcSuccess)
    return err;
  if ((err = mcDriverGetVersion(&driver)) != mcSuccess)
    return err;
  char uuid[33]{};
  for (unsigned i = 0; i < 16; ++i)
    std::snprintf(uuid + 2 * i, 3, "%02x",
                  static_cast<unsigned char>(p.uuid.bytes[i]));
  std::snprintf(
      output, capacity,
      "{\"name\":\"%s\",\"uuid\":\"%s\",\"warp_size\":%d,\"visible_device\":%d,"
      "\"pci\":\"%04x:%02x:%02x\",\"runtime\":%d,\"driver\":%d,\"clock_khz\":%"
      "d}",
      p.name, uuid, p.warpSize, device, p.pciDomainID, p.pciBusID,
      p.pciDeviceID, runtime, driver, p.clockRate);
  return mcSuccess;
}

int load_module(const void *image, void **module) {
  return mcModuleLoadData(reinterpret_cast<mcModule_t *>(module), image);
}
int unload_module(void *module) {
  return mcModuleUnload(static_cast<mcModule_t>(module));
}
int allocate(void **pointer, size_t size) { return mcMalloc(pointer, size); }
int release(void *pointer) { return mcFree(pointer); }

int launch(void *module, const char *symbol, void *input, void *output,
           const void *host_input, void *host_output, size_t size) {
  mcError_t err = mcMemcpy(input, host_input, size, mcMemcpyHostToDevice);
  if (err != mcSuccess)
    return err;
  if ((err = mcMemcpy(output, host_output, size, mcMemcpyHostToDevice)) !=
      mcSuccess)
    return err;
  mcFunction_t function{};
  if ((err = mcModuleGetFunction(&function, static_cast<mcModule_t>(module),
                                 symbol)) != mcSuccess)
    return err;
  struct {
    void *in;
    void *out;
  } args{input, output};
  size_t length = sizeof(args);
  void *extra[] = {reinterpret_cast<void *>(1), &args,
                   reinterpret_cast<void *>(2), &length, nullptr};
  if ((err = mcModuleLaunchKernel(function, 1, 1, 1, 64, 1, 1, 0, nullptr,
                                  nullptr, extra)) != mcSuccess)
    return err;
  if ((err = mcDeviceSynchronize()) != mcSuccess)
    return err;
  return mcMemcpy(host_output, output, size, mcMemcpyDeviceToHost);
}
}
