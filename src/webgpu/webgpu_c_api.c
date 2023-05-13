/*!
 * \file webgpu/webgpu_c_api.c
 * \brief implement for webgpu sync c api
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#if USE_WEBGPU // USE_WEBGPU = 1

#ifdef __EMSCRIPTEN__
#include <webgpu/webgpu.h>
#else
#include <webgpu.h>
#endif // __EMSCRIPTEN__

#include <tvm/runtime/c_runtime_api.h>
#include <utils/common.h>
#include <webgpu/webgpu_c_api.h>

struct WGPU_Device_st {};

struct WGPU_Memory_st {};

struct WGPU_Function_st {};

static struct {
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPURequestAdapterStatus status;
} webGPU_Adapter;

static void adapter_request_callback(WGPURequestAdapterStatus status, WGPUAdapter adapter, char const *message,
                                     void *userdata) {
    webGPU_Adapter.adapter = adapter;
    webGPU_Adapter.status = status;
    if (unlikely(status)) {
        TVMAPISetLastError(message);
    }
}

static __attribute__((constructor)) void webgpu_adapter_constructor() {
    webGPU_Adapter.instance = NULL;
    webGPU_Adapter.adapter = NULL;
    webGPU_Adapter.status = WGPURequestAdapterStatus_Unknown;

    WGPUInstanceDescriptor desc = {
        .nextInChain = NULL,
    };
    WGPUInstance instance = wgpuCreateInstance(&desc);
    if (instance) {
        webGPU_Adapter.instance = instance;
    } else {
        // todo
    }

    WGPURequestAdapterOptions opts = {
        .powerPreference = WGPUPowerPreference_HighPerformance,
        .nextInChain = NULL,
        .compatibleSurface = NULL,
        .forceFallbackAdapter = NULL,
    };

    wgpuInstanceRequestAdapter(instance, &opts, adapter_request_callback, NULL);
}

#ifdef __EMSCRIPTEN__

static __attribute__((destructor)) void webgpu_adapter_destructor() {
    if (webGPU_Adapter.adapter) {
        wgpuAdapterRelease(webGPU_Adapter.adapter);
    }
    if (webGPU_Adapter.instance) {
        wgpuInstanceRelease(webGPU_Adapter.instance);
    }
}

#endif // __EMSCRIPTEN__

int WGPU_DeviceCount(int *count_ptr) { return 0; }

int WGPU_DeviceGet(WGPU_Device *device_ptr, int dev_id) { return 0; }

int WGPU_DeviceFree(WGPU_Device device) { return 0; }

int WGPU_DeviceSync(WGPU_Device device) { return 0; }

int WGPU_MemoryAlloc(WGPU_Memory *memory_ptr, size_t nbytes) { return 0; }

int WGPU_MemoryFree(WGPU_Memory memory) { return 0; }

int WGPU_MemoryCopyHtoD(WGPU_Memory dst, size_t dst_byte_offset, void *src, size_t src_byte_offset, size_t nbytes) {
    return 0;
}

int WGPU_MemoryCopyDtoH(void *dst, size_t dst_byte_offset, WGPU_Memory src, size_t src_byte_offset, size_t nbytes) {
    return 0;
}

int WGPU_MemoryCopyDtoD(WGPU_Memory dst, size_t dst_byte_offset, WGPU_Memory src, size_t src_byte_offset,
                        size_t nbytes) {
    return 0;
}

int WGPU_FunctionCreate(WGPU_Function *func_ptr, char *source) { return 0; }

int WGPU_FunctionRun(WGPU_Function function /* todo */) { return 0; }

int WGPU_FunctionFree(WGPU_Function function) { return 0; }

#endif // USE_WEBGPU = 1
