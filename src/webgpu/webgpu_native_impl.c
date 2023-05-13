/*!
 * \file webgpu/webgpu_native_impl.c
 * \brief implement for webgpu sync c api using native library.
 * \author YangBo MG21330067@smail.nju.edu.cn
 */

#if USE_WEBGPU && !defined(__EMSCRIPTEN__) // USE_WEBGPU = 1 && !defined(__EMSCRIPTEN__)

#include <device/cpu_memory.h>
#include <string.h>
#include <tvm/runtime/c_runtime_api.h>
#include <utils/common.h>
#include <webgpu.h>
#include <webgpu/webgpu_c_api.h>

typedef struct WGPUDeviceImpl WGPU_Device_st;

struct WGPU_Memory_st {
    WGPU_Device device;
    WGPUBuffer buffer;
};

typedef struct WGPUComputePipelineImpl WGPU_Function_st;

static struct {
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
    webGPU_Adapter.adapter = NULL;
    webGPU_Adapter.status = WGPURequestAdapterStatus_Unknown;

    WGPUInstanceDescriptor desc = {
        .nextInChain = NULL,
    };
    WGPUInstance instance = wgpuCreateInstance(&desc);
    if (!instance) {
        TVMAPISetLastError("Cannot create webgpu instance.");
        return;
    }

    WGPURequestAdapterOptions opts = {
        .powerPreference = WGPUPowerPreference_HighPerformance,
        .nextInChain = NULL,
        .compatibleSurface = NULL,
        .forceFallbackAdapter = NULL,
    };

    wgpuInstanceRequestAdapter(instance, &opts, adapter_request_callback, NULL);
}

static void device_request_callback(WGPURequestDeviceStatus status, WGPUDevice device, char const *message,
                                    void *userdata) {
    if (unlikely(status)) {
        TVMAPISetLastError(message);
    } else {
        *(WGPUDevice *)userdata = device;
    }
}

static void handle_device_lost(WGPUDeviceLostReason reason, char const *message, void *userdata) {
    TVMAPISetLastError(message);
}

static void handle_uncaptured_error(WGPUErrorType type, char const *message, void *userdata) {
    TVMAPISetLastError(message);
}

int WGPU_DeviceGet(WGPU_Device *device_ptr) {
    *device_ptr = NULL;
    if (unlikely(webGPU_Adapter.status != WGPURequestAdapterStatus_Success)) {
        SET_ERROR_RETURN((int)webGPU_Adapter.status, "Cannot get GPU Adapters\n");
    }

    wgpuAdapterRequestDevice(webGPU_Adapter.adapter, NULL, device_request_callback, (void *)device_ptr);
    if (unlikely(!*device_ptr)) {
        return -1;
    }

    wgpuDeviceSetUncapturedErrorCallback(*device_ptr, handle_uncaptured_error, NULL);
    wgpuDeviceSetDeviceLostCallback(*device_ptr, handle_device_lost, NULL);
    return 0;
}

inline int WGPU_DeviceFree(WGPU_Device device) {
    wgpuDeviceDestroy(device);
    return 0;
}

static int WGPU_DeviceSync(WGPUQueue queue) {
    fprintf(stderr, "%s %p\n", __func__, queue);
    return 0;
}

int WGPU_MemoryAlloc(WGPU_Device device, WGPU_Memory *memory_ptr, size_t nbytes) {
    const WGPUBufferDescriptor desc = {
        .nextInChain = NULL,
        .label = NULL,
        .usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc,
        .size = (uint64_t)nbytes,
        .mappedAtCreation = false,
    };

    WGPU_Memory ptr = wgpuDeviceCreateBuffer(device, &desc);
    if (unlikely(!ptr)) {
        TVMAPISetLastError("Alloca WebGPU memory fail!");
    }

    *memory_ptr = (WGPU_Memory)TVM_RT_WASM_HeapMemoryAlloc(sizeof(struct WGPU_Memory_st));
    (*memory_ptr)->device = device;
    (*memory_ptr)->buffer = ptr;
    return 0;
}

inline int WGPU_MemoryFree(WGPU_Memory memory) {
    wgpuBufferDestroy(memory->buffer);
    TVM_RT_WASM_HeapMemoryFree(memory);
    return 0;
}

int WGPU_MemoryCopyHtoD(WGPU_Memory dst, size_t dst_byte_offset, void *src, size_t src_byte_offset, size_t nbytes) {
    WGPUQueue q = wgpuDeviceGetQueue(dst->device);
    wgpuQueueWriteBuffer(q, dst->buffer, dst_byte_offset, src + src_byte_offset, nbytes);
    return 0;
}

int WGPU_MemoryCopyDtoH(void *dst, size_t dst_byte_offset, WGPU_Memory src, size_t src_byte_offset, size_t nbytes) {
    fprintf(stderr, "%s %p\n", __func__, dst);

    const WGPUBufferDescriptor desc = {
        .nextInChain = NULL,
        .label = NULL,
        .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
        .size = (uint64_t)nbytes,
        .mappedAtCreation = false,
    };
    WGPU_Memory dst_gpu = wgpuDeviceCreateBuffer(src->device, &desc);
    if (unlikely(!dst_gpu)) {
        TVMAPISetLastError("Alloca WebGPU memory fail!");
    }

    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(src->device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(command_encoder, src->buffer, src_byte_offset, dst_gpu, 0, nbytes);
    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(command_encoder, NULL);
    WGPUQueue q = wgpuDeviceGetQueue(src->device);
    wgpuQueueSubmit(q, 1, &command_buffer);

    /*  wait for tasks done.  */

    const void *dst_cpu = wgpuBufferGetConstMappedRange(dst_gpu, 0, nbytes);
    memcpy(dst + dst_byte_offset, dst_cpu, nbytes);
    wgpuBufferUnmap(dst_gpu);
    wgpuBufferDestroy(dst_gpu);
    return 0;
}

int WGPU_MemoryCopyDtoD(WGPU_Memory dst, size_t dst_byte_offset, WGPU_Memory src, size_t src_byte_offset,
                        size_t nbytes) {
    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(src->device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(command_encoder, src->buffer, src_byte_offset, dst->buffer, dst_byte_offset,
                                         nbytes);
    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(command_encoder, NULL);
    WGPUQueue q = wgpuDeviceGetQueue(src->device);
    wgpuQueueSubmit(q, 1, &command_buffer);
    return 0;
}

int WGPU_FunctionCreate(WGPU_Function *func_ptr, char *source) {
    fprintf(stderr, "%s %d\n", __func__, strlen(source));
    return 0;
}

int WGPU_FunctionRun(WGPU_Function function, size_t block_dims[3], size_t thread_dims[3], WGPU_Memory kernel_args[],
                     uint32_t num_kernel_args) {
    fprintf(stderr, "run %p, blocks=%d,%d,%d, threads=%d,%d,%d, num_kernel_args=%d\n", function, block_dims[0],
            block_dims[1], block_dims[2], thread_dims[0], thread_dims[1], thread_dims[2], num_kernel_args);
    return 0;
}

int WGPU_FunctionFree(WGPU_Function function) {
    fprintf(stderr, "%s %p\n", __func__, function);
    return 0;
}

#endif // // USE_WEBGPU = 1 && !defined(__EMSCRIPTEN__)
