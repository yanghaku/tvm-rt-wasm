/*!
 * \file webgpu/native_impl/webgpu_native_impl.c
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

// wgpu_native functions
typedef struct WGPUWrappedSubmissionIndex WGPUWrappedSubmissionIndex;
extern bool wgpuDevicePoll(WGPUDevice device, bool wait, WGPUWrappedSubmissionIndex const *wrappedSubmissionIndex);
extern void wgpuInstanceDrop(WGPUInstance instance);
extern void wgpuAdapterDrop(WGPUAdapter adapter);
extern void wgpuBindGroupDrop(WGPUBindGroup bindGroup);
extern void wgpuShaderModuleDrop(WGPUShaderModule shader_module);
extern void wgpuComputePipelineDrop(WGPUComputePipeline compute_pipeline);
extern void wgpuAddTasksToQueue(WGPUDevice device);

typedef struct WGPUDeviceImpl WGPU_Device_st;

struct WGPU_Memory_st {
    WGPU_Device device;
    WGPUBuffer buffer;
    uint64_t size;
};

struct WGPU_Function_st {
    WGPU_Device device;
    WGPUComputePipeline compute_pipeline;
    WGPUBindGroupEntry *bind_group_entries;
    uint32_t bind_group_entries_count;
};

static struct {
    WGPUInstance instance;
    WGPUAdapter adapter;
    WGPURequestAdapterStatus status;
} webGPU_Adapter;

static __attribute__((destructor)) void adapter_destructor() {
    if (webGPU_Adapter.adapter) {
        wgpuAdapterDrop(webGPU_Adapter.adapter);
    }
    if (webGPU_Adapter.instance) {
        wgpuInstanceDrop(webGPU_Adapter.instance);
    }
}

static void adapter_request_callback(WGPURequestAdapterStatus status, WGPUAdapter adapter, char const *message,
                                     void *userdata) {
    webGPU_Adapter.adapter = adapter;
    webGPU_Adapter.status = status;
    if (unlikely(status)) {
        TVMAPISetLastError(message);
    }
}

static void webgpu_adapter_constructor() {
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

static void device_lost_callback(WGPUDeviceLostReason reason, char const *message, void *userdata) {
    TVMAPISetLastError(message);
}

static void uncaptured_error_callback(WGPUErrorType type, char const *message, void *userdata) {
    fprintf(stderr, "WebGPU: %s\n", message);
    TVMAPISetLastError(message);
}

static void buffer_map_callback(WGPUBufferMapAsyncStatus status, void *userdata) {
    // do nothing
}

int WGPU_DeviceGet(WGPU_Device *device_ptr) {
    if (webGPU_Adapter.adapter == NULL) {
        webGPU_Adapter.status = WGPURequestAdapterStatus_Unknown;
        webgpu_adapter_constructor();
    }

    *device_ptr = NULL;
    if (unlikely(webGPU_Adapter.status != WGPURequestAdapterStatus_Success)) {
        SET_ERROR_RETURN((int)webGPU_Adapter.status, "Cannot get GPU Adapters\n");
    }

    WGPURequiredLimits limits;
    // use the adapter limits as requested limits.
    wgpuAdapterGetLimits(webGPU_Adapter.adapter, &limits);
    // todo: check limits.

    WGPUDeviceDescriptor device_desc;
    memset(&device_desc, 0, sizeof(WGPUDeviceDescriptor));
    device_desc.requiredLimits = &limits;
    wgpuAdapterRequestDevice(webGPU_Adapter.adapter, &device_desc, device_request_callback, (void *)device_ptr);
    if (unlikely(!*device_ptr)) {
        return -1;
    }

    // if use the dawn, set the submit done callback function.
    wgpuDevicePoll(*device_ptr, true, NULL);

    wgpuDeviceSetUncapturedErrorCallback(*device_ptr, uncaptured_error_callback, NULL);
    wgpuDeviceSetDeviceLostCallback(*device_ptr, device_lost_callback, NULL);
    return 0;
}

inline int WGPU_DeviceFree(WGPU_Device device) {
    wgpuDeviceDestroy(device);
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
        return -1;
    }

    *memory_ptr = (WGPU_Memory)TVM_RT_WASM_HeapMemoryAlloc(sizeof(struct WGPU_Memory_st));
    (*memory_ptr)->device = device;
    (*memory_ptr)->buffer = ptr;
    (*memory_ptr)->size = (uint64_t)nbytes;
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
    const WGPUBufferDescriptor desc = {
        .nextInChain = NULL,
        .label = NULL,
        .usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst,
        .size = (uint64_t)nbytes,
        .mappedAtCreation = false,
    };
    WGPU_Memory dst_gpu = wgpuDeviceCreateBuffer(src->device, &desc);
    if (unlikely(!dst_gpu)) {
        return -1;
    }

    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(src->device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(command_encoder, src->buffer, src_byte_offset, dst_gpu, 0, nbytes);
    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(command_encoder, NULL);

    wgpuAddTasksToQueue(src->device);
    WGPUQueue q = wgpuDeviceGetQueue(src->device);
    wgpuQueueSubmit(q, 1, &command_buffer);

    wgpuBufferMapAsync(dst_gpu, WGPUMapMode_Read, 0, nbytes, buffer_map_callback, NULL);

    /*  wait for tasks done.  */
    wgpuDevicePoll(src->device, true, NULL);

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

    wgpuAddTasksToQueue(src->device);
    WGPUQueue q = wgpuDeviceGetQueue(src->device);
    wgpuQueueSubmit(q, 1, &command_buffer);
    return 0;
}

int WGPU_FunctionCreate(WGPU_Device device, WGPU_Function *func_ptr, char *source, uint32_t source_len,
                        uint32_t num_kernel_args) {
    // todo: avoid copy
    char *source_code = (char *)TVM_RT_WASM_WorkplaceMemoryAlloc(source_len + 1);
    memcpy(source_code, source, source_len);
    source_code[source_len] = '\0';

    // 1. parse source device code and create shader.
    const WGPUShaderModuleWGSLDescriptor wgsl_desc = {
        .code = source_code,
        .chain =
            (const WGPUChainedStruct){
                .next = NULL,
                .sType = WGPUSType_ShaderModuleWGSLDescriptor,
            },
    };
    const WGPUShaderModuleDescriptor shader_module_desc = {
        .nextInChain = (const WGPUChainedStruct *)&wgsl_desc,
        .label = NULL,
        .hintCount = 0,
        .hints = NULL,
    };
    WGPUShaderModule shader_module = wgpuDeviceCreateShaderModule(device, &shader_module_desc);
    // TVM_RT_WASM_WorkplaceMemoryFree(source_code);
    if (unlikely(!shader_module)) {
        return -1;
    }
    const WGPUProgrammableStageDescriptor compute_desc = {
        .nextInChain = NULL,
        .constants = NULL,
        .constantCount = 0,
        .module = shader_module,
        .entryPoint = "main",
    };

    // 2. create the pipeline
    const WGPUComputePipelineDescriptor pipeline_desc = {
        .nextInChain = NULL,
        .label = NULL,
        .layout = NULL,
        .compute = compute_desc,
    };
    WGPUComputePipeline compute_pipeline = wgpuDeviceCreateComputePipeline(device, &pipeline_desc);
    if (unlikely(!compute_pipeline)) {
        wgpuShaderModuleDrop(shader_module);
        return -1;
    }

    *func_ptr = (WGPU_Function)TVM_RT_WASM_HeapMemoryAlloc(sizeof(struct WGPU_Function_st));
    (*func_ptr)->device = device;
    (*func_ptr)->compute_pipeline = compute_pipeline;
    (*func_ptr)->bind_group_entries_count = num_kernel_args;

    // create cached bind group entries
    WGPUBindGroupEntry *bind_group_entries =
        (WGPUBindGroupEntry *)TVM_RT_WASM_HeapMemoryAlloc(sizeof(WGPUBindGroupEntry) * num_kernel_args);
    memset(bind_group_entries, 0, sizeof(WGPUBindGroupEntry) * num_kernel_args);
    for (uint32_t i = 0; i < num_kernel_args; ++i) {
        bind_group_entries[i].binding = i;
        bind_group_entries[i].offset = 0;
    }
    (*func_ptr)->bind_group_entries = bind_group_entries;

    wgpuShaderModuleDrop(shader_module);
    return 0;
}

int WGPU_FunctionRun(WGPU_Function function, size_t grid_dims[3], size_t block_dims[3], WGPU_Memory kernel_args[],
                     uint32_t num_kernel_args) {
    for (uint32_t i = 0; i < num_kernel_args; ++i) {
        function->bind_group_entries[i].buffer = kernel_args[i]->buffer;
        function->bind_group_entries[i].size = kernel_args[i]->size;
    }
    WGPUBindGroupLayout bind_group_layout = wgpuComputePipelineGetBindGroupLayout(function->compute_pipeline, 0);
    if (unlikely(!bind_group_layout)) {
        return -1;
    }
    const WGPUBindGroupDescriptor bind_group_desc = {
        .label = NULL,
        .layout = bind_group_layout,
        .entryCount = num_kernel_args,
        .entries = function->bind_group_entries,
    };
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(function->device, &bind_group_desc);

    WGPUCommandEncoder command_encoder = wgpuDeviceCreateCommandEncoder(function->device, NULL);
    WGPUComputePassEncoder compute_pass_encoder = wgpuCommandEncoderBeginComputePass(command_encoder, NULL);
    wgpuComputePassEncoderSetPipeline(compute_pass_encoder, function->compute_pipeline);
    wgpuComputePassEncoderSetBindGroup(compute_pass_encoder, 0, bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass_encoder, block_dims[0], block_dims[1], block_dims[2]);
    wgpuComputePassEncoderEnd(compute_pass_encoder);
    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(command_encoder, NULL);

    wgpuAddTasksToQueue(function->device);
    WGPUQueue q = wgpuDeviceGetQueue(function->device);
    wgpuQueueSubmit(q, 1, &command_buffer);

    wgpuBindGroupDrop(bind_group);
    return 0;
}

int WGPU_FunctionFree(WGPU_Function function) {
    if (function->compute_pipeline) {
        wgpuComputePipelineDrop(function->compute_pipeline);
    }
    if (function->bind_group_entries) {
        TVM_RT_WASM_HeapMemoryFree(function->bind_group_entries);
    }
    TVM_RT_WASM_HeapMemoryFree(function);
    return 0;
}

#endif // // USE_WEBGPU = 1 && !defined(__EMSCRIPTEN__)
